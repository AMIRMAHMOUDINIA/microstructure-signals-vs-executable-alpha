from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microalpha.backtest.economic_v07 import (
    DEFAULT_COVERAGES,
    DEFAULT_EXTRA_RT_COST_BPS,
    EconomicSpec,
    add_reconstructed_bbo,
    confidence_cutoffs,
    monthly_metrics,
    simulate_nonoverlapping,
    trade_metrics,
    weekly_positive_fraction,
)
from microalpha.data.feature_dataset import load_and_prepare_feature_dataset
from microalpha.research.targets import add_exact_forward_target
from microalpha.validation.periods import locked_period_map, select_purged_period

# Import the exact same baseline model/features used in v0.5/v0.6.
from chronological_logistic_baseline_v05 import MODEL_FEATURES, build_model


NOV_START = "2023-11-01T00:00:00Z"
DEC_START = "2023-12-01T00:00:00Z"
TEST_START = "2024-01-01T00:00:00Z"
END = "2024-04-01T00:00:00Z"


def verify_prior_gate(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing prior gate file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not obj.get("gate_pass", False):
        raise SystemExit("Prior statistical gate did not pass; economic escalation is forbidden.")
    return obj


def prepare(path: Path, horizon: int = 10):
    grid, qa = load_and_prepare_feature_dataset(path)
    x = add_exact_forward_target(grid, horizon)
    x = add_reconstructed_bbo(x)

    target = f"fwd_mid_logret_{horizon}m"
    x["y"] = np.where(x[target].notna(), (x[target] > 0).astype(int), np.nan)

    dev_period = locked_period_map()["development"]
    dev = select_purged_period(x, dev_period, horizon)
    dev = dev.dropna(subset=MODEL_FEATURES + [target, "y"])

    model = build_model()
    model.fit(dev[MODEL_FEATURES], dev["y"].astype(int))

    # Predict every row with complete model features. Test probabilities are not
    # used for threshold selection.
    valid = x[MODEL_FEATURES].notna().all(axis=1)
    x["pred_prob"] = np.nan
    x.loc[valid, "pred_prob"] = model.predict_proba(x.loc[valid, MODEL_FEATURES])[:, 1]
    return x, qa


def choose_threshold(selection_rows, cost_bps, spec):
    cutoffs = confidence_cutoffs(selection_rows["pred_prob"], DEFAULT_COVERAGES)
    rows = []
    trades_by_cutoff = {}

    for _, c in cutoffs.iterrows():
        cutoff = float(c["confidence_cutoff"])
        trades = simulate_nonoverlapping(
            selection_rows,
            confidence_cutoff=cutoff,
            extra_roundtrip_cost_bps=cost_bps,
            period_start=NOV_START,
            period_end_exclusive=DEC_START,
            spec=spec,
        )
        m = trade_metrics(trades)
        row = {
            "extra_roundtrip_cost_bps": float(cost_bps),
            "coverage_target": float(c["coverage_target"]),
            "confidence_cutoff": cutoff,
            **m,
        }
        rows.append(row)
        trades_by_cutoff[cutoff] = trades

    table = pd.DataFrame(rows)
    eligible = table[table["n_trades"] >= spec.min_trades_selection].copy()

    if len(eligible) == 0:
        # Mark scenario as unselectable.
        return table, None

    # Frozen selection rule:
    # 1) highest November net total bps;
    # 2) if tied, higher trade count (simpler / more data).
    eligible = eligible.sort_values(
        ["net_total_bps", "n_trades"],
        ascending=[False, False],
    )
    selected = eligible.iloc[0].to_dict()
    return table, selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument(
        "--prior-gate",
        type=Path,
        default=Path("prior_results/validation_gate.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/economic_v07"),
    )
    args = p.parse_args()

    prior_gate = verify_prior_gate(args.prior_gate)
    spec = EconomicSpec()
    x, qa = prepare(args.data, spec.forecast_horizon_minutes)

    selection_rows = x[
        (x["timestamp"] >= pd.Timestamp(NOV_START))
        & (x["timestamp"] < pd.Timestamp(DEC_START))
    ].copy()
    confirmation_rows = x[
        (x["timestamp"] >= pd.Timestamp(DEC_START))
        & (x["timestamp"] < pd.Timestamp(TEST_START))
    ].copy()
    test_rows = x[
        (x["timestamp"] >= pd.Timestamp(TEST_START))
        & (x["timestamp"] < pd.Timestamp(END))
    ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_parts = []
    selected_rows = []
    confirmation_out = []
    test_out = []
    test_monthly_parts = []
    economic_gate = {}

    for cost in DEFAULT_EXTRA_RT_COST_BPS:
        table, selected = choose_threshold(selection_rows, cost, spec)
        selection_parts.append(table)

        if selected is None:
            economic_gate[str(cost)] = {
                "pass": False,
                "reason": "No November threshold met minimum trade count.",
            }
            continue

        selected_rows.append(selected)
        cutoff = float(selected["confidence_cutoff"])

        dec_trades = simulate_nonoverlapping(
            confirmation_rows,
            confidence_cutoff=cutoff,
            extra_roundtrip_cost_bps=cost,
            period_start=DEC_START,
            period_end_exclusive=TEST_START,
            spec=spec,
        )
        dec_metrics = trade_metrics(dec_trades)
        dec_week_frac = weekly_positive_fraction(dec_trades)
        dec_row = {
            "extra_roundtrip_cost_bps": float(cost),
            "selected_coverage_target": float(selected["coverage_target"]),
            "selected_confidence_cutoff": cutoff,
            "november_selected_net_total_bps": float(selected["net_total_bps"]),
            **dec_metrics,
            "positive_week_fraction": dec_week_frac,
        }
        confirmation_out.append(dec_row)

        # Economic confirmation gate frozen before test PnL:
        checks = {
            "december_min_100_trades": dec_metrics["n_trades"] >= spec.min_trades_confirmation,
            "december_net_mean_positive": (
                np.isfinite(dec_metrics["net_mean_bps"]) and dec_metrics["net_mean_bps"] > 0
            ),
            "december_net_total_positive": dec_metrics["net_total_bps"] > 0,
            "december_positive_weeks_ge_half": (
                np.isfinite(dec_week_frac) and dec_week_frac >= 0.50
            ),
        }
        passed = all(checks.values())
        economic_gate[str(cost)] = {
            "pass": bool(passed),
            "checks": checks,
            "selected_confidence_cutoff": cutoff,
            "selected_coverage_target": float(selected["coverage_target"]),
        }

        # The economic test is evaluated only for cost scenarios that passed
        # December confirmation. Threshold/cost are unchanged.
        if passed:
            test_trades = simulate_nonoverlapping(
                test_rows,
                confidence_cutoff=cutoff,
                extra_roundtrip_cost_bps=cost,
                period_start=TEST_START,
                period_end_exclusive=END,
                spec=spec,
            )
            tm = trade_metrics(test_trades)
            test_out.append({
                "extra_roundtrip_cost_bps": float(cost),
                "selected_coverage_target": float(selected["coverage_target"]),
                "selected_confidence_cutoff": cutoff,
                **tm,
            })
            mm = monthly_metrics(test_trades)
            if len(mm):
                mm.insert(0, "extra_roundtrip_cost_bps", float(cost))
                test_monthly_parts.append(mm)

    selection_df = pd.concat(selection_parts, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    confirmation_df = pd.DataFrame(confirmation_out)
    test_df = pd.DataFrame(test_out)
    test_monthly = (
        pd.concat(test_monthly_parts, ignore_index=True)
        if test_monthly_parts else pd.DataFrame()
    )

    selection_df.to_csv(args.output_dir / "november_threshold_grid.csv", index=False)
    selected_df.to_csv(args.output_dir / "november_selected_thresholds.csv", index=False)
    confirmation_df.to_csv(args.output_dir / "december_confirmation.csv", index=False)
    test_df.to_csv(args.output_dir / "economic_test_oos.csv", index=False)
    test_monthly.to_csv(args.output_dir / "economic_test_monthly.csv", index=False)

    gate_obj = {
        "protocol_version": "v0.7",
        "prior_statistical_gate_pass": bool(prior_gate.get("gate_pass", False)),
        "model_horizon_minutes": spec.forecast_horizon_minutes,
        "execution_delay_minutes": spec.execution_delay_minutes,
        "realized_primary_holding_minutes": (
            spec.forecast_horizon_minutes - spec.execution_delay_minutes
        ),
        "spread_handling": "Crossed at reconstructed BBO on entry and exit.",
        "extra_roundtrip_cost_bps_grid": list(DEFAULT_EXTRA_RT_COST_BPS),
        "threshold_coverages": list(DEFAULT_COVERAGES),
        "selection_period": "2023-11",
        "confirmation_period": "2023-12",
        "economic_oos_period": "2024-01 through 2024-03",
        "scenario_gates": economic_gate,
        "important": (
            "Economic test results are produced only for pre-specified cost scenarios "
            "that passed December confirmation. No threshold or cost is chosen from test PnL."
        ),
    }
    (args.output_dir / "economic_gate.json").write_text(
        json.dumps(gate_obj, indent=2), encoding="utf-8"
    )

    print("NOVEMBER SELECTED THRESHOLDS")
    if len(selected_df):
        cols = [
            "extra_roundtrip_cost_bps", "coverage_target",
            "confidence_cutoff", "n_trades", "net_mean_bps", "net_total_bps",
            "break_even_extra_rt_cost_bps",
        ]
        print(selected_df[[c for c in cols if c in selected_df]].to_string(index=False))
    else:
        print("None")

    print("\nDECEMBER CONFIRMATION")
    if len(confirmation_df):
        cols = [
            "extra_roundtrip_cost_bps", "selected_coverage_target",
            "n_trades", "gross_mean_bps", "net_mean_bps", "net_total_bps",
            "positive_week_fraction", "hac_t_net_mean", "hac_p_net_mean",
        ]
        print(confirmation_df[[c for c in cols if c in confirmation_df]].to_string(index=False))
    else:
        print("None")

    print("\nECONOMIC OOS TEST (ONLY DECEMBER-PASSED SCENARIOS)")
    if len(test_df):
        cols = [
            "extra_roundtrip_cost_bps", "selected_coverage_target",
            "n_trades", "gross_mean_bps", "net_mean_bps", "net_total_bps",
            "hit_rate", "max_drawdown_bps_fixed_notional",
            "hac_t_net_mean", "hac_p_net_mean",
        ]
        print(test_df[[c for c in cols if c in test_df]].to_string(index=False))
    else:
        print("No cost scenario passed December confirmation; economic test was not opened.")

    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
