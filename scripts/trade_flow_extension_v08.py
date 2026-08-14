from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microalpha.data.feature_dataset import load_and_prepare_feature_dataset
from microalpha.data.trade_flow_v08 import (
    add_trade_flow_features,
    flow_qa,
    load_monthly_klines,
    merge_flow_with_book,
    normalize_klines,
    qa_to_dict,
)
from microalpha.research.targets import add_exact_forward_target
from microalpha.validation.incremental_v08 import metrics, paired_day_bootstrap_auc_delta
from microalpha.validation.inference import ols_naive_and_hac
from microalpha.validation.periods import Period, select_purged_period

from chronological_logistic_baseline_v05 import MODEL_FEATURES, build_model


HORIZON = 10

FLOW_FEATURES = [
    "tfi_1m",
    "tfi_5m",
    "tfi_10m",
    "log_quote_volume",
    "log_num_trades",
    "tfi5_x_book_imbalance",
]
AUGMENTED_FEATURES = MODEL_FEATURES + FLOW_FEATURES

PERIODS = {
    "development": Period(
        "development", "2023-05-16T11:49:00Z", "2023-11-01T00:00:00Z"
    ),
    "november": Period(
        "november", "2023-11-01T00:00:00Z", "2023-12-01T00:00:00Z"
    ),
    "december": Period(
        "december", "2023-12-01T00:00:00Z", "2024-01-01T00:00:00Z"
    ),
    "extension_oos": Period(
        "extension_oos", "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z"
    ),
}


def prepare(book_path: Path, kline_dir: Path):
    book, book_qa = load_and_prepare_feature_dataset(book_path)
    raw = load_monthly_klines(kline_dir)
    k = normalize_klines(raw)
    duplicate_removed = int(k.attrs.get("duplicate_minutes_removed", 0))
    f = add_trade_flow_features(k)
    merged = merge_flow_with_book(book, f)
    qa = flow_qa(merged, duplicate_removed)
    x = add_exact_forward_target(merged, HORIZON)
    target = f"fwd_mid_logret_{HORIZON}m"
    x["y"] = np.where(x[target].notna(), (x[target] > 0).astype(int), np.nan)
    return x, book_qa, qa


def period_data(x, name, features):
    p = PERIODS[name]
    z = select_purged_period(x, p, HORIZON)
    target = f"fwd_mid_logret_{HORIZON}m"
    return z.dropna(subset=features + [target, "y"]).copy()


def fit_models(x):
    dev_base = period_data(x, "development", MODEL_FEATURES)
    dev_aug = period_data(x, "development", AUGMENTED_FEATURES)

    # Use the common development rows for an apples-to-apples comparison.
    common_idx = dev_base.index.intersection(dev_aug.index)
    dev_base = dev_base.loc[common_idx]
    dev_aug = dev_aug.loc[common_idx]

    base = build_model()
    aug = build_model()
    base.fit(dev_base[MODEL_FEATURES], dev_base["y"].astype(int))
    aug.fit(dev_aug[AUGMENTED_FEATURES], dev_aug["y"].astype(int))
    return base, aug, dev_aug


def evaluate_period(x, name, base, aug):
    z = period_data(x, name, AUGMENTED_FEATURES)
    y = z["y"].astype(int).to_numpy()
    pb = base.predict_proba(z[MODEL_FEATURES])[:, 1]
    pa = aug.predict_proba(z[AUGMENTED_FEATURES])[:, 1]

    mb = metrics(y, pb)
    ma = metrics(y, pa)

    row = {
        "period": name,
        "n": len(z),
        "baseline_auc": mb["roc_auc"],
        "augmented_auc": ma["roc_auc"],
        "delta_auc": ma["roc_auc"] - mb["roc_auc"],
        "baseline_log_loss": mb["log_loss"],
        "augmented_log_loss": ma["log_loss"],
        "delta_log_loss": ma["log_loss"] - mb["log_loss"],
        "baseline_brier": mb["brier"],
        "augmented_brier": ma["brier"],
        "delta_brier": ma["brier"] - mb["brier"],
        "baseline_mcc": mb["mcc"],
        "augmented_mcc": ma["mcc"],
        "delta_mcc": ma["mcc"] - mb["mcc"],
    }
    boot = paired_day_bootstrap_auc_delta(
        z["timestamp"], y, pb, pa, reps=1000, seed=100 + len(name)
    )
    row.update(boot)
    return row


def tfi_monthly_stability(x):
    target = f"fwd_mid_logret_{HORIZON}m"
    pre = x[
        (x["timestamp"] >= pd.Timestamp("2023-05-16T11:49:00Z"))
        & (x["timestamp"] < pd.Timestamp("2024-01-01T00:00:00Z"))
    ].copy()
    pre["month"] = pd.to_datetime(pre["timestamp"], utc=True).dt.strftime("%Y-%m")

    rows = []
    for month, g in pre.groupby("month", sort=True):
        z = g[["tfi_5m", target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 500:
            continue
        r = ols_naive_and_hac(z["tfi_5m"], z[target], hac_lags=HORIZON)
        rows.append({
            "month": month,
            "n": r.n,
            "beta_bps_per_unit": r.beta * 10_000,
            "hac_t": r.t_hac,
            "hac_p": r.p_hac,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("book_data", type=Path)
    p.add_argument(
        "--kline-dir",
        type=Path,
        default=Path("data/raw/binance_klines"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trade_flow_v08"),
    )
    args = p.parse_args()

    x, book_qa, flowqa = prepare(args.book_data, args.kline_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "flow_qa.json").write_text(
        json.dumps(qa_to_dict(flowqa), indent=2), encoding="utf-8"
    )

    base, aug, dev = fit_models(x)

    # Primary augmented coefficients.
    coef = pd.DataFrame({
        "feature": AUGMENTED_FEATURES,
        "standardized_logit_coef": aug.named_steps["model"].coef_[0],
    })
    coef["abs_coef"] = coef["standardized_logit_coef"].abs()
    coef = coef.sort_values("abs_coef", ascending=False)
    coef.to_csv(args.output_dir / "augmented_coefficients.csv", index=False)

    monthly = tfi_monthly_stability(x)
    monthly.to_csv(args.output_dir / "tfi5_monthly_stability.csv", index=False)
    positive_fraction = float((monthly["beta_bps_per_unit"] > 0).mean()) if len(monthly) else 0.0

    # Evaluate only November and December before deciding whether extension OOS may be opened.
    nov = evaluate_period(x, "november", base, aug)
    dec = evaluate_period(x, "december", base, aug)
    pre_oos = pd.DataFrame([nov, dec])
    pre_oos.to_csv(args.output_dir / "incremental_pre_oos.csv", index=False)

    tfi5_coef = float(
        coef.loc[coef["feature"] == "tfi_5m", "standardized_logit_coef"].iloc[0]
    )

    checks = {
        "november_delta_auc_ge_0_002": nov["delta_auc"] >= 0.002,
        "december_delta_auc_ge_0_001": dec["delta_auc"] >= 0.001,
        "december_log_loss_not_worse": dec["delta_log_loss"] <= 0.0,
        "development_tfi5_coefficient_positive": tfi5_coef > 0.0,
        "pre_oos_monthly_tfi5_positive_fraction_ge_0_70": positive_fraction >= 0.70,
    }
    gate_pass = all(checks.values())

    gate = {
        "gate_pass": bool(gate_pass),
        "checks": {k: bool(v) for k, v in checks.items()},
        "primary_horizon_minutes": HORIZON,
        "flow_features": FLOW_FEATURES,
        "november_delta_auc": nov["delta_auc"],
        "december_delta_auc": dec["delta_auc"],
        "december_delta_log_loss": dec["delta_log_loss"],
        "development_tfi5_standardized_coef": tfi5_coef,
        "monthly_tfi5_positive_fraction": positive_fraction,
        "policy": (
            "v0.8 feature set and gate fixed before dynamic-flow results. "
            "If gate fails, Jan-Mar extension OOS remains closed."
        ),
    }
    (args.output_dir / "extension_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )

    if gate_pass:
        oos = evaluate_period(x, "extension_oos", base, aug)
        pd.DataFrame([oos]).to_csv(
            args.output_dir / "incremental_extension_oos.csv", index=False
        )
    else:
        (args.output_dir / "incremental_extension_oos.csv").write_text("", encoding="utf-8")

    print("FLOW QA")
    print(json.dumps(qa_to_dict(flowqa), indent=2))
    print("\nPRE-OOS INCREMENTAL RESULTS")
    print(pre_oos.to_string(index=False))
    print("\nTFI_5M MONTHLY STABILITY")
    print(monthly.to_string(index=False))
    print("\nEXTENSION GATE")
    print(json.dumps(gate, indent=2))

    if gate_pass:
        print("\nGate PASSED: Jan-Mar extension OOS evaluated once.")
        print(pd.read_csv(args.output_dir / "incremental_extension_oos.csv").to_string(index=False))
    else:
        print("\nGate FAILED: Jan-Mar extension OOS was not opened.")

    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
