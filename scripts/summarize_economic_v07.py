from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def f(x, d=3):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/economic_v07"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/economic_v07/ECONOMIC_SUMMARY.md"),
    )
    args = p.parse_args()

    sel = pd.read_csv(args.input_dir / "november_selected_thresholds.csv")
    dec = pd.read_csv(args.input_dir / "december_confirmation.csv")
    test_path = args.input_dir / "economic_test_oos.csv"
    test = pd.read_csv(test_path) if test_path.exists() and test_path.stat().st_size else pd.DataFrame()
    gate = json.loads((args.input_dir / "economic_gate.json").read_text(encoding="utf-8"))

    lines = [
        "# Economic Validation Summary — v0.7",
        "",
        "## Locked design",
        "",
        "- Model: unchanged v0.5 logistic regression trained on May–Oct 2023.",
        "- Forecast horizon: 10 minutes.",
        "- Signal selection: November 2023 only.",
        "- Economic confirmation: December 2023.",
        "- Economic OOS: Jan–Mar 2024, only for cost scenarios that passed December.",
        "- Execution: signal at minute t; entry at reconstructed BBO at t+1; exit at reconstructed BBO at t+10.",
        "- Spread: crossed on both sides.",
        "- Extra cost: fixed round-trip fee+slippage sensitivity after spread.",
        "",
        "## November threshold selection",
        "",
        "| Extra RT cost (bps) | Coverage | Trades | Gross mean (bps) | Net mean (bps) | Net total (bps) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in sel.iterrows():
        lines.append(
            f"| {f(r['extra_roundtrip_cost_bps'],1)} | {f(r['coverage_target'],2)} | "
            f"{int(r['n_trades'])} | {f(r['gross_mean_bps'])} | {f(r['net_mean_bps'])} | "
            f"{f(r['net_total_bps'],1)} |"
        )

    lines += [
        "",
        "## December confirmation",
        "",
        "| Extra RT cost (bps) | Passed? | Trades | Gross mean | Net mean | Net total | Positive weeks |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]

    dec_lookup = {float(r["extra_roundtrip_cost_bps"]): r for _, r in dec.iterrows()}
    for cost_s, info in gate["scenario_gates"].items():
        cost = float(cost_s)
        r = dec_lookup.get(cost)
        if r is None:
            lines.append(f"| {f(cost,1)} | NO | 0 | NA | NA | NA | NA |")
        else:
            lines.append(
                f"| {f(cost,1)} | {'YES' if info.get('pass') else 'NO'} | "
                f"{int(r['n_trades'])} | {f(r['gross_mean_bps'])} | {f(r['net_mean_bps'])} | "
                f"{f(r['net_total_bps'],1)} | {f(r['positive_week_fraction'],2)} |"
            )

    lines += [
        "",
        "## Economic OOS test",
        "",
    ]

    if len(test):
        lines += [
            "| Extra RT cost (bps) | Trades | Gross mean | Net mean | Net total | Hit rate | Max DD (bps) | HAC p |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in test.iterrows():
            lines.append(
                f"| {f(r['extra_roundtrip_cost_bps'],1)} | {int(r['n_trades'])} | "
                f"{f(r['gross_mean_bps'])} | {f(r['net_mean_bps'])} | {f(r['net_total_bps'],1)} | "
                f"{f(r['hit_rate'],3)} | {f(r['max_drawdown_bps_fixed_notional'],1)} | "
                f"{f(r['hac_p_net_mean'],4)} |"
            )
    else:
        lines.append(
            "**No pre-specified cost scenario passed December confirmation, so economic OOS PnL was not opened.**"
        )

    lines += [
        "",
        "## Interpretation rule",
        "",
        (
            "A statistically predictive model is not called tradable merely because AUC is above 0.50. "
            "The primary economic question is whether conservative BBO crossing plus realistic additional costs "
            "leave a stable positive net return without test-driven threshold selection."
        ),
        "",
        (
            "Funding, queue effects, latency below one minute, market impact, and position-size capacity are not modeled here. "
            "This is a conservative small-notional research backtest, not a production trading simulator."
        ),
    ]

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
