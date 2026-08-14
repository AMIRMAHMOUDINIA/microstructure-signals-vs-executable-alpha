from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def f(x, d=4):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/trade_flow_v08"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/trade_flow_v08/TRADE_FLOW_SUMMARY.md"),
    )
    args = p.parse_args()

    pre = pd.read_csv(args.input_dir / "incremental_pre_oos.csv")
    monthly = pd.read_csv(args.input_dir / "tfi5_monthly_stability.csv")
    coef = pd.read_csv(args.input_dir / "augmented_coefficients.csv")
    qa = json.loads((args.input_dir / "flow_qa.json").read_text())
    gate = json.loads((args.input_dir / "extension_gate.json").read_text())
    oos_path = args.input_dir / "incremental_extension_oos.csv"
    # Empty OOS file is a valid outcome when the extension gate fails.

    lines = [
        "# Dynamic Trade-Flow Extension — v0.8",
        "",
        "## Data QA",
        "",
        f"- Flow rows: **{qa['rows']:,}**",
        f"- Flow completeness on book grid: **{f(100*qa['completeness'],3)}%**",
        f"- Median same-minute kline-close vs reconstructed-mid difference: **{f(qa['median_same_minute_close_mid_diff_bps'],3)} bps**",
        "",
        "## Pre-OOS incremental comparison",
        "",
        "| Period | Baseline AUC | Augmented AUC | ΔAUC | ΔLogLoss | AUC bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in pre.iterrows():
        lines.append(
            f"| {r['period']} | {f(r['baseline_auc'])} | {f(r['augmented_auc'])} | "
            f"{f(r['delta_auc'])} | {f(r['delta_log_loss'],6)} | "
            f"[{f(r['delta_auc_ci_low_95'])}, {f(r['delta_auc_ci_high_95'])}] |"
        )

    lines += [
        "",
        "## Primary TFI_5m monthly stability",
        "",
        "| Month | Beta (bps/unit) | HAC p |",
        "|---|---:|---:|",
    ]
    for _, r in monthly.iterrows():
        lines.append(
            f"| {r['month']} | {f(r['beta_bps_per_unit'])} | {f(r['hac_p'],5)} |"
        )

    lines += [
        "",
        "## Largest augmented standardized coefficients",
        "",
        "| Feature | Coefficient |",
        "|---|---:|",
    ]
    for _, r in coef.head(10).iterrows():
        lines.append(
            f"| `{r['feature']}` | {f(r['standardized_logit_coef'],5)} |"
        )

    lines += [
        "",
        "## Pre-specified extension gate",
        "",
        f"**{'PASS' if gate['gate_pass'] else 'FAIL'}**",
        "",
    ]
    for k, v in gate["checks"].items():
        lines.append(f"- {'PASS' if v else 'FAIL'} — `{k}`")

    if gate["gate_pass"] and oos_path.exists() and oos_path.stat().st_size > 1:
        oos = pd.read_csv(oos_path).iloc[0]
        lines += [
            "",
            "## Jan–Mar 2024 extension OOS",
            "",
            f"- Baseline AUC: **{f(oos['baseline_auc'])}**",
            f"- Augmented AUC: **{f(oos['augmented_auc'])}**",
            f"- ΔAUC: **{f(oos['delta_auc'])}**",
            f"- ΔLogLoss: **{f(oos['delta_log_loss'],6)}**",
            f"- Paired day-bootstrap ΔAUC 95% CI: **[{f(oos['delta_auc_ci_low_95'])}, {f(oos['delta_auc_ci_high_95'])}]**",
        ]
    else:
        lines += [
            "",
            "**Jan–Mar extension OOS was not opened because the pre-OOS gate failed.**",
        ]

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
