from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PRIMARY_FEATURE = "bt_imbalance_close"
PRIMARY_HORIZON = 10


def fmt(x, digits=4):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("results/RESULTS_SUMMARY.md"))
    args = p.parse_args()

    desc_path = args.results_root / "full_descriptive_v05" / "signal_summary.csv"
    monthly_path = args.results_root / "full_descriptive_v05" / "monthly_stability.csv"
    qa_path = args.results_root / "full_descriptive_v05" / "dataset_qa.json"
    model_path = args.results_root / "logistic_v05" / "metrics.csv"
    coef_path = args.results_root / "logistic_v05" / "standardized_coefficients.csv"
    gate_path = args.results_root / "validation_gate.json"

    required = [desc_path, monthly_path, qa_path, model_path, coef_path, gate_path]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise SystemExit("Missing required result files:\n" + "\n".join(missing))

    desc = pd.read_csv(desc_path)
    monthly = pd.read_csv(monthly_path)
    metrics = pd.read_csv(model_path)
    coefs = pd.read_csv(coef_path)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    d = desc[
        (desc["sample"] == "development")
        & (desc["feature"] == PRIMARY_FEATURE)
        & (desc["horizon_minutes"] == PRIMARY_HORIZON)
    ].iloc[0]
    vdesc = desc[
        (desc["sample"] == "validation")
        & (desc["feature"] == PRIMARY_FEATURE)
        & (desc["horizon_minutes"] == PRIMARY_HORIZON)
    ].iloc[0]

    devm = metrics[metrics["period"] == "development"].iloc[0]
    valm = metrics[metrics["period"] == "validation"].iloc[0]

    m = monthly[
        (monthly["feature"] == PRIMARY_FEATURE)
        & (monthly["horizon_minutes"] == PRIMARY_HORIZON)
    ].copy()
    positive_fraction = float((m["beta_bps_per_unit"] > 0).mean()) if len(m) else float("nan")

    top_coefs = coefs.head(7)

    lines = []
    lines.append("# Full-Data Research Result Summary")
    lines.append("")
    lines.append("## Dataset integrity")
    lines.append("")
    lines.append(f"- Observed rows: **{qa.get('observed_rows', 'NA'):,}**" if isinstance(qa.get("observed_rows"), int) else f"- Observed rows: **{qa.get('observed_rows', 'NA')}**")
    lines.append(f"- Grid completeness: **{fmt(100 * qa.get('completeness', float('nan')), 3)}%**")
    lines.append(f"- Missing grid rows: **{qa.get('missing_grid_rows', 'NA')}**")
    lines.append(f"- Premium identity correlation: **{fmt(qa.get('premium_identity_corr', float('nan')), 6)}**")
    lines.append("")
    lines.append("## Primary descriptive path — 10-minute close imbalance")
    lines.append("")
    lines.append("| Sample | N | Beta (bps/unit) | HAC p | BH q | Non-overlap positive fraction |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, row in [("Development", d), ("Validation", vdesc)]:
        lines.append(
            f"| {label} | {int(row['n']):,} | {fmt(row['beta_bps_per_unit'], 4)} | "
            f"{fmt(row['hac_p'], 5)} | {fmt(row['hac_q_bh'], 5)} | "
            f"{fmt(row['nonoverlap_positive_fraction'], 3)} |"
        )
    lines.append("")
    lines.append(f"- Positive monthly beta fraction across pre-test months: **{fmt(positive_fraction, 3)}**")
    lines.append("")
    lines.append("## Logistic baseline")
    lines.append("")
    lines.append("| Sample | N | AUC | Balanced accuracy | MCC | Log loss | Brier |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, row in [("Development", devm), ("Validation", valm)]:
        lines.append(
            f"| {label} | {int(row['n']):,} | {fmt(row['roc_auc'])} | "
            f"{fmt(row['balanced_accuracy'])} | {fmt(row['mcc'])} | "
            f"{fmt(row['log_loss'])} | {fmt(row['brier'])} |"
        )
    lines.append("")
    lines.append("## Standardized logistic coefficients")
    lines.append("")
    lines.append("| Feature | Coefficient |")
    lines.append("|---|---:|")
    for _, row in top_coefs.iterrows():
        lines.append(f"| `{row['feature']}` | {fmt(row['standardized_logit_coef'], 5)} |")
    lines.append("")
    lines.append("## Pre-specified validation gate")
    lines.append("")
    lines.append(f"**Gate result: {'PASS' if gate.get('gate_pass') else 'FAIL'}**")
    lines.append("")
    for key, value in gate.get("checks", {}).items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{key}`")
    lines.append("")
    if gate.get("gate_pass"):
        lines.append(
            "The pre-specified gate passed. The untouched January–March 2024 test may now be evaluated **once**."
        )
    else:
        lines.append(
            "The pre-specified gate failed. The correct action is to stop complexity escalation and report the signal as fragile/null under this specification. The untouched test remains closed."
        )

    # Include test metrics only if a deliberately-unlocked run exists.
    test_candidates = [
        args.results_root / "logistic_v05_test" / "metrics.csv",
        args.results_root / "logistic_test" / "metrics.csv",
    ]
    for path in test_candidates:
        if path.exists():
            tdf = pd.read_csv(path)
            test = tdf[tdf["period"] == "test"]
            if len(test):
                row = test.iloc[0]
                lines += [
                    "",
                    "## Untouched test (opened after gate)",
                    "",
                    "| N | AUC | Balanced accuracy | MCC | Log loss | Brier |",
                    "|---:|---:|---:|---:|---:|---:|",
                    (
                        f"| {int(row['n']):,} | {fmt(row['roc_auc'])} | {fmt(row['balanced_accuracy'])} | "
                        f"{fmt(row['mcc'])} | {fmt(row['log_loss'])} | {fmt(row['brier'])} |"
                    ),
                ]
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
