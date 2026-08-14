from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--descriptive", type=Path, required=True)
    p.add_argument("--model-metrics", type=Path, required=True)
    p.add_argument("--monthly", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("results/validation_gate.json"))
    args = p.parse_args()

    desc = pd.read_csv(args.descriptive)
    metrics = pd.read_csv(args.model_metrics)
    monthly = pd.read_csv(args.monthly)

    d = desc[
        (desc["sample"] == "development")
        & (desc["feature"] == "bt_imbalance_close")
        & (desc["horizon_minutes"] == args.horizon)
    ]
    if len(d) != 1:
        raise SystemExit("Expected exactly one primary descriptive row.")
    d = d.iloc[0]

    val = metrics[metrics["period"] == "validation"]
    if len(val) != 1:
        raise SystemExit("Validation metrics missing.")
    val = val.iloc[0]

    m = monthly[
        (monthly["feature"] == "bt_imbalance_close")
        & (monthly["horizon_minutes"] == args.horizon)
        & (monthly["month"] <= "2023-12")
    ]
    positive_fraction = float((m["beta_bps_per_unit"] > 0).mean()) if len(m) else 0.0

    checks = {
        "development_hac_fdr_q_le_005": bool(d["hac_q_bh"] <= 0.05),
        "nonoverlap_sign_agreement_ge_075": bool(
            d["nonoverlap_positive_fraction"] >= 0.75
        ),
        "monthly_positive_fraction_ge_070": bool(positive_fraction >= 0.70),
        "validation_auc_ge_0505": bool(val["roc_auc"] >= 0.505),
        "validation_mcc_positive": bool(val["mcc"] > 0.0),
    }
    gate_pass = all(checks.values())

    result = {
        "gate_pass": gate_pass,
        "horizon_minutes": args.horizon,
        "checks": checks,
        "monthly_positive_fraction": positive_fraction,
        "validation_auc": float(val["roc_auc"]),
        "validation_mcc": float(val["mcc"]),
        "policy": (
            "Internal complexity/test-unlock gate fixed before full-data results. "
            "Failure means stop escalation; it does not prove the null."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
