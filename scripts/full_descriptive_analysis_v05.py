from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from microalpha.data.feature_dataset import (
    load_and_prepare_feature_dataset,
    qa_to_dict,
)
from microalpha.research.targets import add_exact_forward_target
from microalpha.validation.inference import (
    moving_block_bootstrap_beta_fast,
    ols_naive_and_hac,
)
from microalpha.validation.multiple_testing import (
    benjamini_hochberg,
    timestamp_nonoverlap_mask,
)
from microalpha.validation.periods import locked_period_map, select_purged_period


PRIMARY_FEATURES = [
    "bt_imbalance_close",
    "bt_imbalance_twap",
    # Composite pressure proxy; not independent of close imbalance + close spread.
    "bt_microprice_premium_close",
]


def nonoverlap_offset_stability(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    horizon: int,
) -> dict:
    betas, pvals, ns = [], [], []
    for offset in range(horizon):
        mask = timestamp_nonoverlap_mask(df["timestamp"], horizon, offset)
        z = (
            df.loc[mask, [feature, target_col]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(z) < 20:
            continue
        reg = ols_naive_and_hac(z[feature], z[target_col], hac_lags=1)
        betas.append(reg.beta)
        pvals.append(reg.p_naive)
        ns.append(reg.n)

    if not betas:
        return {
            "offsets_used": 0,
            "nonoverlap_beta_median": np.nan,
            "nonoverlap_beta_min": np.nan,
            "nonoverlap_beta_max": np.nan,
            "nonoverlap_positive_fraction": np.nan,
            "nonoverlap_p_median": np.nan,
            "nonoverlap_n_median": np.nan,
        }

    b = np.asarray(betas)
    return {
        "offsets_used": int(len(b)),
        "nonoverlap_beta_median": float(np.median(b)),
        "nonoverlap_beta_min": float(np.min(b)),
        "nonoverlap_beta_max": float(np.max(b)),
        "nonoverlap_positive_fraction": float(np.mean(b > 0)),
        "nonoverlap_p_median": float(np.median(pvals)),
        "nonoverlap_n_median": float(np.median(ns)),
    }


def analyze_one(
    df: pd.DataFrame,
    sample: str,
    feature: str,
    horizon: int,
    bootstrap_reps: int,
    block_length: int,
) -> dict | None:
    target_col = f"fwd_mid_logret_{horizon}m"
    z = df[["timestamp", feature, target_col]].replace([np.inf, -np.inf], np.nan)
    z = z.dropna(subset=[feature, target_col])
    if len(z) < 20:
        return None

    x = z[feature].astype(float)
    y = z[target_col].astype(float)

    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    reg = ols_naive_and_hac(x, y, hac_lags=max(1, horizon))

    out = {
        "sample": sample,
        "horizon_minutes": horizon,
        "feature": feature,
        "n": int(len(z)),
        "pearson_r": float(pr.statistic),
        "pearson_p_naive": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p_naive": float(sr.pvalue),
        "beta_logret_per_unit": float(reg.beta),
        "beta_bps_per_unit": float(reg.beta * 10_000),
        "ols_p_naive": float(reg.p_naive),
        "hac_t": float(reg.t_hac),
        "hac_p": float(reg.p_hac),
        "r2": float(reg.r2),
    }
    out.update(nonoverlap_offset_stability(df, feature, target_col, horizon))

    if bootstrap_reps > 0 and len(z) >= 2 * block_length:
        out.update(
            moving_block_bootstrap_beta_fast(
                x,
                y,
                block_length=block_length,
                reps=bootstrap_reps,
                seed=10_000 + horizon,
            )
        )
    return out


def monthly_stability(df: pd.DataFrame, feature: str, horizon: int) -> pd.DataFrame:
    target_col = f"fwd_mid_logret_{horizon}m"
    x = df[["timestamp", feature, target_col]].copy()
    x["month"] = pd.to_datetime(x["timestamp"], utc=True).dt.strftime("%Y-%m")

    rows = []
    for month, g in x.groupby("month", sort=True):
        z = g[[feature, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 100:
            continue
        reg = ols_naive_and_hac(z[feature], z[target_col], hac_lags=max(1, horizon))
        rows.append(
            {
                "month": month,
                "feature": feature,
                "horizon_minutes": horizon,
                "n": reg.n,
                "beta_bps_per_unit": reg.beta * 10_000,
                "hac_t": reg.t_hac,
                "hac_p": reg.p_hac,
                "r2": reg.r2,
            }
        )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/full_descriptive_v05"))
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 30])
    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--block-length", type=int, default=240)
    args = p.parse_args()

    grid, qa = load_and_prepare_feature_dataset(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_qa.json").write_text(
        json.dumps(qa_to_dict(qa), indent=2), encoding="utf-8"
    )

    periods = locked_period_map()
    summary_rows = []
    monthly_parts = []

    for h in args.horizons:
        with_target = add_exact_forward_target(grid, h)

        # Descriptive discovery/validation are explicitly pre-test only.
        dev = select_purged_period(with_target, periods["development"], h)
        val = select_purged_period(with_target, periods["validation"], h)

        for sample_name, sample_df in (("development", dev), ("validation", val)):
            for feature in PRIMARY_FEATURES:
                row = analyze_one(
                    sample_df,
                    sample_name,
                    feature,
                    h,
                    bootstrap_reps=args.bootstrap_reps,
                    block_length=args.block_length,
                )
                if row is not None:
                    summary_rows.append(row)

        # Monthly stability may use development + validation, but never test.
        pretest = pd.concat([dev, val], ignore_index=True)
        for feature in PRIMARY_FEATURES:
            part = monthly_stability(pretest, feature, h)
            if not part.empty:
                monthly_parts.append(part)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise SystemExit("No analyzable development/validation observations found.")

    # FDR is controlled separately within each pre-test sample.
    summary["hac_q_bh"] = np.nan
    for sample_name, idx in summary.groupby("sample").groups.items():
        summary.loc[idx, "hac_q_bh"] = benjamini_hochberg(summary.loc[idx, "hac_p"])

    monthly = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()

    summary.to_csv(args.output_dir / "signal_summary.csv", index=False)
    monthly.to_csv(args.output_dir / "monthly_stability.csv", index=False)

    print("DATASET QA")
    print(json.dumps(qa_to_dict(qa), indent=2))
    print("\nPRE-TEST SIGNAL SUMMARY (TEST EXCLUDED)")
    show = [
        "sample",
        "horizon_minutes",
        "feature",
        "n",
        "pearson_r",
        "beta_bps_per_unit",
        "hac_p",
        "hac_q_bh",
        "nonoverlap_beta_median",
        "nonoverlap_positive_fraction",
        "beta_ci_low_95",
        "beta_ci_high_95",
    ]
    show = [c for c in show if c in summary.columns]
    print(summary[show].to_string(index=False))
    print("\nJanuary-March 2024 test rows were not analyzed.")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
