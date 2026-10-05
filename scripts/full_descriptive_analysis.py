from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from microalpha.validation.inference import (
    moving_block_bootstrap_beta,
    non_overlapping_mask,
    ols_naive_and_hac,
)

FEATURES = [
    "bt_imbalance_close",
    "bt_imbalance_twap",
    "bt_microprice_premium_close",
]

PRICE = "bt_microprice_close"


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    missing = set(FEATURES + [PRICE]).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def verify_minute_grid(df: pd.DataFrame) -> dict:
    dt = df["timestamp"].diff().dropna().dt.total_seconds()
    return {
        "rows": len(df),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "duplicate_timestamps_after_cleanup": int(df["timestamp"].duplicated().sum()),
        "median_spacing_seconds": float(dt.median()) if len(dt) else np.nan,
        "p99_spacing_seconds": float(dt.quantile(0.99)) if len(dt) else np.nan,
        "max_spacing_seconds": float(dt.max()) if len(dt) else np.nan,
        "rows_not_60s_after_previous": int((dt != 60).sum()) if len(dt) else 0,
    }


def target_return(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    return np.log(df[PRICE].shift(-horizon_minutes) / df[PRICE])


def analyze_feature(
    df: pd.DataFrame,
    feature: str,
    horizon: int,
    bootstrap_reps: int,
    block_length: int,
) -> dict:
    y = target_return(df, horizon)
    x = df[feature].astype(float)
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)

    xx = x[valid]
    yy = y[valid]

    pr = pearsonr(xx, yy)
    sr = spearmanr(xx, yy)
    reg = ols_naive_and_hac(xx, yy, hac_lags=horizon)

    # A second inference path using non-overlapping observations.
    valid_idx = np.flatnonzero(valid.to_numpy())
    stride_mask = non_overlapping_mask(len(valid_idx), horizon_rows=horizon)
    pick = valid_idx[stride_mask]
    x_non = x.iloc[pick]
    y_non = y.iloc[pick]
    reg_non = ols_naive_and_hac(x_non, y_non, hac_lags=1)

    out = {
        "horizon_minutes": int(horizon),
        "feature": feature,
        "n": int(valid.sum()),
        "pearson_r": float(pr.statistic),
        "pearson_p_naive": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p_naive": float(sr.pvalue),
        **{f"all_{k}": v for k, v in asdict(reg).items()},
        "nonoverlap_n": len(pick),
        "nonoverlap_beta": float(reg_non.beta),
        "nonoverlap_p": float(reg_non.p_naive),
    }

    if bootstrap_reps > 0:
        boot = moving_block_bootstrap_beta(
            xx,
            yy,
            block_length=block_length,
            reps=bootstrap_reps,
            seed=42 + horizon,
        )
        out.update(boot)

    return out


def monthly_stability(df: pd.DataFrame, feature: str, horizon: int) -> pd.DataFrame:
    x = df.copy()
    x["target"] = target_return(x, horizon)
    x["month"] = x["timestamp"].dt.to_period("M").astype(str)

    rows = []
    for month, g in x.groupby("month", sort=True):
        z = g[[feature, "target"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 100:
            continue
        reg = ols_naive_and_hac(z[feature], z["target"], hac_lags=horizon)
        rows.append({
            "month": month,
            "feature": feature,
            "horizon_minutes": horizon,
            **asdict(reg),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/full_descriptive"))
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 30])
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--block-length", type=int, default=60)
    args = p.parse_args()

    df = load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qa = verify_minute_grid(df)
    (args.output_dir / "grid_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    summary_rows = []
    monthly_parts = []

    for h in args.horizons:
        for feature in FEATURES:
            summary_rows.append(
                analyze_feature(
                    df, feature, h,
                    bootstrap_reps=args.bootstrap_reps,
                    block_length=args.block_length,
                )
            )
            monthly_parts.append(monthly_stability(df, feature, h))

    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_parts, ignore_index=True)

    summary.to_csv(args.output_dir / "signal_summary.csv", index=False)
    monthly.to_csv(args.output_dir / "monthly_stability.csv", index=False)

    print("GRID QA")
    print(json.dumps(qa, indent=2))
    print("\nSIGNAL SUMMARY")
    show = [
        "horizon_minutes", "feature", "n", "pearson_r",
        "all_beta", "all_p_naive", "all_p_hac",
        "nonoverlap_beta", "nonoverlap_p",
        "beta_ci_low_95", "beta_ci_high_95",
    ]
    existing = [c for c in show if c in summary.columns]
    print(summary[existing].to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
