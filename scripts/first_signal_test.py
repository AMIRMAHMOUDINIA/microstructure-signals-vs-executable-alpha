from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from microalpha.features.microstructure import add_forward_log_return


def safe_corr(x, y, method="pearson"):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)
    if method == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return float(r), float(p), len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--horizons", type=int, nargs="+", default=[30, 60, 300])
    p.add_argument("--output", type=Path, default=Path("results/first_signal_test.csv"))
    args = p.parse_args()

    if args.data.suffix.lower() == ".csv":
        df = pd.read_csv(args.data, parse_dates=["timestamp"])
    else:
        df = pd.read_parquet(args.data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    features = ["book_imbalance", "microprice_disp", "rel_spread"]
    rows = []

    for h in args.horizons:
        x = add_forward_log_return(df, h)
        target = f"fwd_logret_{h}s"
        for feature in features:
            pr, pp, n1 = safe_corr(x[feature], x[target], "pearson")
            sr, sp, n2 = safe_corr(x[feature], x[target], "spearman")
            rows.append({
                "horizon_s": h,
                "feature": feature,
                "n": min(n1, n2),
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
            })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
