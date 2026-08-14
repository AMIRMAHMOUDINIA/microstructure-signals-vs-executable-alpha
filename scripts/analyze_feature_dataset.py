from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm


FEATURES = [
    "bt_imbalance_close",
    "bt_imbalance_twap",
    "bt_microprice_premium_close",
]


def analyze(df: pd.DataFrame, horizons=(1, 5, 10)) -> pd.DataFrame:
    rows = []
    price = df["bt_microprice_close"].astype(float)

    for h in horizons:
        target = np.log(price.shift(-h) / price)

        for feat in FEATURES:
            x = df[feat].astype(float)
            mask = x.notna() & target.notna()
            xx = x[mask]
            yy = target[mask]

            pr = pearsonr(xx, yy)
            sr = spearmanr(xx, yy)

            X = sm.add_constant(xx)
            # HAC/Newey-West-style covariance because forward returns overlap.
            model = sm.OLS(yy, X).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(1, h)},
            )

            rows.append({
                "horizon_minutes": h,
                "feature": feat,
                "n": int(mask.sum()),
                "pearson_r": float(pr.statistic),
                "pearson_p_naive": float(pr.pvalue),
                "spearman_rho": float(sr.statistic),
                "spearman_p_naive": float(sr.pvalue),
                "ols_beta": float(model.params[feat]),
                "hac_t": float(model.tvalues[feat]),
                "hac_p": float(model.pvalues[feat]),
            })

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/feature_dataset_analysis.csv"))
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10])
    args = p.parse_args()

    if args.data.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, parse_dates=["timestamp"])

    out = analyze(df, horizons=args.horizons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
