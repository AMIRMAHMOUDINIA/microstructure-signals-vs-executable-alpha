from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "bt_imbalance_close",
    "bt_imbalance_twap",
    "bt_microprice_premium_close",
    "bt_spread_bps_close",
    "bt_spread_bps_twap",
    "bt_update_rate",
]
PRICE = "bt_microprice_close"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = df.copy()
    x["fwd_ret"] = np.log(x[PRICE].shift(-horizon) / x[PRICE])
    x["y"] = (x["fwd_ret"] > 0).astype(int)
    return x


def split_periods(df: pd.DataFrame):
    ts = df["timestamp"]
    dev = df[(ts >= "2023-05-16") & (ts < "2023-11-01")]
    val = df[(ts >= "2023-11-01") & (ts < "2024-01-01")]
    test = df[(ts >= "2024-01-01") & (ts < "2024-04-01")]
    return {"development": dev, "validation": val, "test": test}


def build_model(C: float = 1.0):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000)),
    ])


def evaluate(name, model, df):
    x = df.dropna(subset=["fwd_ret"]).copy()
    y = x["y"].astype(int)
    p = model.predict_proba(x[FEATURES])[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "period": name,
        "n": len(x),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("results/logistic_baseline.csv"))
    args = p.parse_args()

    df = add_target(load(args.data), args.horizon)
    periods = split_periods(df)

    if min(len(v) for v in periods.values()) == 0:
        raise SystemExit("One or more locked calendar periods is empty.")

    model = build_model()
    dev = periods["development"].dropna(subset=["fwd_ret"])
    model.fit(dev[FEATURES], dev["y"].astype(int))

    rows = [evaluate(name, model, data) for name, data in periods.items()]
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print("\nIMPORTANT: C and the 0.5 threshold were not tuned here.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
