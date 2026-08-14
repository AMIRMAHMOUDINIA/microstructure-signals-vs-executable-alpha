from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def paired_day_bootstrap_auc_delta(
    timestamps,
    y,
    p_base,
    p_aug,
    reps=1000,
    seed=42,
):
    """
    Paired day-block bootstrap for AUC(augmented) - AUC(baseline).
    """
    x = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "y": np.asarray(y, dtype=int),
        "p_base": np.asarray(p_base, dtype=float),
        "p_aug": np.asarray(p_aug, dtype=float),
    })
    x["day"] = x["timestamp"].dt.strftime("%Y-%m-%d")
    days = x["day"].unique()
    if len(days) < 2:
        return {
            "bootstrap_reps": 0,
            "delta_auc_median": np.nan,
            "delta_auc_ci_low_95": np.nan,
            "delta_auc_ci_high_95": np.nan,
            "positive_fraction": np.nan,
        }

    groups = {d: g.index.to_numpy() for d, g in x.groupby("day")}
    rng = np.random.default_rng(seed)
    deltas = []

    for _ in range(reps):
        chosen = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([groups[d] for d in chosen])
        yy = x.loc[idx, "y"].to_numpy()
        if len(np.unique(yy)) < 2:
            continue
        pb = x.loc[idx, "p_base"].to_numpy()
        pa = x.loc[idx, "p_aug"].to_numpy()
        deltas.append(roc_auc_score(yy, pa) - roc_auc_score(yy, pb))

    a = np.asarray(deltas, dtype=float)
    return {
        "bootstrap_reps": int(len(a)),
        "delta_auc_median": float(np.median(a)) if len(a) else np.nan,
        "delta_auc_ci_low_95": float(np.quantile(a, 0.025)) if len(a) else np.nan,
        "delta_auc_ci_high_95": float(np.quantile(a, 0.975)) if len(a) else np.nan,
        "positive_fraction": float(np.mean(a > 0)) if len(a) else np.nan,
    }
