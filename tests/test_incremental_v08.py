import numpy as np
import pandas as pd

from microalpha.validation.incremental_v08 import metrics, paired_day_bootstrap_auc_delta


def test_metrics_auc():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    m = metrics(y, p)
    assert m["roc_auc"] == 1.0


def test_paired_day_bootstrap_detects_better_model():
    rng = np.random.default_rng(1)
    days = pd.date_range("2023-11-01", periods=20, freq="D", tz="UTC")
    ts = np.repeat(days, 100)
    y = rng.integers(0, 2, size=len(ts))
    base = np.clip(0.5 + rng.normal(0, 0.02, len(ts)), 0.01, 0.99)
    aug = np.clip(0.5 + (y - 0.5) * 0.08 + rng.normal(0, 0.02, len(ts)), 0.01, 0.99)
    b = paired_day_bootstrap_auc_delta(ts, y, base, aug, reps=100, seed=2)
    assert b["delta_auc_median"] > 0
    assert b["positive_fraction"] > 0.9
