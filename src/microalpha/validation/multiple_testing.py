from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(pvalues) -> np.ndarray:
    """
    Benjamini-Hochberg FDR-adjusted q-values.
    NaNs remain NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    back = np.empty_like(adjusted)
    back[order] = adjusted
    q[valid] = back
    return q


def timestamp_nonoverlap_mask(
    timestamps: pd.Series,
    horizon_minutes: int,
    offset: int = 0,
) -> np.ndarray:
    """
    Select exact clock-aligned, non-overlapping observations.

    Example h=5, offset=0 -> minute epochs divisible by 5.
    """
    if not 0 <= offset < horizon_minutes:
        raise ValueError("offset must be in [0, horizon_minutes)")
    ts = pd.to_datetime(timestamps, utc=True)
    minute_epoch = (ts.astype("int64") // 60_000_000_000).to_numpy()
    return (minute_epoch % horizon_minutes) == offset
