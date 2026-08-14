import numpy as np
import pandas as pd

from microalpha.validation.multiple_testing import (
    benjamini_hochberg,
    timestamp_nonoverlap_mask,
)


def test_bh_adjustment():
    p = np.array([0.001, 0.01, 0.20, np.nan])
    q = benjamini_hochberg(p)
    assert np.isnan(q[3])
    assert q[0] <= q[1] <= q[2]
    assert np.all(q[:3] >= p[:3])


def test_timestamp_nonoverlap_is_clock_aligned():
    ts = pd.date_range("2024-01-01", periods=15, freq="1min", tz="UTC")
    mask = timestamp_nonoverlap_mask(pd.Series(ts), 5, 0)
    chosen = ts[mask]
    assert len(chosen) == 3
    assert all(t.minute % 5 == 0 for t in chosen)
