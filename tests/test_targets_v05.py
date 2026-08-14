import numpy as np
import pandas as pd

from microalpha.research.targets import add_exact_forward_target


def test_missing_future_minute_produces_nan_not_skip():
    ts = pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "observed": [True, True, False, True],
        "mid": [100.0, 101.0, np.nan, 104.0],
    })
    out = add_exact_forward_target(df, 2)

    # t=0 targets exact t+2, which was missing. It must NOT jump to minute 3.
    assert np.isnan(out.loc[0, "fwd_mid_logret_2m"])

    # t=1 -> t=3 is exact and observed.
    assert np.isclose(out.loc[1, "fwd_mid_logret_2m"], np.log(104.0 / 101.0))
