import numpy as np
import pandas as pd

from microalpha.data.feature_dataset import (
    expected_premium_from_identity,
    reconstruct_mid,
)


def test_mid_reconstruction():
    mid = 100.0
    premium = 0.001
    micro = mid * (1 + premium)
    df = pd.DataFrame({
        "bt_microprice_close": [micro],
        "bt_microprice_premium_close": [premium],
    })
    got = reconstruct_mid(df).iloc[0]
    assert np.isclose(got, mid)


def test_l1_premium_identity():
    # spread = 2, mid = 100 => spread_bps = 200 bps.
    # imbalance = 0.5 => premium = 200*0.5/20000 = 0.005.
    df = pd.DataFrame({
        "bt_spread_bps_close": [200.0],
        "bt_imbalance_close": [0.5],
    })
    assert np.isclose(expected_premium_from_identity(df).iloc[0], 0.005)
