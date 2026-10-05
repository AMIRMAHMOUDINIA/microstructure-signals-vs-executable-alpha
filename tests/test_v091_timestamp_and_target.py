import numpy as np
import pandas as pd

from microalpha.data.trade_flow_v08 import add_trade_flow_features, normalize_klines


def base_df(times):
    n=len(times)
    return pd.DataFrame({
        "open_time":times,"open":[100]*n,"high":[101]*n,"low":[99]*n,
        "close":np.arange(n)+100.0,"volume":[100.0]*n,
        "close_time":[t+59999 for t in times],"quote_volume":[10000.0]*n,
        "num_trades":[100]*n,"taker_buy_base":[60.0]*n,
        "taker_buy_quote":[6000.0]*n,"ignore":[0]*n,
    })

def test_ms_timestamp_inference():
    x=normalize_klines(base_df([1735689600000,1735689660000]))
    assert str(x.loc[0,"timestamp"])=="2025-01-01 00:00:00+00:00"

def test_microsecond_timestamp_inference_is_defensive():
    x=normalize_klines(base_df([1735689600000000,1735689660000000]))
    assert str(x.loc[0,"timestamp"])=="2025-01-01 00:00:00+00:00"

def test_tfi_still_correct_after_timestamp_change():
    times=[1735689600000+i*60000 for i in range(10)]
    x=add_trade_flow_features(normalize_klines(base_df(times)))
    assert np.isclose(x.loc[4,"tfi_5m"],0.2)
