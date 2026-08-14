import numpy as np
import pandas as pd
from microalpha.features.microstructure import add_book_features, add_forward_log_return


def test_book_imbalance_and_microprice():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"]),
        "bid_price": [99.0],
        "bid_qty": [30.0],
        "ask_price": [101.0],
        "ask_qty": [10.0],
    })
    out = add_book_features(df)
    assert np.isclose(out.loc[0, "mid"], 100.0)
    assert np.isclose(out.loc[0, "book_imbalance"], 0.5)
    assert np.isclose(out.loc[0, "microprice"], 100.5)


def test_forward_return_uses_future_timestamp():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="1s", tz="UTC"),
        "bid_price": [99, 100, 101, 102],
        "ask_price": [101, 102, 103, 104],
        "bid_qty": [10, 10, 10, 10],
        "ask_qty": [10, 10, 10, 10],
    })
    out = add_book_features(df)
    out = add_forward_log_return(out, horizon_seconds=2)
    expected = np.log(102.0 / 100.0)
    assert np.isclose(out.loc[0, "fwd_logret_2s"], expected)
