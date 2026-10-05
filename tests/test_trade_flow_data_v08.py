import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from microalpha.data.trade_flow_v08 import (
    KLINE_COLUMNS,
    add_trade_flow_features,
    merge_flow_with_book,
    normalize_klines,
    read_kline_zip,
)


def test_headerless_kline_archive(tmp_path: Path):
    row = [
        1685577600000, 27000, 27010, 26990, 27005, 100,
        1685577659999, 2700500, 200, 60, 1620300, 0
    ]
    archive = tmp_path / "x.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("x.csv", ",".join(map(str, row)) + "\n")
    df = read_kline_zip(archive)
    assert list(df.columns) == KLINE_COLUMNS
    assert len(df) == 1


def test_tfi_formula():
    raw = pd.DataFrame({
        "open_time": [1685577600000] * 10,
        "open": [1]*10, "high": [1]*10, "low": [1]*10, "close": [1]*10,
        "volume": [100.0]*10,
        "close_time": [1685577659999]*10,
        "quote_volume": [1000.0]*10,
        "num_trades": [10]*10,
        "taker_buy_base": [60.0]*10,
        "taker_buy_quote": [600.0]*10,
        "ignore": [0]*10,
    })
    # Make timestamps unique.
    raw["open_time"] = np.arange(10) * 60_000 + 1685577600000
    x = add_trade_flow_features(normalize_klines(raw))
    assert np.allclose(x["tfi_1m"], 0.2)
    assert np.isclose(x.loc[4, "tfi_5m"], 0.2)
    assert np.isclose(x.loc[9, "tfi_10m"], 0.2)


def test_exact_merge_no_forward_fill():
    book = pd.DataFrame({
        "timestamp": pd.date_range("2023-06-01", periods=3, freq="1min", tz="UTC"),
        "bt_imbalance_close": [0.1, 0.2, 0.3],
    })
    flow = pd.DataFrame({
        "timestamp": [pd.Timestamp("2023-06-01T00:00:00Z"), pd.Timestamp("2023-06-01T00:02:00Z")],
        "close": [100, 101], "volume": [10, 10], "quote_volume": [1000, 1000],
        "num_trades": [5, 5], "taker_buy_base": [6, 4], "taker_buy_quote": [600, 400],
        "signed_taker_base": [2, -2], "signed_taker_quote": [200, -200],
        "tfi_1m": [0.2, -0.2], "tfi_5m": [0.1, -0.1], "tfi_10m": [0.05, -0.05],
        "log_quote_volume": [1,1], "log_num_trades": [1,1], "avg_trade_base": [2,2],
    })
    m = merge_flow_with_book(book, flow)
    assert np.isnan(m.loc[1, "tfi_1m"])
