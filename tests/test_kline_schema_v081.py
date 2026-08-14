from pathlib import Path
import zipfile
import pandas as pd

from microalpha.data.trade_flow_v08 import KLINE_COLUMNS, read_kline_zip


def test_alternative_header_is_canonicalized_by_position(tmp_path: Path):
    # Simulates a valid Binance 12-field file with labels different from our
    # internal canonical names (e.g. count instead of num_trades).
    header = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ]
    row = [
        1685577600000, 27000, 27010, 26990, 27005, 100,
        1685577659999, 2700500, 200, 60, 1620300, 0
    ]
    archive = tmp_path / "alt_header.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "x.csv",
            ",".join(header) + "\n" + ",".join(map(str, row)) + "\n"
        )

    df = read_kline_zip(archive)
    assert list(df.columns) == KLINE_COLUMNS
    assert df.loc[0, "num_trades"] == 200
    assert df.loc[0, "taker_buy_base"] == 60


def test_title_case_header_is_canonicalized(tmp_path: Path):
    header = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ]
    row = [
        1685577600000, 27000, 27010, 26990, 27005, 100,
        1685577659999, 2700500, 200, 60, 1620300, 0
    ]
    archive = tmp_path / "title_header.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "x.csv",
            ",".join(header) + "\n" + ",".join(map(str, row)) + "\n"
        )

    df = read_kline_zip(archive)
    assert list(df.columns) == KLINE_COLUMNS
    assert df.loc[0, "num_trades"] == 200
