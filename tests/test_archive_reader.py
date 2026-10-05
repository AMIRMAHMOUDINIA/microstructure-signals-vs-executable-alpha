import zipfile
from pathlib import Path

from microalpha.data.archive import read_first_csv_from_zip


def test_headerless_bookticker_archive(tmp_path: Path):
    archive = tmp_path / "sample.zip"
    csv_name = "BTCUSDT-bookTicker-sample.csv"
    row = "1001,60000.0,2.0,60000.5,1.5,1700000000000,1700000000001\n"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(csv_name, row)

    df = read_first_csv_from_zip(archive, "bookTicker")
    assert list(df.columns) == [
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "transaction_time",
        "event_time",
    ]
    assert len(df) == 1
