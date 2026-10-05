from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

BOOKTICKER_DEFAULT_COLUMNS = [
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
]

TRADES_DEFAULT_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "quote_qty",
    "time",
    "is_buyer_maker",
]


def _looks_like_header(first_row: list[str]) -> bool:
    joined = ",".join(str(x).strip().lower() for x in first_row)
    tokens = ["price", "qty", "time", "symbol", "bid", "ask", "id"]
    return any(t in joined for t in tokens)


def read_first_csv_from_zip(
    path: str | Path,
    dataset: str,
    nrows: int | None = None,
) -> pd.DataFrame:
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"No CSV found in {path}")
        target = csvs[0]

        # Peek first row without consuming the archive permanently.
        with z.open(target) as f:
            first_line = f.readline().decode("utf-8", errors="replace").strip()
        first_row = first_line.split(",")
        has_header = _looks_like_header(first_row)

        if dataset == "bookTicker":
            defaults = BOOKTICKER_DEFAULT_COLUMNS
        elif dataset == "trades":
            defaults = TRADES_DEFAULT_COLUMNS
        else:
            defaults = None

        with z.open(target) as f:
            if has_header:
                return pd.read_csv(f, nrows=nrows)

            df = pd.read_csv(f, header=None, nrows=nrows)
            if defaults is not None:
                if len(df.columns) == len(defaults):
                    df.columns = defaults
                else:
                    df.columns = [f"col_{i}" for i in range(len(df.columns))]
            return df
