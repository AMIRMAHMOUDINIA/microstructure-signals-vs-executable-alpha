from __future__ import annotations
import numpy as np
import pandas as pd


BOOKTICKER_ALIASES = {
    "update_id": ["update_id", "u", "updateId"],
    "event_time": ["event_time", "E", "eventTime"],
    "transaction_time": ["transaction_time", "T", "transactionTime", "time"],
    "symbol": ["symbol", "s"],
    "bid_price": ["bid_price", "best_bid_price", "b"],
    "bid_qty": ["bid_qty", "best_bid_qty", "B"],
    "ask_price": ["ask_price", "best_ask_price", "a"],
    "ask_qty": ["ask_qty", "best_ask_qty", "A"],
}

TRADE_ALIASES = {
    "trade_id": ["trade_id", "id", "trade Id"],
    "price": ["price"],
    "qty": ["qty", "quantity"],
    "quote_qty": ["quote_qty", "quoteQty"],
    "time": ["time", "timestamp"],
    "is_buyer_maker": ["is_buyer_maker", "isBuyerMaker"],
}


def _resolve(columns, aliases):
    lookup = {str(c).strip(): c for c in columns}
    mapping = {}
    missing = []
    for canonical, candidates in aliases.items():
        found = None
        for c in candidates:
            if c in lookup:
                found = lookup[c]
                break
        if found is not None:
            mapping[found] = canonical
        else:
            missing.append(canonical)
    return mapping, missing


def infer_epoch_unit(values: pd.Series) -> str:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        raise ValueError("Cannot infer timestamp unit from empty/non-numeric series.")
    med = float(x.abs().median())
    # Approximate modern Unix epoch magnitudes.
    if med >= 1e17:
        return "ns"
    if med >= 1e14:
        return "us"
    if med >= 1e11:
        return "ms"
    if med >= 1e8:
        return "s"
    raise ValueError(f"Timestamp magnitude {med:g} is not a plausible Unix epoch.")


def to_timestamp_ns(values: pd.Series) -> pd.Series:
    unit = infer_epoch_unit(values)
    numeric = pd.to_numeric(values, errors="coerce")
    dt = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return dt.astype("int64")


def normalize_bookticker(df: pd.DataFrame) -> pd.DataFrame:
    mapping, missing = _resolve(df.columns, BOOKTICKER_ALIASES)
    essential = {"bid_price", "bid_qty", "ask_price", "ask_qty"}
    if essential.intersection(missing):
        raise ValueError(
            "Could not identify essential bookTicker columns. "
            f"Columns seen: {list(df.columns)}. "
            "Inspect the archive and adapt aliases/default schema if required."
        )
    out = df.rename(columns=mapping).copy()

    for c in ["bid_price", "bid_qty", "ask_price", "ask_qty"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    time_col = "transaction_time" if "transaction_time" in out.columns else (
        "event_time" if "event_time" in out.columns else None
    )
    if time_col is None:
        raise ValueError("No event/transaction timestamp identified in bookTicker data.")

    out["source_time"] = pd.to_numeric(out[time_col], errors="coerce")
    out = out.dropna(subset=["source_time", "bid_price", "ask_price", "bid_qty", "ask_qty"])
    unit = infer_epoch_unit(out["source_time"])
    out["timestamp"] = pd.to_datetime(out["source_time"], unit=unit, utc=True)

    if "update_id" in out.columns:
        out["update_id"] = pd.to_numeric(out["update_id"], errors="coerce")
        out = out.sort_values(["timestamp", "update_id"], kind="stable")
    else:
        out = out.sort_values(["timestamp"], kind="stable")

    # IMPORTANT: no drop_duplicates(timestamp). Same-timestamp events can be legitimate.
    return out.reset_index(drop=True)


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    mapping, missing = _resolve(df.columns, TRADE_ALIASES)
    essential = {"price", "qty", "time", "is_buyer_maker"}
    if essential.intersection(missing):
        raise ValueError(
            f"Could not identify trade columns. Columns seen: {list(df.columns)}"
        )
    out = df.rename(columns=mapping).copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce")
    out["source_time"] = pd.to_numeric(out["time"], errors="coerce")

    raw_bool = out["is_buyer_maker"]
    if raw_bool.dtype == bool:
        parsed = raw_bool
    else:
        parsed = raw_bool.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    out["is_buyer_maker"] = parsed

    out = out.dropna(subset=["price", "qty", "source_time", "is_buyer_maker"])
    unit = infer_epoch_unit(out["source_time"])
    out["timestamp"] = pd.to_datetime(out["source_time"], unit=unit, utc=True)
    out["signed_qty"] = out["qty"] * out["is_buyer_maker"].map({True: -1.0, False: 1.0})
    out["signed_notional"] = out["signed_qty"] * out["price"]

    keys = ["timestamp"]
    if "trade_id" in out.columns:
        out["trade_id"] = pd.to_numeric(out["trade_id"], errors="coerce")
        keys.append("trade_id")
    return out.sort_values(keys, kind="stable").reset_index(drop=True)
