from __future__ import annotations
import pandas as pd


def last_quote_grid(df: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    """
    Convert event-level quote updates to a regular UTC grid using the last known quote.

    This is explicit state sampling, not raw-event de-duplication.
    """
    required = {"timestamp", "bid_price", "bid_qty", "ask_price", "ask_qty"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x = df.sort_values("timestamp", kind="stable").copy()
    if "update_id" in x.columns:
        x = x.sort_values(["timestamp", "update_id"], kind="stable")

    # Last event at each exact timestamp, then sample last known state per interval.
    # Same-timestamp events remain preserved in raw storage/QA; this is only the research grid.
    x = x.set_index("timestamp")
    cols = ["bid_price", "bid_qty", "ask_price", "ask_qty"]
    state = x[cols].resample(freq).last().ffill()

    # Do not back-fill before first observation.
    state = state.dropna()
    return state.reset_index()
