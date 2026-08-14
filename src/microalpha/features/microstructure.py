from __future__ import annotations
import numpy as np
import pandas as pd


def add_book_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"bid_price", "bid_qty", "ask_price", "ask_qty"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    out["mid"] = (out["bid_price"] + out["ask_price"]) / 2.0
    out["spread"] = out["ask_price"] - out["bid_price"]
    out["rel_spread"] = out["spread"] / out["mid"]

    denom = out["bid_qty"] + out["ask_qty"]
    out["book_imbalance"] = np.where(
        denom > 0,
        (out["bid_qty"] - out["ask_qty"]) / denom,
        np.nan,
    )

    out["microprice"] = np.where(
        denom > 0,
        (out["ask_price"] * out["bid_qty"] + out["bid_price"] * out["ask_qty"]) / denom,
        np.nan,
    )
    out["microprice_disp"] = (out["microprice"] - out["mid"]) / out["mid"]

    bad = (out["spread"] < 0) | (out["bid_qty"] < 0) | (out["ask_qty"] < 0)
    if bad.any():
        raise ValueError(f"Found {int(bad.sum())} crossed/invalid top-of-book rows.")

    return out


def add_forward_log_return(
    df: pd.DataFrame,
    horizon_seconds: int,
    timestamp_col: str = "timestamp",
    price_col: str = "mid",
) -> pd.DataFrame:
    """
    Forward target using the first observation at or after t+h.

    On a regular 1-second grid this is effectively an exact horizon, barring gaps.
    """
    out = df.sort_values(timestamp_col).copy().reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(out[timestamp_col]):
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True)

    left = out[[timestamp_col]].copy()
    left["target_ts"] = left[timestamp_col] + pd.to_timedelta(horizon_seconds, unit="s")

    future = out[[timestamp_col, price_col]].rename(
        columns={timestamp_col: "future_ts", price_col: "future_price"}
    )

    aligned = pd.merge_asof(
        left.sort_values("target_ts"),
        future.sort_values("future_ts"),
        left_on="target_ts",
        right_on="future_ts",
        direction="forward",
        allow_exact_matches=True,
    ).sort_index()

    out[f"future_price_{horizon_seconds}s"] = aligned["future_price"].to_numpy()
    out[f"fwd_logret_{horizon_seconds}s"] = np.log(
        out[f"future_price_{horizon_seconds}s"] / out[price_col]
    )
    return out


def aggregate_trade_flow(
    trades: pd.DataFrame,
    freq: str = "1s",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    required = {timestamp_col, "signed_qty", "signed_notional", "qty"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"Missing trade columns: {sorted(missing)}")

    x = trades.copy()
    if not pd.api.types.is_datetime64_any_dtype(x[timestamp_col]):
        x[timestamp_col] = pd.to_datetime(x[timestamp_col], utc=True)
    x = x.set_index(timestamp_col)

    agg = x.resample(freq).agg(
        signed_qty=("signed_qty", "sum"),
        signed_notional=("signed_notional", "sum"),
        total_qty=("qty", "sum"),
        trade_count=("qty", "size"),
    )
    agg["trade_flow_imbalance"] = np.where(
        agg["total_qty"] > 0, agg["signed_qty"] / agg["total_qty"], 0.0
    )
    return agg.reset_index()
