from __future__ import annotations

import numpy as np
import pandas as pd


def add_exact_forward_target(
    grid: pd.DataFrame,
    horizon_minutes: int,
    price_col: str = "mid",
) -> pd.DataFrame:
    """
    Build an exact clock-time target on an already complete 1-minute grid.

    If either t or t+h is missing in the source dataset, the return is NaN.
    No row-based 'skip across a gap' behavior is allowed.
    """
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    if "timestamp" not in grid.columns:
        raise ValueError("timestamp column is required")
    if "observed" not in grid.columns:
        raise ValueError("observed column is required")

    x = grid.sort_values("timestamp").reset_index(drop=True).copy()
    ts = pd.to_datetime(x["timestamp"], utc=True)

    # Because the input is a complete one-minute grid, shift(h) is now an exact
    # h-minute shift. Missing source observations remain NaN/observed=False.
    future_price = pd.to_numeric(x[price_col], errors="coerce").shift(-horizon_minutes)
    future_observed = x["observed"].shift(-horizon_minutes, fill_value=False).astype(bool)

    current_price = pd.to_numeric(x[price_col], errors="coerce")
    valid = (
        x["observed"]
        & future_observed
        & current_price.notna()
        & future_price.notna()
        & (current_price > 0)
        & (future_price > 0)
    )

    target = np.log(future_price / current_price).where(valid)

    x[f"target_timestamp_{horizon_minutes}m"] = ts + pd.to_timedelta(
        horizon_minutes, unit="min"
    )
    x[f"future_mid_{horizon_minutes}m"] = future_price.where(valid)
    x[f"fwd_mid_logret_{horizon_minutes}m"] = target
    return x
