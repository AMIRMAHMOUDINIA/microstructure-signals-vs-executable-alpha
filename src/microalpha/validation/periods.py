from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end_exclusive: str


LOCKED_PERIODS = (
    Period("development", "2023-05-16T11:49:00Z", "2023-11-01T00:00:00Z"),
    Period("validation", "2023-11-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    Period("test", "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z"),
)


def select_purged_period(
    df: pd.DataFrame,
    period: Period,
    horizon_minutes: int,
) -> pd.DataFrame:
    """
    Select rows whose feature timestamp AND label endpoint are inside the period.

    Requiring target_timestamp < period end prevents the last development label
    from consuming validation prices (and likewise at the validation/test boundary).
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    target_col = f"target_timestamp_{horizon_minutes}m"
    if target_col not in df.columns:
        raise ValueError(f"Missing {target_col}")
    target_ts = pd.to_datetime(df[target_col], utc=True)

    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end_exclusive)
    mask = (ts >= start) & (ts < end) & (target_ts < end)
    return df.loc[mask].copy()


def locked_period_map():
    return {p.name: p for p in LOCKED_PERIODS}
