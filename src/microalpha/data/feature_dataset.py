from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_REQUIRED = [
    "timestamp",
    "bt_spread_bps_close",
    "bt_spread_bps_twap",
    "bt_bid_qty_close",
    "bt_ask_qty_close",
    "bt_imbalance_close",
    "bt_imbalance_twap",
    "bt_microprice_close",
    "bt_microprice_premium_close",
    "bt_update_rate",
]


@dataclass
class FeatureDatasetQA:
    observed_rows: int
    grid_rows: int
    missing_grid_rows: int
    completeness: float
    start_utc: str
    end_utc: str
    duplicate_timestamps_removed: int
    non_60s_source_gaps: int
    reconstructed_mid_nonpositive: int
    premium_identity_mae: float
    premium_identity_max_abs: float
    premium_identity_corr: float


def _read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def reconstruct_mid(df: pd.DataFrame) -> pd.Series:
    """
    Dataset definition:
        premium = (microprice - mid) / mid
    therefore:
        mid = microprice / (1 + premium)

    This is preferred as the response price because microprice itself embeds
    contemporaneous queue imbalance.
    """
    prem = pd.to_numeric(df["bt_microprice_premium_close"], errors="coerce")
    micro = pd.to_numeric(df["bt_microprice_close"], errors="coerce")
    denom = 1.0 + prem
    mid = micro / denom
    mid = mid.where(denom > 0)
    return mid


def expected_premium_from_identity(df: pd.DataFrame) -> pd.Series:
    """
    For L1 top of book:
        microprice_premium = spread_bps * imbalance / 20,000

    This shows that close microprice premium is a deterministic composite of
    close spread and close imbalance (up to source rounding).
    """
    spread_bps = pd.to_numeric(df["bt_spread_bps_close"], errors="coerce")
    imb = pd.to_numeric(df["bt_imbalance_close"], errors="coerce")
    return spread_bps * imb / 20_000.0


def load_and_prepare_feature_dataset(path: str | Path) -> tuple[pd.DataFrame, FeatureDatasetQA]:
    raw = _read(path).copy()
    missing = set(RAW_REQUIRED).difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])

    n_before = len(raw)
    raw = raw.sort_values("timestamp", kind="stable")
    raw = raw.drop_duplicates("timestamp", keep="last")
    duplicate_removed = n_before - len(raw)

    source_diff = raw["timestamp"].diff().dropna().dt.total_seconds()
    non_60s = int((source_diff != 60).sum())

    raw["mid"] = reconstruct_mid(raw)
    raw["premium_identity_expected"] = expected_premium_from_identity(raw)
    raw["premium_identity_residual"] = (
        pd.to_numeric(raw["bt_microprice_premium_close"], errors="coerce")
        - raw["premium_identity_expected"]
    )

    # Explicit complete minute grid. Missing minutes stay missing; never forward-fill.
    start = raw["timestamp"].min().floor("min")
    end = raw["timestamp"].max().floor("min")
    grid_index = pd.date_range(start, end, freq="1min", tz="UTC")
    raw = raw.set_index("timestamp")
    raw["observed"] = True
    grid = raw.reindex(grid_index)
    grid.index.name = "timestamp"
    grid["observed"] = grid["observed"].eq(True)
    grid = grid.reset_index()

    # Derived model features. These use only information at t.
    grid["log_update_rate"] = np.log1p(
        pd.to_numeric(grid["bt_update_rate"], errors="coerce").clip(lower=0)
    )
    depth = (
        pd.to_numeric(grid["bt_bid_qty_close"], errors="coerce")
        + pd.to_numeric(grid["bt_ask_qty_close"], errors="coerce")
    )
    grid["log_depth_close"] = np.log1p(depth.clip(lower=0))
    grid["imbalance_close_minus_twap"] = (
        pd.to_numeric(grid["bt_imbalance_close"], errors="coerce")
        - pd.to_numeric(grid["bt_imbalance_twap"], errors="coerce")
    )

    observed = grid["observed"]
    resid = grid.loc[observed, "premium_identity_residual"].dropna()
    prem = pd.to_numeric(
        grid.loc[observed, "bt_microprice_premium_close"], errors="coerce"
    )
    expected = grid.loc[observed, "premium_identity_expected"]
    corr_mask = prem.notna() & expected.notna()
    corr = float(prem[corr_mask].corr(expected[corr_mask])) if corr_mask.sum() >= 2 else np.nan

    qa = FeatureDatasetQA(
        observed_rows=int(observed.sum()),
        grid_rows=len(grid),
        missing_grid_rows=int((~observed).sum()),
        completeness=float(observed.mean()),
        start_utc=str(start),
        end_utc=str(end),
        duplicate_timestamps_removed=int(duplicate_removed),
        non_60s_source_gaps=non_60s,
        reconstructed_mid_nonpositive=int((grid.loc[observed, "mid"] <= 0).sum()),
        premium_identity_mae=float(resid.abs().mean()) if len(resid) else np.nan,
        premium_identity_max_abs=float(resid.abs().max()) if len(resid) else np.nan,
        premium_identity_corr=corr,
    )
    return grid, qa


def qa_to_dict(qa: FeatureDatasetQA) -> dict:
    return asdict(qa)
