from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from microalpha.data.archive import read_first_csv_from_zip
from microalpha.data.normalize import infer_epoch_unit, normalize_bookticker


@dataclass
class BookTickerQA:
    rows: int
    start_utc: str
    end_utc: str
    inferred_time_unit: str
    same_timestamp_rows: int
    exact_duplicate_rows: int
    duplicate_update_ids: int | None
    out_of_order_rows_before_sort: int
    crossed_rows: int
    locked_rows: int
    nonpositive_bid_qty_rows: int
    nonpositive_ask_qty_rows: int
    median_spread: float
    p99_spread: float
    median_event_gap_ms: float
    p99_event_gap_ms: float
    max_event_gap_ms: float


def analyze_bookticker_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, BookTickerQA]:
    # Capture source-order timing diagnostics before normalization sorts.
    time_candidates = [
        c for c in raw.columns
        if str(c) in {"transaction_time", "event_time", "T", "E", "time"}
    ]
    source_order_negative = 0
    time_unit = "unknown"
    if time_candidates:
        ts_raw = pd.to_numeric(raw[time_candidates[0]], errors="coerce")
        valid = ts_raw.dropna()
        if not valid.empty:
            time_unit = infer_epoch_unit(valid)
            source_order_negative = int((valid.diff().dropna() < 0).sum())

    df = normalize_bookticker(raw)
    spread = df["ask_price"] - df["bid_price"]

    same_ts = int(df["timestamp"].duplicated(keep=False).sum())
    exact_dupes = int(df.duplicated().sum())

    if "update_id" in df.columns:
        dup_update = int(df["update_id"].duplicated(keep=False).sum())
    else:
        dup_update = None

    unique_times = df["timestamp"].drop_duplicates().sort_values()
    gaps_ms = unique_times.diff().dropna().dt.total_seconds().mul(1000)

    def q(series, p, default=float("nan")):
        if len(series) == 0:
            return default
        return float(series.quantile(p))

    report = BookTickerQA(
        rows=len(df),
        start_utc=str(df["timestamp"].min()),
        end_utc=str(df["timestamp"].max()),
        inferred_time_unit=time_unit if time_unit != "unknown" else infer_epoch_unit(df["source_time"]),
        same_timestamp_rows=same_ts,
        exact_duplicate_rows=exact_dupes,
        duplicate_update_ids=dup_update,
        out_of_order_rows_before_sort=source_order_negative,
        crossed_rows=int((spread < 0).sum()),
        locked_rows=int((spread == 0).sum()),
        nonpositive_bid_qty_rows=int((df["bid_qty"] <= 0).sum()),
        nonpositive_ask_qty_rows=int((df["ask_qty"] <= 0).sum()),
        median_spread=float(spread.median()),
        p99_spread=q(spread, 0.99),
        median_event_gap_ms=q(gaps_ms, 0.50),
        p99_event_gap_ms=q(gaps_ms, 0.99),
        max_event_gap_ms=float(gaps_ms.max()) if len(gaps_ms) else float("nan"),
    )
    return df, report


def qa_bookticker_zip(path: str | Path, nrows: int | None = None):
    raw = read_first_csv_from_zip(path, dataset="bookTicker", nrows=nrows)
    return analyze_bookticker_raw(raw)


def save_report(report: BookTickerQA, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
