from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


@dataclass
class FlowQA:
    rows: int
    start_utc: str
    end_utc: str
    duplicate_minutes_removed: int
    missing_minutes: int
    completeness: float
    nonpositive_volume_rows: int
    taker_buy_gt_volume_rows: int
    median_same_minute_close_mid_diff_bps: float
    p99_same_minute_close_mid_diff_bps: float


def _first_csv_name(z: zipfile.ZipFile):
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise ValueError("No CSV in archive")
    return names[0]


def read_kline_zip(path: str | Path) -> pd.DataFrame:
    """
    Read an official Binance USD-M futures kline archive robustly.

    Binance documents klines as a 12-field positional record:
      open time, OHLC, volume, close time, quote volume,
      number of trades, taker-buy base, taker-buy quote, ignore.

    Archive header labels have varied across exported datasets/tools, so for
    exactly 12-column kline files we canonicalize BY POSITION rather than
    depending on a specific spelling such as "num_trades".
    """
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        name = _first_csv_name(z)

        # Read first row to detect whether it is data or a header.
        with z.open(name) as f:
            first_line = f.readline().decode("utf-8", errors="replace").strip()
        first = [v.strip() for v in first_line.split(",")]

        def _is_number(s: str) -> bool:
            try:
                float(s)
                return True
            except Exception:
                return False

        # Official numeric kline rows begin with Unix open time.
        has_header = not (first and _is_number(first[0]))

        with z.open(name) as f:
            if has_header:
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, header=None)

        if df.shape[1] != len(KLINE_COLUMNS):
            raise ValueError(
                f"Expected exactly {len(KLINE_COLUMNS)} Binance kline fields, "
                f"found {df.shape[1]} in {path.name}. "
                f"Columns seen: {list(df.columns)}"
            )

        # Canonicalize by the official documented field order.
        df.columns = KLINE_COLUMNS
        return df


def load_monthly_klines(directory: str | Path) -> pd.DataFrame:
    directory = Path(directory)
    parts = []
    files = sorted(directory.glob("BTCUSDT-1m-*.zip"))
    if not files:
        raise ValueError(f"No BTCUSDT 1m ZIPs found in {directory}")
    for path in files:
        x = read_kline_zip(path)
        x["source_file"] = path.name
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def normalize_klines(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.copy()
    for c in [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote",
    ]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna(
        subset=[
            "open_time", "close", "volume", "quote_volume",
            "num_trades", "taker_buy_base", "taker_buy_quote",
        ]
    )
    # Infer Unix timestamp unit from magnitude instead of hard-coding an era.
    med = float(pd.to_numeric(x["open_time"], errors="coerce").dropna().abs().median())
    if med >= 1e17:
        unit = "ns"
    elif med >= 1e14:
        unit = "us"
    elif med >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    x["timestamp"] = pd.to_datetime(x["open_time"], unit=unit, utc=True)
    x = x.sort_values("timestamp", kind="stable")
    before = len(x)
    x = x.drop_duplicates("timestamp", keep="last")
    x.attrs["duplicate_minutes_removed"] = before - len(x)
    return x.reset_index(drop=True)


def add_trade_flow_features(klines: pd.DataFrame) -> pd.DataFrame:
    x = klines.copy()

    vol = pd.to_numeric(x["volume"], errors="coerce")
    qvol = pd.to_numeric(x["quote_volume"], errors="coerce")
    buy = pd.to_numeric(x["taker_buy_base"], errors="coerce")
    buyq = pd.to_numeric(x["taker_buy_quote"], errors="coerce")
    ntr = pd.to_numeric(x["num_trades"], errors="coerce")

    x["seller_taker_base"] = (vol - buy).clip(lower=0)
    x["signed_taker_base"] = 2.0 * buy - vol
    x["signed_taker_quote"] = 2.0 * buyq - qvol

    x["tfi_1m"] = np.where(vol > 0, x["signed_taker_base"] / vol, np.nan)

    # Causal, volume-weighted rolling trade-flow imbalance.
    for w in (5, 10):
        signed_sum = x["signed_taker_base"].rolling(w, min_periods=w).sum()
        volume_sum = vol.rolling(w, min_periods=w).sum()
        x[f"tfi_{w}m"] = np.where(volume_sum > 0, signed_sum / volume_sum, np.nan)

    x["log_quote_volume"] = np.log1p(qvol.clip(lower=0))
    x["log_num_trades"] = np.log1p(ntr.clip(lower=0))
    x["avg_trade_base"] = np.where(ntr > 0, vol / ntr, np.nan)

    return x


def merge_flow_with_book(book_grid: pd.DataFrame, flow: pd.DataFrame):
    """
    Exact minute merge. The book timestamp and Binance kline open_time both label
    the one-minute bucket. No as-of/forward fill is used.
    """
    b = book_grid.copy()
    f = flow.copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)

    keep = [
        "timestamp", "close", "volume", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote",
        "signed_taker_base", "signed_taker_quote",
        "tfi_1m", "tfi_5m", "tfi_10m",
        "log_quote_volume", "log_num_trades", "avg_trade_base",
    ]
    m = b.merge(f[keep], on="timestamp", how="left", validate="one_to_one")

    m["tfi5_x_book_imbalance"] = (
        m["tfi_5m"] * pd.to_numeric(m["bt_imbalance_close"], errors="coerce")
    )
    return m


def flow_qa(merged: pd.DataFrame, duplicate_removed: int = 0) -> FlowQA:
    obs = merged["close"].notna()
    k = merged.loc[obs].copy()

    total_grid = len(merged)
    missing = int((~obs).sum())

    close = pd.to_numeric(k["close"], errors="coerce")
    mid = pd.to_numeric(k["mid"], errors="coerce")
    diff_bps = ((close - mid).abs() / mid * 10_000).replace([np.inf, -np.inf], np.nan).dropna()

    vol = pd.to_numeric(k["volume"], errors="coerce")
    buy = pd.to_numeric(k["taker_buy_base"], errors="coerce")

    return FlowQA(
        rows=int(obs.sum()),
        start_utc=str(k["timestamp"].min()) if len(k) else "NA",
        end_utc=str(k["timestamp"].max()) if len(k) else "NA",
        duplicate_minutes_removed=int(duplicate_removed),
        missing_minutes=missing,
        completeness=float(obs.mean()) if total_grid else np.nan,
        nonpositive_volume_rows=int((vol <= 0).sum()),
        taker_buy_gt_volume_rows=int((buy > vol + 1e-12).sum()),
        median_same_minute_close_mid_diff_bps=float(diff_bps.median()) if len(diff_bps) else np.nan,
        p99_same_minute_close_mid_diff_bps=float(diff_bps.quantile(0.99)) if len(diff_bps) else np.nan,
    )


def qa_to_dict(qa: FlowQA):
    return asdict(qa)
