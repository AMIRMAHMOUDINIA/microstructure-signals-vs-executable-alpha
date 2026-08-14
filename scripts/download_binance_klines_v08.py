from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
import requests


BASE = "https://data.binance.vision/data/futures/um/monthly/klines"


def month_range(start: str, end: str):
    """Inclusive YYYY-MM month range."""
    import pandas as pd
    s = pd.Period(start, freq="M")
    e = pd.Period(end, freq="M")
    if e < s:
        raise ValueError("end must be >= start")
    for p in pd.period_range(s, e, freq="M"):
        yield str(p)


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, timeout=180):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[skip] {path}")
        return
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"[get] {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)
    print(f"[save] {path}")


def verify(zip_path: Path, checksum_url: str, timeout=30):
    r = requests.get(checksum_url, timeout=timeout)
    r.raise_for_status()
    text = r.text.strip()
    m = re.search(r"\b([0-9a-fA-F]{64})\b", text)
    if not m:
        raise ValueError(f"No SHA256 found in {checksum_url}")
    expected = m.group(1).lower()
    actual = sha256_file(zip_path).lower()
    print(f"[sha256] {zip_path.name}: {'PASS' if expected == actual else 'FAIL'}")
    if expected != actual:
        raise SystemExit(f"Checksum mismatch: {zip_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--start", default="2023-05")
    p.add_argument("--end", default="2024-03")
    p.add_argument("--output", type=Path, default=Path("data/raw/binance_klines"))
    p.add_argument("--skip-checksum", action="store_true")
    args = p.parse_args()

    for month in month_range(args.start, args.end):
        fn = f"{args.symbol}-{args.interval}-{month}.zip"
        url = f"{BASE}/{args.symbol}/{args.interval}/{fn}"
        path = args.output / fn
        download(url, path)
        if not args.skip_checksum:
            verify(path, url + ".CHECKSUM")


if __name__ == "__main__":
    main()
