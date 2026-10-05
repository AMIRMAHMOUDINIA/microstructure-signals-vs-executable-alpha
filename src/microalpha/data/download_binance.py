from __future__ import annotations

import argparse
import hashlib
import re
from datetime import date, timedelta
from pathlib import Path

import requests

BASE = "https://data.binance.vision/data/futures/um/daily"


def iter_dates(start: str, end: str):
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    if e < s:
        raise ValueError("end must be >= start")
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


def build_url(dataset: str, symbol: str, d: date) -> str:
    stamp = d.isoformat()
    return f"{BASE}/{dataset}/{symbol}/{symbol}-{dataset}-{stamp}.zip"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path, timeout: int = 120) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] {out_path}")
        return

    print(f"[get]  {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            print(f"[404]  unavailable: {url}")
            return
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)
    print(f"[save] {out_path}")


def download_text(url: str, out_path: Path, timeout: int = 30) -> str | None:
    with requests.get(url, timeout=timeout) as r:
        if r.status_code == 404:
            print(f"[404]  unavailable: {url}")
            return None
        r.raise_for_status()
        text = r.text.strip()
    out_path.write_text(text + "\n", encoding="utf-8")
    return text


def verify_checksum(zip_path: Path, checksum_text: str) -> bool:
    match = re.search(r"\b([0-9a-fA-F]{64})\b", checksum_text)
    if not match:
        raise ValueError("Could not find SHA-256 digest in CHECKSUM file.")
    expected = match.group(1).lower()
    actual = sha256_file(zip_path).lower()
    ok = actual == expected
    print(f"[sha256] expected={expected}")
    print(f"[sha256] actual  ={actual}")
    print(f"[sha256] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["bookTicker", "bookDepth", "trades", "aggTrades"], required=True)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checksum", action="store_true", help="Download and verify Binance .CHECKSUM")
    args = p.parse_args()

    for d in iter_dates(args.start, args.end):
        url = build_url(args.dataset, args.symbol, d)
        filename = url.rsplit("/", 1)[-1]
        zip_path = args.output / filename
        download_file(url, zip_path)

        if args.checksum and zip_path.exists():
            checksum_url = url + ".CHECKSUM"
            checksum_path = args.output / (filename + ".CHECKSUM")
            checksum_text = download_text(checksum_url, checksum_path)
            if checksum_text is not None and not verify_checksum(zip_path, checksum_text):
                raise SystemExit(f"Checksum verification failed: {zip_path}")


if __name__ == "__main__":
    main()
