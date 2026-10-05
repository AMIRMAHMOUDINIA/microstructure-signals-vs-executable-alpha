from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from microalpha.data.trade_flow_v08 import read_kline_zip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()

    files = sorted(args.directory.glob("BTCUSDT-1m-*.zip"))
    if not files:
        raise SystemExit(f"No monthly BTCUSDT 1m ZIP files found in {args.directory}")

    path = files[0]
    print("Inspecting:", path)
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(member) as f:
            first = f.readline().decode("utf-8", errors="replace").strip()
        print("Raw first line:")
        print(first[:1000])

    df = read_kline_zip(path)
    print("\nCanonical columns:")
    print(list(df.columns))
    print("\nFirst 3 canonical rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
