from __future__ import annotations
import argparse
from pathlib import Path

from microalpha.data.archive import read_first_csv_from_zip
from microalpha.data.normalize import normalize_bookticker
from microalpha.data.grid import last_quote_grid
from microalpha.features.microstructure import add_book_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--freq", default="1s")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    raw = read_first_csv_from_zip(args.archive, dataset="bookTicker")
    events = normalize_bookticker(raw)
    grid = last_quote_grid(events, freq=args.freq)
    grid = add_book_features(grid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        grid.to_csv(args.output, index=False)
    else:
        try:
            grid.to_parquet(args.output, index=False)
        except ImportError as e:
            raise SystemExit(
                "Parquet output requires pyarrow. Install it or use an .csv output path."
            ) from e

    print(f"events: {len(events):,}")
    print(f"grid rows: {len(grid):,}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
