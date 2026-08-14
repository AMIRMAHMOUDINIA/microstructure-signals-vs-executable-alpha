from __future__ import annotations
import argparse
from pathlib import Path
import zipfile

from microalpha.data.archive import read_first_csv_from_zip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--dataset", choices=["bookTicker", "trades"], default="bookTicker")
    args = p.parse_args()

    with zipfile.ZipFile(args.archive) as z:
        print("Archive members:")
        for info in z.infolist():
            print(f" - {info.filename} ({info.file_size:,} bytes uncompressed)")

    df = read_first_csv_from_zip(args.archive, dataset=args.dataset, nrows=10)
    print("\nDetected columns:")
    print(list(df.columns))
    print("\nFirst rows:")
    print(df.to_string(index=False))
    print("\nDtypes:")
    print(df.dtypes)


if __name__ == "__main__":
    main()
