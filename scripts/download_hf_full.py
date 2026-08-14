from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import requests


DEFAULT_URL = (
    "https://huggingface.co/datasets/"
    "Mindbyte-89/btcusdt_perp_bookticker_features_1m_05_2023_to_03_2024/"
    "resolve/main/data/train-00000-of-00001.parquet?download=true"
)
EXPECTED_SHA256 = "274eb8e87c7d7185a0162271144b30a0e387ae496fe657c6af83833448f08624"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url: str, output: Path, timeout: int = 180) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"[skip] {output}")
        return

    tmp = output.with_suffix(output.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(output)
    print(f"[save] {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/hf_btcusdt_bookticker_features.parquet"),
    )
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--skip-hash-check", action="store_true")
    args = p.parse_args()

    download(args.url, args.output)

    if not args.skip_hash_check:
        actual = sha256_file(args.output)
        print(f"expected SHA256: {EXPECTED_SHA256}")
        print(f"actual   SHA256: {actual}")
        if actual.lower() != EXPECTED_SHA256.lower():
            raise SystemExit("SHA-256 mismatch. Delete the file and re-download.")
        print("SHA-256: PASS")


if __name__ == "__main__":
    main()
