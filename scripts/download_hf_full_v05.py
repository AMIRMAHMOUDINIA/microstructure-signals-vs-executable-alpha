from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ID = "Mindbyte-89/btcusdt_perp_bookticker_features_1m_05_2023_to_03_2024"
FILENAME = "data/train-00000-of-00001.parquet"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/hf_btcusdt_bookticker_features.parquet"),
    )
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    cached = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    # Copy instead of symlink so the research archive remains self-contained.
    import shutil
    shutil.copy2(cached, args.output)

    actual = sha256_file(args.output)
    print(f"expected SHA256: {EXPECTED_SHA256}")
    print(f"actual   SHA256: {actual}")
    if actual.lower() != EXPECTED_SHA256.lower():
        raise SystemExit("SHA-256 mismatch.")
    print("SHA-256: PASS")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
