from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


INCLUDE_PATTERNS = [
    "results/environment.json",
    "results/validation_gate.json",
    "results/RESULTS_SUMMARY.md",
    "results/full_descriptive_v05/dataset_qa.json",
    "results/full_descriptive_v05/signal_summary.csv",
    "results/full_descriptive_v05/monthly_stability.csv",
    "results/logistic_v05/metrics.csv",
    "results/logistic_v05/standardized_coefficients.csv",
    "results/logistic_v05_test/metrics.csv",
    "results/logistic_v05_test/standardized_coefficients.csv",
    "FULL_DATA_PROTOCOL.md",
    "METHODOLOGY_REVIEW_V05.md",
    "TEST_SET_POLICY.md",
    "README.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("microstructure_results_bundle.zip"),
    )
    args = p.parse_args()

    files = []
    for rel in INCLUDE_PATTERNS:
        path = args.root / rel
        if path.exists() and path.is_file():
            files.append((rel, path))

    if not files:
        raise SystemExit("No result files found to package.")

    manifest = {
        "files": [
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for rel, path in files
        ]
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, path in files:
            z.write(path, arcname=rel)
        z.writestr("RESULTS_MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"Packaged {len(files)} files: {args.output}")
    print(f"Bundle size: {args.output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
