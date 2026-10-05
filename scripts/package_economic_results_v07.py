from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

INCLUDE = [
    "prior_results/MODEL_RESULTS_SUMMARY.md",
    "prior_results/validation_gate.json",
    "prior_results/logistic_model_metrics.csv",
    "results/economic_v07/november_threshold_grid.csv",
    "results/economic_v07/november_selected_thresholds.csv",
    "results/economic_v07/december_confirmation.csv",
    "results/economic_v07/economic_gate.json",
    "results/economic_v07/economic_test_oos.csv",
    "results/economic_v07/economic_test_monthly.csv",
    "results/economic_v07/ECONOMIC_SUMMARY.md",
    "ECONOMIC_PROTOCOL_V07.md",
    "FEE_AND_EXECUTION_ASSUMPTIONS_V07.md",
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
        default=Path("microstructure_economic_results_v07.zip"),
    )
    args = p.parse_args()

    files = []
    for rel in INCLUDE:
        pth = args.root / rel
        if pth.exists() and pth.is_file():
            files.append((rel, pth))

    manifest = {
        "protocol": "v0.7",
        "files": [
            {"path": rel, "bytes": pth.stat().st_size, "sha256": sha256(pth)}
            for rel, pth in files
        ],
    }

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, pth in files:
            z.write(pth, rel)
        z.writestr("ECONOMIC_RESULTS_MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"Packaged {len(files)} files: {args.output}")
    print(f"Size: {args.output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
