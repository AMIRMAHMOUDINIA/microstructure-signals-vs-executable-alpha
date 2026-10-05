from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

FILES = [
    "results/trade_flow_v08/flow_qa.json",
    "results/trade_flow_v08/incremental_pre_oos.csv",
    "results/trade_flow_v08/tfi5_monthly_stability.csv",
    "results/trade_flow_v08/augmented_coefficients.csv",
    "results/trade_flow_v08/extension_gate.json",
    "results/trade_flow_v08/incremental_extension_oos.csv",
    "results/trade_flow_v08/TRADE_FLOW_SUMMARY.md",
    "TRADE_FLOW_PROTOCOL_V08.md",
    "prior_results/economic_v07/results/economic_v07/economic_gate.json",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1024*1024), b""):
            h.update(c)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("microstructure_trade_flow_results_v08.zip"),
    )
    args = p.parse_args()

    existing = []
    for rel in FILES:
        path = args.root / rel
        if path.exists() and path.is_file():
            existing.append((rel, path))

    manifest = {
        "protocol": "v0.8",
        "files": [
            {"path": rel, "bytes": path.stat().st_size, "sha256": sha(path)}
            for rel, path in existing
        ],
    }

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, path in existing:
            z.write(path, rel)
        z.writestr("TRADE_FLOW_RESULTS_MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"Packaged {len(existing)} files: {args.output}")


if __name__ == "__main__":
    main()
