from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "pyarrow",
    "huggingface_hub",
    "requests",
]


def main():
    out = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
    }
    for name in PACKAGES:
        try:
            out["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out["packages"][name] = None

    path = Path("results/environment.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
