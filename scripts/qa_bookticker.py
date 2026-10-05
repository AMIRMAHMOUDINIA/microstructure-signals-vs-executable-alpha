from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from microalpha.data.qa import qa_bookticker_zip, save_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--report", type=Path, default=Path("results/qa_bookticker.json"))
    p.add_argument("--nrows", type=int, default=None)
    args = p.parse_args()

    _, report = qa_bookticker_zip(args.archive, nrows=args.nrows)
    save_report(report, args.report)
    print(json.dumps(asdict(report), indent=2))
    print(f"\nSaved QA report: {args.report}")


if __name__ == "__main__":
    main()
