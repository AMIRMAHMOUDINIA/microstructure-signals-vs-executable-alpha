from pathlib import Path
import subprocess
import sys
import zipfile


def test_economic_packager_runs_with_partial_results(tmp_path: Path):
    root = tmp_path / "r"
    (root / "results/economic_v07").mkdir(parents=True)
    (root / "results/economic_v07/ECONOMIC_SUMMARY.md").write_text("# test\n")
    (root / "ECONOMIC_PROTOCOL_V07.md").write_text("# p\n")

    script = Path(__file__).resolve().parents[1] / "scripts/package_economic_results_v07.py"
    out = tmp_path / "bundle.zip"
    r = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    with zipfile.ZipFile(out) as z:
        assert "ECONOMIC_RESULTS_MANIFEST.json" in z.namelist()
