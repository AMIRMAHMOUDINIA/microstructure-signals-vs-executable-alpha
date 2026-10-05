import json
import subprocess
import sys
import zipfile
from pathlib import Path


def test_package_results_creates_manifest(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "results" / "validation_gate.json").write_text(
        json.dumps({"gate_pass": False}), encoding="utf-8"
    )
    (root / "README.md").write_text("# x\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "package_results.py"
    output = tmp_path / "bundle.zip"
    r = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert output.exists()

    with zipfile.ZipFile(output) as z:
        names = set(z.namelist())
        assert "RESULTS_MANIFEST.json" in names
        assert "results/validation_gate.json" in names
        manifest = json.loads(z.read("RESULTS_MANIFEST.json"))
        assert len(manifest["files"]) >= 1
