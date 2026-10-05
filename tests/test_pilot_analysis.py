import importlib.util
from pathlib import Path

import pandas as pd


def load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "analyze_feature_dataset.py"
    spec = importlib.util.spec_from_file_location("analyze_feature_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_analysis_columns():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "sample" / "hf_viewer_first_100.csv")
    mod = load_module()
    out = mod.analyze(df, horizons=[1, 5])
    assert len(out) == 6
    assert {"pearson_r", "hac_p", "ols_beta"}.issubset(out.columns)
