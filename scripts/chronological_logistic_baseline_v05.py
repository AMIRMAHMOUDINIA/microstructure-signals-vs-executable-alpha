from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from microalpha.data.feature_dataset import load_and_prepare_feature_dataset
from microalpha.research.targets import add_exact_forward_target
from microalpha.validation.periods import locked_period_map, select_purged_period


MODEL_FEATURES = [
    "bt_imbalance_close",
    "bt_imbalance_twap",
    "bt_spread_bps_close",
    "bt_spread_bps_twap",
    "log_update_rate",
    "log_depth_close",
    "imbalance_close_minus_twap",
]

# Deliberately excluded:
# bt_microprice_premium_close
# It is essentially spread_close * imbalance_close / 20,000 and would be
# redundant with the two primitive close-state features.


def make_dataset(path: Path, horizon: int) -> pd.DataFrame:
    grid, _ = load_and_prepare_feature_dataset(path)
    x = add_exact_forward_target(grid, horizon)
    target = f"fwd_mid_logret_{horizon}m"
    x["y"] = np.where(
        x[target].notna(),
        (x[target] > 0).astype(int),
        np.nan,
    )
    return x


def clean_period(df: pd.DataFrame, name: str, horizon: int) -> pd.DataFrame:
    p = locked_period_map()[name]
    x = select_purged_period(df, p, horizon)
    target = f"fwd_mid_logret_{horizon}m"
    required = MODEL_FEATURES + [target, "y"]
    return x.dropna(subset=required).copy()


def build_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")),
    ])


def evaluate(name: str, model, x: pd.DataFrame) -> dict:
    y = x["y"].astype(int)
    p = model.predict_proba(x[MODEL_FEATURES])[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "period": name,
        "n": int(len(x)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def standardized_coefficients(model) -> pd.DataFrame:
    coef = model.named_steps["model"].coef_[0]
    return pd.DataFrame({
        "feature": MODEL_FEATURES,
        "standardized_logit_coef": coef,
        "abs_coef": np.abs(coef),
    }).sort_values("abs_coef", ascending=False)


def gate_allows_test(gate_path: Path | None) -> bool:
    if gate_path is None or not gate_path.exists():
        return False
    obj = json.loads(gate_path.read_text(encoding="utf-8"))
    return bool(obj.get("gate_pass", False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("results/logistic_v05"))
    p.add_argument(
        "--unlock-test",
        action="store_true",
        help="Evaluate untouched test ONLY after validation gate has passed.",
    )
    p.add_argument(
        "--gate",
        type=Path,
        default=None,
        help="Path to validation_gate.json with gate_pass=true.",
    )
    args = p.parse_args()

    df = make_dataset(args.data, args.horizon)
    dev = clean_period(df, "development", args.horizon)
    val = clean_period(df, "validation", args.horizon)

    model = build_model()
    model.fit(dev[MODEL_FEATURES], dev["y"].astype(int))

    rows = [
        evaluate("development", model, dev),
        evaluate("validation", model, val),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.unlock_test:
        if not gate_allows_test(args.gate):
            raise SystemExit(
                "TEST REMAINS LOCKED: provide --gate validation_gate.json "
                "with gate_pass=true."
            )
        test = clean_period(df, "test", args.horizon)
        rows.append(evaluate("test", model, test))

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "metrics.csv", index=False)
    standardized_coefficients(model).to_csv(
        args.output_dir / "standardized_coefficients.csv", index=False
    )

    print(out.to_string(index=False))
    print("\nTest set status:", "UNLOCKED" if args.unlock_test else "LOCKED")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
