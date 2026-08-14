from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from microalpha.features.microstructure import add_book_features, add_forward_log_return
from microalpha.models.baselines import classification_models


def make_data(n=25_000, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1s", tz="UTC")
    latent = rng.normal(size=n)
    bid_qty = np.exp(1.0 + 0.35 * latent + rng.normal(scale=0.35, size=n))
    ask_qty = np.exp(1.0 - 0.35 * latent + rng.normal(scale=0.35, size=n))

    base_ret = 1e-5 * latent + rng.normal(scale=6e-5, size=n)
    mid = 60_000 * np.exp(np.cumsum(base_ret))
    spread = np.maximum(0.5, rng.lognormal(mean=-0.4, sigma=0.25, size=n))
    bid = mid - spread / 2
    ask = mid + spread / 2

    return pd.DataFrame({
        "timestamp": ts,
        "bid_price": bid,
        "bid_qty": bid_qty,
        "ask_price": ask,
        "ask_qty": ask_qty,
    })


def main():
    df = add_book_features(make_data())
    df = add_forward_log_return(df, horizon_seconds=5)
    df = df.dropna().reset_index(drop=True)

    features = ["book_imbalance", "microprice_disp", "rel_spread"]
    target = "fwd_logret_5s"

    split = int(len(df) * 0.8)
    X_train, X_test = df.loc[:split-1, features], df.loc[split:, features]
    y_train = (df.loc[:split-1, target] > 0).astype(int)
    y_test = (df.loc[split:, target] > 0).astype(int)

    model = classification_models()[0].estimator
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    print("Synthetic pipeline smoke test")
    print(f"rows:     {len(df):,}")
    print(f"accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"roc_auc:  {roc_auc_score(y_test, prob):.4f}")
    print("\nSynthetic data only; these are NOT trading results.")


if __name__ == "__main__":
    main()
