from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from microalpha.data.trade_flow_v08 import (
    add_trade_flow_features,
    load_monthly_klines,
    normalize_klines,
)
from microalpha.validation.incremental_v08 import paired_day_bootstrap_auc_delta
from microalpha.validation.inference import ols_naive_and_hac

H = 10

PERIODS = {
    "confirmation": ("2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z"),
    "replication": ("2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z"),
    "final_oos": ("2025-07-01T00:00:00Z", "2026-01-01T00:00:00Z"),
}


def complete_grid(k: pd.DataFrame) -> pd.DataFrame:
    x = k.sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
    x = x.set_index("timestamp")
    grid = pd.date_range(x.index.min(), x.index.max(), freq="1min", tz="UTC")
    x["observed"] = True
    x = x.reindex(grid)
    x.index.name = "timestamp"
    x["observed"] = x["observed"].fillna(False).astype(bool)
    return x.reset_index()


def add_features_targets(k: pd.DataFrame) -> pd.DataFrame:
    x = complete_grid(k)

    close = pd.to_numeric(x["close"], errors="coerce")
    x["past_ret_5m"] = np.log(close / close.shift(5))
    x["past_ret_10m"] = np.log(close / close.shift(10))
    x["price_reversal_score"] = -x["past_ret_5m"]

    future = close.shift(-H)
    future_observed = x["observed"].shift(-H).fillna(False).astype(bool)
    valid = (
        x["observed"] & future_observed
        & close.notna() & future.notna()
        & (close > 0) & (future > 0)
    )
    x["fwd_close_logret_10m"] = np.log(future / close).where(valid)
    x["y_up"] = np.where(
        x["fwd_close_logret_10m"].notna(),
        (x["fwd_close_logret_10m"] > 0).astype(int),
        np.nan,
    )

    # Pre-registered discovery from Study II: positive aggressive flow is
    # hypothesized to predict reversal, hence negative TFI is the positive score.
    x["flow_reversal_score"] = -x["tfi_5m"]
    return x


def period(x: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = PERIODS[name]
    ts = pd.to_datetime(x["timestamp"], utc=True)
    target_ts = ts + pd.Timedelta(minutes=H)
    return x[
        (ts >= pd.Timestamp(start))
        & (ts < pd.Timestamp(end))
        & (target_ts < pd.Timestamp(end))
    ].copy()


def univariate_tfi(z: pd.DataFrame) -> dict:
    d = z[["tfi_5m", "fwd_close_logret_10m"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    r = ols_naive_and_hac(
        d["tfi_5m"], d["fwd_close_logret_10m"], hac_lags=H
    )
    rho, p = spearmanr(d["tfi_5m"], d["fwd_close_logret_10m"])
    return {
        "n": r.n,
        "beta_bps_per_unit": r.beta * 10_000,
        "hac_t": r.t_hac,
        "hac_p": r.p_hac,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
    }


def partial_tfi_control_price(z: pd.DataFrame) -> dict:
    d = z[["tfi_5m", "past_ret_5m", "fwd_close_logret_10m"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    X = sm.add_constant(d[["tfi_5m", "past_ret_5m"]])
    fit = sm.OLS(d["fwd_close_logret_10m"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": H}
    )
    return {
        "n": len(d),
        "tfi_beta_bps_per_unit": float(fit.params["tfi_5m"] * 10_000),
        "tfi_hac_t": float(fit.tvalues["tfi_5m"]),
        "tfi_hac_p": float(fit.pvalues["tfi_5m"]),
        "past_ret_beta": float(fit.params["past_ret_5m"]),
        "past_ret_hac_p": float(fit.pvalues["past_ret_5m"]),
        "r2": float(fit.rsquared),
    }


def auc_comparison(z: pd.DataFrame, name: str) -> dict:
    d = z[
        ["timestamp", "y_up", "flow_reversal_score", "price_reversal_score"]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    y = d["y_up"].astype(int)
    flow = d["flow_reversal_score"].astype(float)
    price = d["price_reversal_score"].astype(float)

    flow_auc = float(roc_auc_score(y, flow))
    price_auc = float(roc_auc_score(y, price))

    boot = paired_day_bootstrap_auc_delta(
        d["timestamp"],
        y,
        price,
        flow,
        reps=1000,
        seed=910 + len(name),
    )
    return {
        "period": name,
        "n": len(d),
        "flow_reversal_auc": flow_auc,
        "price_reversal_auc": price_auc,
        "delta_auc_flow_vs_price_reversal": flow_auc - price_auc,
        **boot,
    }


def monthly(z: pd.DataFrame) -> pd.DataFrame:
    x = z.copy()
    x["month"] = pd.to_datetime(x["timestamp"], utc=True).dt.strftime("%Y-%m")
    rows = []
    for month, g in x.groupby("month", sort=True):
        d = g[["tfi_5m", "fwd_close_logret_10m"]].dropna()
        if len(d) < 1000:
            continue
        r = ols_naive_and_hac(
            d["tfi_5m"], d["fwd_close_logret_10m"], hac_lags=H
        )
        rows.append({
            "month": month,
            "n": r.n,
            "beta_bps_per_unit": r.beta * 10_000,
            "hac_p": r.p_hac,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kline-dir",
        type=Path,
        default=Path("data/raw/binance_klines_2025"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/flow_reversal_v091"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_monthly_klines(args.kline_dir)
    k = add_trade_flow_features(normalize_klines(raw))
    x = add_features_targets(k)

    qa = {
        "rows_raw_normalized": len(k),
        "grid_rows": len(x),
        "observed_minutes": int(x["observed"].sum()),
        "missing_minutes": int((~x["observed"]).sum()),
        "completeness": float(x["observed"].mean()),
        "start_utc": str(x["timestamp"].min()),
        "end_utc": str(x["timestamp"].max()),
        "tfi5_nonnull": int(x["tfi_5m"].notna().sum()),
    }
    (args.output_dir / "qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )

    rows = []
    partial_rows = []
    auc_rows = []
    monthly_parts = []

    for name in ["confirmation", "replication"]:
        z = period(x, name)
        rows.append({"period": name, **univariate_tfi(z)})
        partial_rows.append({"period": name, **partial_tfi_control_price(z)})
        auc_rows.append(auc_comparison(z, name))
        m = monthly(z)
        m.insert(0, "period", name)
        monthly_parts.append(m)

    uni = pd.DataFrame(rows)
    partial = pd.DataFrame(partial_rows)
    auc = pd.DataFrame(auc_rows)
    mon = pd.concat(monthly_parts, ignore_index=True)

    uni.to_csv(args.output_dir / "pre_final_tfi_regression.csv", index=False)
    partial.to_csv(args.output_dir / "pre_final_partial_regression.csv", index=False)
    auc.to_csv(args.output_dir / "pre_final_auc.csv", index=False)
    mon.to_csv(args.output_dir / "pre_final_monthly.csv", index=False)

    c = uni[uni["period"] == "confirmation"].iloc[0]
    r = uni[uni["period"] == "replication"].iloc[0]
    rp = partial[partial["period"] == "replication"].iloc[0]
    ca = auc[auc["period"] == "confirmation"].iloc[0]
    ra = auc[auc["period"] == "replication"].iloc[0]

    neg_fraction = float((mon["beta_bps_per_unit"] < 0).mean()) if len(mon) else 0.0

    checks = {
        "confirmation_tfi_beta_negative": c["beta_bps_per_unit"] < 0,
        "replication_tfi_beta_negative": r["beta_bps_per_unit"] < 0,
        "h1_2025_monthly_negative_fraction_ge_0_67": neg_fraction >= 2/3,
        "confirmation_flow_reversal_auc_ge_0_505": ca["flow_reversal_auc"] >= 0.505,
        "replication_flow_reversal_auc_ge_0_505": ra["flow_reversal_auc"] >= 0.505,
        "replication_partial_tfi_beta_negative": rp["tfi_beta_bps_per_unit"] < 0,
        "replication_partial_tfi_hac_p_le_0_10": rp["tfi_hac_p"] <= 0.10,
        "replication_flow_auc_beats_price_reversal_by_0_001": (
            ra["delta_auc_flow_vs_price_reversal"] >= 0.001
        ),
    }
    gate_pass = all(checks.values())

    gate = {
        "gate_pass": bool(gate_pass),
        "checks": {k: bool(v) for k, v in checks.items()},
        "monthly_negative_fraction": neg_fraction,
        "policy": (
            "v0.9.1 2025 flow-reversal replication design and gate fixed before "
            "any 2025 result. Jul-Dec 2025 opens only if Jan-Mar confirmation "
            "and Apr-Jun replication pass."
        ),
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )

    if gate_pass:
        z = period(x, "final_oos")
        final_uni = pd.DataFrame([{"period": "final_oos", **univariate_tfi(z)}])
        final_partial = pd.DataFrame(
            [{"period": "final_oos", **partial_tfi_control_price(z)}]
        )
        final_auc = pd.DataFrame([auc_comparison(z, "final_oos")])
        final_mon = monthly(z)
        final_uni.to_csv(args.output_dir / "final_oos_tfi_regression.csv", index=False)
        final_partial.to_csv(
            args.output_dir / "final_oos_partial_regression.csv", index=False
        )
        final_auc.to_csv(args.output_dir / "final_oos_auc.csv", index=False)
        final_mon.to_csv(args.output_dir / "final_oos_monthly.csv", index=False)
    else:
        for fn in [
            "final_oos_tfi_regression.csv",
            "final_oos_partial_regression.csv",
            "final_oos_auc.csv",
            "final_oos_monthly.csv",
        ]:
            (args.output_dir / fn).write_text("", encoding="utf-8")

    print("QA")
    print(json.dumps(qa, indent=2))
    print("\nUNIVARIATE TFI")
    print(uni.to_string(index=False))
    print("\nPARTIAL TFI | PRICE CONTROL")
    print(partial.to_string(index=False))
    print("\nFLOW REVERSAL VS PRICE REVERSAL AUC")
    print(auc.to_string(index=False))
    print("\nMONTHLY")
    print(mon.to_string(index=False))
    print("\nGATE")
    print(json.dumps(gate, indent=2))

    if gate_pass:
        print("\nGATE PASSED — JUL-DEC 2025 FINAL OOS OPENED")
        print(pd.read_csv(args.output_dir / "final_oos_auc.csv").to_string(index=False))
    else:
        print("\nGATE FAILED — JUL-DEC 2025 FINAL OOS REMAINS CLOSED")


if __name__ == "__main__":
    main()
