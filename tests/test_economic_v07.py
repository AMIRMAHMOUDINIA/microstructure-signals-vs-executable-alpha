import numpy as np
import pandas as pd

from microalpha.backtest.economic_v07 import (
    EconomicSpec,
    add_reconstructed_bbo,
    confidence_cutoffs,
    simulate_nonoverlapping,
    trade_metrics,
)


def make_grid(n=40):
    ts = pd.date_range("2023-11-01", periods=n, freq="1min", tz="UTC")
    mid = 100.0 + np.arange(n) * 0.01
    df = pd.DataFrame({
        "timestamp": ts,
        "observed": True,
        "mid": mid,
        "bt_spread_bps_close": 2.0,
        "pred_prob": 0.60,
    })
    return add_reconstructed_bbo(df)


def test_bbo_reconstruction():
    df = pd.DataFrame({
        "mid": [100.0],
        "bt_spread_bps_close": [10.0],
    })
    out = add_reconstructed_bbo(df)
    assert np.isclose(out.loc[0, "ask_close"] - out.loc[0, "bid_close"], 0.1)
    assert np.isclose((out.loc[0, "ask_close"] + out.loc[0, "bid_close"]) / 2, 100.0)


def test_confidence_cutoffs_monotonic():
    p = pd.Series(np.linspace(0.45, 0.55, 1001))
    c = confidence_cutoffs(p, [1.0, 0.5, 0.1])
    assert c.loc[0, "confidence_cutoff"] <= c.loc[1, "confidence_cutoff"]
    assert c.loc[1, "confidence_cutoff"] <= c.loc[2, "confidence_cutoff"]


def test_nonoverlap_and_execution_delay():
    df = make_grid(40)
    t = simulate_nonoverlapping(
        df,
        confidence_cutoff=0.01,
        extra_roundtrip_cost_bps=0,
        period_start="2023-11-01T00:00:00Z",
        period_end_exclusive="2023-11-01T00:40:00Z",
        spec=EconomicSpec(forecast_horizon_minutes=10, execution_delay_minutes=1),
    )
    assert len(t) >= 3
    first = t.iloc[0]
    assert first["entry_timestamp"] == first["decision_timestamp"] + pd.Timedelta(minutes=1)
    assert first["exit_timestamp"] == first["decision_timestamp"] + pd.Timedelta(minutes=10)
    if len(t) > 1:
        assert t.iloc[1]["decision_timestamp"] >= first["exit_timestamp"]


def test_extra_cost_subtracts_exactly():
    df = make_grid(40)
    a = simulate_nonoverlapping(
        df, 0.01, 0,
        "2023-11-01T00:00:00Z", "2023-11-01T00:40:00Z"
    )
    b = simulate_nonoverlapping(
        df, 0.01, 4,
        "2023-11-01T00:00:00Z", "2023-11-01T00:40:00Z"
    )
    assert len(a) == len(b)
    assert np.allclose(a["net_bps"] - b["net_bps"], 4.0)


def test_trade_metrics_break_even_is_gross_mean():
    df = make_grid(40)
    t = simulate_nonoverlapping(
        df, 0.01, 2,
        "2023-11-01T00:00:00Z", "2023-11-01T00:40:00Z"
    )
    m = trade_metrics(t)
    assert np.isclose(m["break_even_extra_rt_cost_bps"], m["gross_mean_bps"])
