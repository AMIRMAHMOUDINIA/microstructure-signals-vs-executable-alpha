import numpy as np
from microalpha.backtest.taker import one_period_taker_pnl


def test_long_crosses_spread():
    out = one_period_taker_pnl(
        signal=[1],
        bid_entry=[99],
        ask_entry=[101],
        bid_exit=[104],
        ask_exit=[106],
    )
    assert np.isclose(out.loc[0, "gross_pnl"], 3.0)


def test_short_crosses_spread():
    out = one_period_taker_pnl(
        signal=[-1],
        bid_entry=[99],
        ask_entry=[101],
        bid_exit=[94],
        ask_exit=[96],
    )
    assert np.isclose(out.loc[0, "gross_pnl"], 3.0)


def test_cost_reduces_pnl():
    a = one_period_taker_pnl([1], [99], [101], [104], [106], 0, 0)
    b = one_period_taker_pnl([1], [99], [101], [104], [106], 2, 1)
    assert b.loc[0, "net_pnl"] < a.loc[0, "net_pnl"]
