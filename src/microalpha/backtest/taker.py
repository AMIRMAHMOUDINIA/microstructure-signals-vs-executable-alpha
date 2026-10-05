from __future__ import annotations

import numpy as np
import pandas as pd


def one_period_taker_pnl(
    signal,
    bid_entry,
    ask_entry,
    bid_exit,
    ask_exit,
    fee_bps_per_side: float = 0.0,
    slippage_bps_per_side: float = 0.0,
):
    """
    Conservative one-unit round-trip PnL:
      signal +1: buy at ask, sell at bid
      signal -1: sell at bid, buy at ask
      signal  0: no trade

    Fee/slippage are charged on entry and exit notionals.
    """
    s = np.asarray(signal)
    be = np.asarray(bid_entry, dtype=float)
    ae = np.asarray(ask_entry, dtype=float)
    bx = np.asarray(bid_exit, dtype=float)
    ax = np.asarray(ask_exit, dtype=float)

    gross = np.where(
        s > 0,
        bx - ae,
        np.where(s < 0, be - ax, 0.0),
    )

    entry_px = np.where(s > 0, ae, np.where(s < 0, be, 0.0))
    exit_px = np.where(s > 0, bx, np.where(s < 0, ax, 0.0))
    traded = s != 0

    cost_rate = (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
    costs = np.where(traded, cost_rate * (entry_px + exit_px), 0.0)
    net = gross - costs

    return pd.DataFrame({"gross_pnl": gross, "cost": costs, "net_pnl": net})
