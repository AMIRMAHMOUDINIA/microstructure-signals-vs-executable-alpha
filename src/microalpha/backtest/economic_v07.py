from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class EconomicSpec:
    forecast_horizon_minutes: int = 10
    execution_delay_minutes: int = 1
    # Exit remains at the model's pre-specified target endpoint t+h.
    # With 1-minute delayed entry, primary realized holding time is 9 minutes.
    min_trades_selection: int = 100
    min_trades_confirmation: int = 100


DEFAULT_COVERAGES = (1.00, 0.75, 0.50, 0.25, 0.10, 0.05)
DEFAULT_EXTRA_RT_COST_BPS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def add_reconstructed_bbo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the close best bid/ask from mid and close spread in basis points.

    spread_bps = (ask - bid) / mid * 10,000
    bid = mid - spread/2
    ask = mid + spread/2
    """
    x = df.copy()
    mid = pd.to_numeric(x["mid"], errors="coerce")
    spread_bps = pd.to_numeric(x["bt_spread_bps_close"], errors="coerce")
    spread_abs = mid * spread_bps / 10_000.0
    x["bid_close"] = mid - spread_abs / 2.0
    x["ask_close"] = mid + spread_abs / 2.0
    return x


def confidence_cutoffs(
    probabilities: pd.Series,
    coverages=DEFAULT_COVERAGES,
) -> pd.DataFrame:
    """
    Pre-specified threshold family defined by validation-selection coverage.

    For coverage c, choose the confidence cutoff such that approximately the
    strongest c fraction of |p-0.5| signals is eligible.
    """
    p = pd.to_numeric(probabilities, errors="coerce").dropna()
    conf = (p - 0.5).abs()
    rows = []
    for coverage in coverages:
        if not (0 < coverage <= 1):
            raise ValueError("coverage must be in (0, 1]")
        q = max(0.0, 1.0 - float(coverage))
        cutoff = float(conf.quantile(q))
        rows.append({
            "coverage_target": float(coverage),
            "confidence_cutoff": cutoff,
            "long_probability_cutoff": 0.5 + cutoff,
            "short_probability_cutoff": 0.5 - cutoff,
        })
    return pd.DataFrame(rows)


def _safe_log_ratio(num, den) -> float:
    if not (np.isfinite(num) and np.isfinite(den) and num > 0 and den > 0):
        return np.nan
    return float(np.log(num / den))


def simulate_nonoverlapping(
    df: pd.DataFrame,
    confidence_cutoff: float,
    extra_roundtrip_cost_bps: float,
    period_start: str,
    period_end_exclusive: str,
    spec: EconomicSpec | None = None,
) -> pd.DataFrame:
    """
    Conservative execution rule.

    At minute t:
      1) Features for minute t are already complete.
      2) If flat and |p_t - 0.5| >= cutoff, choose direction.
      3) Enter at the CLOSE BBO of t+1 (one-minute execution delay).
      4) Exit at the CLOSE BBO of the original model endpoint t+h.
      5) Cross the spread on both entry and exit.
      6) Subtract an additional pre-specified round-trip cost (fees+slippage).
      7) Do not overlap positions.

    The primary economic test therefore uses the same t→t+10 forecast horizon
    while imposing a conservative one-minute delay before entry.
    """
    if spec is None:
        spec = EconomicSpec()

    if extra_roundtrip_cost_bps < 0:
        raise ValueError("extra_roundtrip_cost_bps must be nonnegative")

    x = df.sort_values("timestamp").reset_index(drop=True).copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)

    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end_exclusive)
    ts = x["timestamp"]

    # Lookup by exact minute timestamp.
    lookup = {t: i for i, t in enumerate(ts)}
    next_allowed = start
    trades = []

    for i, row in x.iterrows():
        t = row["timestamp"]
        if t < start or t >= end:
            continue
        if t < next_allowed:
            continue

        p = row.get("pred_prob", np.nan)
        if not np.isfinite(p):
            continue
        confidence = abs(float(p) - 0.5)
        if confidence < confidence_cutoff:
            continue

        side = 1 if p >= 0.5 else -1
        entry_t = t + pd.Timedelta(minutes=spec.execution_delay_minutes)
        exit_t = t + pd.Timedelta(minutes=spec.forecast_horizon_minutes)

        # Keep the whole realized trade inside the evaluated period.
        if entry_t >= end or exit_t >= end:
            continue
        if entry_t not in lookup or exit_t not in lookup:
            continue

        entry = x.iloc[lookup[entry_t]]
        exit_ = x.iloc[lookup[exit_t]]

        # Require actual observations, not missing-grid placeholders.
        if not bool(entry.get("observed", False)) or not bool(exit_.get("observed", False)):
            continue

        be = float(entry["bid_close"])
        ae = float(entry["ask_close"])
        bx = float(exit_["bid_close"])
        ax = float(exit_["ask_close"])

        if side > 0:
            gross_logret = _safe_log_ratio(bx, ae)
            entry_px, exit_px = ae, bx
        else:
            gross_logret = _safe_log_ratio(be, ax)
            entry_px, exit_px = be, ax

        if not np.isfinite(gross_logret):
            continue

        gross_bps = gross_logret * 10_000.0
        net_bps = gross_bps - float(extra_roundtrip_cost_bps)

        trades.append({
            "decision_timestamp": t,
            "entry_timestamp": entry_t,
            "exit_timestamp": exit_t,
            "side": int(side),
            "pred_prob": float(p),
            "confidence": confidence,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "gross_bps_after_spread": gross_bps,
            "extra_roundtrip_cost_bps": float(extra_roundtrip_cost_bps),
            "net_bps": net_bps,
        })

        # Next decision can occur at the exit minute; any new position enters
        # at the following minute and therefore cannot overlap.
        next_allowed = exit_t

    return pd.DataFrame(trades)


def _hac_mean_test(values: pd.Series, maxlags: int = 5) -> tuple[float, float]:
    y = pd.to_numeric(values, errors="coerce").dropna()
    if len(y) < 20:
        return np.nan, np.nan
    X = np.ones((len(y), 1))
    fit = sm.OLS(y.to_numpy(float), X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(maxlags)},
    )
    return float(fit.tvalues[0]), float(fit.pvalues[0])


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0,
            "long_fraction": np.nan,
            "gross_mean_bps": np.nan,
            "gross_total_bps": 0.0,
            "net_mean_bps": np.nan,
            "net_median_bps": np.nan,
            "net_std_bps": np.nan,
            "net_total_bps": 0.0,
            "hit_rate": np.nan,
            "max_drawdown_bps_fixed_notional": np.nan,
            "break_even_extra_rt_cost_bps": np.nan,
            "hac_t_net_mean": np.nan,
            "hac_p_net_mean": np.nan,
        }

    gross = trades["gross_bps_after_spread"].astype(float)
    net = trades["net_bps"].astype(float)
    cumulative = net.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    hac_t, hac_p = _hac_mean_test(net, maxlags=5)

    return {
        "n_trades": len(trades),
        "long_fraction": float((trades["side"] > 0).mean()),
        "gross_mean_bps": float(gross.mean()),
        "gross_total_bps": float(gross.sum()),
        "net_mean_bps": float(net.mean()),
        "net_median_bps": float(net.median()),
        "net_std_bps": float(net.std(ddof=1)),
        "net_total_bps": float(net.sum()),
        "hit_rate": float((net > 0).mean()),
        "max_drawdown_bps_fixed_notional": float(drawdown.min()),
        # Spread is already included in gross execution.
        "break_even_extra_rt_cost_bps": float(gross.mean()),
        "hac_t_net_mean": hac_t,
        "hac_p_net_mean": hac_p,
    }


def weekly_positive_fraction(trades: pd.DataFrame) -> float:
    if trades is None or len(trades) == 0:
        return np.nan
    x = trades.copy()
    x["week"] = pd.to_datetime(x["decision_timestamp"], utc=True).dt.to_period("W").astype(str)
    weekly = x.groupby("week")["net_bps"].sum()
    return float((weekly > 0).mean()) if len(weekly) else np.nan


def monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    x = trades.copy()
    x["month"] = pd.to_datetime(x["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
    rows = []
    for month, g in x.groupby("month", sort=True):
        row = {"month": month}
        row.update(trade_metrics(g))
        rows.append(row)
    return pd.DataFrame(rows)
