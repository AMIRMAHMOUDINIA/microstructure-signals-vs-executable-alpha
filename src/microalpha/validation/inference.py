from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class RegressionResult:
    n: int
    beta: float
    intercept: float
    t_naive: float
    p_naive: float
    t_hac: float
    p_hac: float
    r2: float


def ols_naive_and_hac(
    x: pd.Series,
    y: pd.Series,
    hac_lags: int,
) -> RegressionResult:
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 20:
        raise ValueError("At least 20 valid observations are required.")

    X = sm.add_constant(z["x"])
    naive = sm.OLS(z["y"], X).fit()
    hac = sm.OLS(z["y"], X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(max(1, hac_lags))},
    )

    return RegressionResult(
        n=len(z),
        beta=float(naive.params["x"]),
        intercept=float(naive.params["const"]),
        t_naive=float(naive.tvalues["x"]),
        p_naive=float(naive.pvalues["x"]),
        t_hac=float(hac.tvalues["x"]),
        p_hac=float(hac.pvalues["x"]),
        r2=float(naive.rsquared),
    )


def non_overlapping_mask(n: int, horizon_rows: int, offset: int = 0) -> np.ndarray:
    if horizon_rows <= 0:
        raise ValueError("horizon_rows must be positive")
    mask = np.zeros(n, dtype=bool)
    mask[offset::horizon_rows] = True
    return mask


def moving_block_bootstrap_beta(
    x: pd.Series,
    y: pd.Series,
    block_length: int = 60,
    reps: int = 1000,
    seed: int = 42,
) -> dict:
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < max(50, block_length * 2):
        raise ValueError("Not enough observations for the requested block length.")

    xa = z["x"].to_numpy(float)
    ya = z["y"].to_numpy(float)
    n = len(z)
    rng = np.random.default_rng(seed)

    starts = np.arange(0, n - block_length + 1)
    betas = np.empty(reps, dtype=float)

    for r in range(reps):
        idx_parts = []
        while sum(len(p) for p in idx_parts) < n:
            s = int(rng.choice(starts))
            idx_parts.append(np.arange(s, s + block_length))
        idx = np.concatenate(idx_parts)[:n]

        xb = xa[idx]
        yb = ya[idx]
        vx = np.var(xb)
        if vx == 0:
            betas[r] = np.nan
        else:
            betas[r] = np.cov(xb, yb, ddof=0)[0, 1] / vx

    betas = betas[np.isfinite(betas)]
    return {
        "bootstrap_reps": len(betas),
        "block_length": int(block_length),
        "beta_median": float(np.median(betas)),
        "beta_ci_low_95": float(np.quantile(betas, 0.025)),
        "beta_ci_high_95": float(np.quantile(betas, 0.975)),
        "p_sign_two_sided": float(
            min(1.0, 2 * min(np.mean(betas <= 0), np.mean(betas >= 0)))
        ),
    }


def moving_block_bootstrap_beta_fast(
    x: pd.Series,
    y: pd.Series,
    block_length: int = 240,
    reps: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Fast moving-block bootstrap for a univariate OLS slope.

    It samples overlapping fixed-length blocks using precomputed block sums,
    avoiding construction of an O(n) resampled vector on every replication.
    The bootstrap sample length is ceil(n / L) * L.
    """
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    xa = z["x"].to_numpy(float)
    ya = z["y"].to_numpy(float)
    n = len(z)
    L = int(block_length)
    if L <= 1:
        raise ValueError("block_length must exceed 1")
    if n < 2 * L:
        raise ValueError("Need at least two block lengths of observations.")

    def window_sums(a):
        c = np.concatenate([[0.0], np.cumsum(a)])
        return c[L:] - c[:-L]

    sx = window_sums(xa)
    sy = window_sums(ya)
    sxx = window_sums(xa * xa)
    sxy = window_sums(xa * ya)

    n_blocks = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)

    # Chunk reps to keep memory bounded for large datasets.
    betas = []
    done = 0
    while done < reps:
        batch = min(200, reps - done)
        idx = rng.integers(0, len(sx), size=(batch, n_blocks))
        Sx = sx[idx].sum(axis=1)
        Sy = sy[idx].sum(axis=1)
        Sxx = sxx[idx].sum(axis=1)
        Sxy = sxy[idx].sum(axis=1)
        N = float(n_blocks * L)

        denom = Sxx - (Sx * Sx / N)
        numer = Sxy - (Sx * Sy / N)
        b = np.where(np.abs(denom) > 1e-18, numer / denom, np.nan)
        betas.append(b)
        done += batch

    beta = np.concatenate(betas)
    beta = beta[np.isfinite(beta)]
    if len(beta) == 0:
        raise ValueError("All bootstrap slopes were non-finite.")

    return {
        "bootstrap_reps": len(beta),
        "block_length": L,
        "bootstrap_sample_rows": int(n_blocks * L),
        "beta_median": float(np.median(beta)),
        "beta_ci_low_95": float(np.quantile(beta, 0.025)),
        "beta_ci_high_95": float(np.quantile(beta, 0.975)),
        "p_sign_two_sided": float(
            min(1.0, 2.0 * min(np.mean(beta <= 0), np.mean(beta >= 0)))
        ),
    }
