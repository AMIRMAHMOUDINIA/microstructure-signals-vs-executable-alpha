import numpy as np
import pandas as pd

from microalpha.validation.inference import (
    moving_block_bootstrap_beta,
    non_overlapping_mask,
    ols_naive_and_hac,
)


def test_non_overlapping_mask():
    m = non_overlapping_mask(20, 5)
    assert m.sum() == 4
    assert np.flatnonzero(m).tolist() == [0, 5, 10, 15]


def test_hac_regression_positive_slope():
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(size=500))
    y = 0.2 * x + pd.Series(rng.normal(scale=0.2, size=500))
    r = ols_naive_and_hac(x, y, hac_lags=5)
    assert r.beta > 0.1
    assert r.p_hac < 0.01


def test_block_bootstrap_ci_contains_positive_beta():
    rng = np.random.default_rng(9)
    x = pd.Series(rng.normal(size=500))
    y = 0.15 * x + pd.Series(rng.normal(scale=0.25, size=500))
    b = moving_block_bootstrap_beta(x, y, block_length=20, reps=100, seed=1)
    assert b["beta_median"] > 0
    assert b["beta_ci_high_95"] > b["beta_ci_low_95"]
