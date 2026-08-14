import pandas as pd
import numpy as np

from microalpha.validation.inference import moving_block_bootstrap_beta_fast


def test_fast_bootstrap_detects_positive_slope():
    rng = np.random.default_rng(123)
    x = pd.Series(rng.normal(size=4000))
    y = 0.2 * x + pd.Series(rng.normal(scale=0.5, size=4000))
    out = moving_block_bootstrap_beta_fast(
        x, y, block_length=50, reps=100, seed=1
    )
    assert out["beta_median"] > 0.1
    assert out["beta_ci_low_95"] > 0
