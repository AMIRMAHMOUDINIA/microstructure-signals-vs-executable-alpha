from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    valid_idx: np.ndarray


def expanding_walk_forward(
    n_samples: int,
    min_train: int,
    valid_size: int,
    step: int | None = None,
    gap: int = 0,
):
    """
    Expanding-window chronological splits.

    train = [0, train_end)
    gap   = [train_end, train_end+gap)
    valid = [train_end+gap, train_end+gap+valid_size)
    """
    if step is None:
        step = valid_size
    if min_train <= 0 or valid_size <= 0:
        raise ValueError("min_train and valid_size must be positive")

    train_end = min_train
    while train_end + gap + valid_size <= n_samples:
        tr = np.arange(0, train_end)
        va = np.arange(train_end + gap, train_end + gap + valid_size)
        yield Split(tr, va)
        train_end += step
