from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    estimator: object


def classification_models():
    return [
        ModelSpec(
            "logistic_l2",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2000)),
            ]),
        )
    ]


def regression_models():
    return [
        ModelSpec(
            "ridge",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
        )
    ]


def majority_class_prediction(y_train, n):
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[np.argmax(counts)]
    return np.repeat(majority, n)
