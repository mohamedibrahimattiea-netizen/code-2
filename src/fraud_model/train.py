"""Training utilities for the fraud model package."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train(features: pd.DataFrame, target: pd.Series) -> LogisticRegression:
    """Train a simple logistic regression model.

    Parameters
    ----------
    features:
        Feature matrix used for training.
    target:
        Target values corresponding to ``features``.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        The fitted model.
    """

    model = LogisticRegression(max_iter=1000)
    model.fit(features, target)
    return model
