"""Utility functions for fraud model training."""

from __future__ import annotations

import numpy as np
import pandas as pd

def preprocess_features_safely(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df_in* with categorical and numeric data cleaned.

    - Categorical columns are converted to integer codes.
    - Numeric columns are coerced to numbers and NaNs replaced with the median.
    """
    df_p = df_in.copy()
    for col in df_p.select_dtypes(include=["object", "category"]).columns:
        df_p[col] = df_p[col].astype("category").cat.codes
    for col in df_p.select_dtypes(include=[np.number]).columns:
        df_p[col] = pd.to_numeric(df_p[col], errors="coerce")
        df_p[col] = df_p[col].fillna(df_p[col].median())
    return df_p
