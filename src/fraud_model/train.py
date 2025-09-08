"""Training logic extracted from the notebook."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from .utils import preprocess_features_safely


def train(df: pd.DataFrame):
    """Train a baseline model on *df* and return the fitted model and metrics."""
    df_proc = preprocess_features_safely(df)
    X_all, y_all = df_proc.drop("Class", axis=1), df_proc["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train_b, y_train_b = smote.fit_resample(X_train, y_train)
    model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_b, y_train_b)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba > 0.5).astype(int)
    metrics = {
        "PR_AUC": average_precision_score(y_test, proba),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "F1": f1_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    }
    return model, metrics


__all__ = ["train"]
