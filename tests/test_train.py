import pandas as pd
from sklearn.datasets import make_classification

from fraud_model.train import train


def test_train_runs():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["Class"] = y
    model, metrics = train(df)
    assert hasattr(model, "predict")
    assert set(metrics.keys()) == {"PR_AUC", "ROC_AUC", "F1", "Precision", "Recall", "MCC"}
