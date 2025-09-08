from fraud_model.train import train
import pandas as pd

def test_train_runs():
    df = pd.DataFrame({"x": [0, 1], "y": [1, 0]})
    model = train(df[["x"]], df["y"])
    assert hasattr(model, "predict")
