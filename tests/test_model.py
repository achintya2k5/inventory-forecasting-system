import pandas as pd

from ml_data_loader.model import train_model, evaluate_model

def test_model_training_and_prediction():
    df=pd.DataFrame({
        "num": [1,2,3,4],
        "cat": ["a", "b", "a", "b"],
        "target": [10, 20, 15, 25]
    })

    x=df.drop(columns=["target"])
    y=df["target"]

    model, model_type=train_model(x, y)

    assert model is not None
    assert model_type in ["regressor", "classification"]

    preds= model.predict(x)

    assert len(preds) == len(y)

def test_model_evaluation():
    df=pd.DataFrame({
        "num": [1,2,3,4],
        "cat": ["a", "b", "a", "b"],
        "target": [10, 20, 15, 25]
    })

    x=df.drop(columns=["target"])
    y=df["target"]

    model, _=train_model(x, y)

    metrics=evaluate_model(model, x, y)

    assert isinstance(metrics, dict)
    assert "mae" in metrics or "accuracy" in metrics