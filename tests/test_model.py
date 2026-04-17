import pandas as pd

from ml_data_loader.model import train_model, evaluate_model


def test_model_training_and_prediction():
    size = 30
    df = pd.DataFrame(
        {
            "num": list(range(1, size + 1)),
            "cat": ["a" if i % 2 == 0 else "b" for i in range(size)],
            "target": [100 + i for i in range(size)],
        }
    )

    x = df.drop(columns=["target"])
    y = df["target"]

    model, model_type = train_model(x, y)

    assert model is not None
    assert model_type == "regression"

    preds = model.predict(x)

    assert len(preds) == len(y)


def test_model_evaluation():
    size = 30
    df = pd.DataFrame(
        {
            "num": list(range(1, size + 1)),
            "cat": ["a" if i % 2 == 0 else "b" for i in range(size)],
            "target": [100 + i for i in range(size)],
        }
    )

    x = df.drop(columns=["target"])
    y = df["target"]

    model, _ = train_model(x, y)

    metrics = evaluate_model(model, x, y)

    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert metrics["mae"] >= 0


def test_classification_model_training_prediction_and_probabilities():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    x = df.drop(columns=["target"])
    y = df["target"]

    model, model_type = train_model(x, y)

    assert model_type == "classification"

    preds = model.predict(x)
    probs = model.predict_proba(x)

    assert len(preds) == len(y)
    assert len(probs) == len(y)

    metrics = evaluate_model(model, x, y)
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1
