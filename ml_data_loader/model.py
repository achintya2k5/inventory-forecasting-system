import logging
import joblib
import json
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from ml_data_loader.preprocess import build_preprocessor

LOGGER = logging.getLogger(__name__)


def _is_classification_target(y) -> bool:
    return int(pd.Series(y).nunique(dropna=True)) < 20


def train_model(X, y):
    preprocessor = build_preprocessor(X)

    if _is_classification_target(y):
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
        model_type = "classification"
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
        model_type = "regression"

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)
    return pipeline, model_type


def evaluate_model(model, x_test, y_test):
    preds = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        return {"accuracy": float(accuracy_score(y_test, preds))}
    return {"mae": float(mean_absolute_error(y_test, preds))}


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_metadata(
    model_type,
    X,
    metrics,
    model_path,
    label_column,
    metadata_path="model_metadata.json",
):
    if isinstance(X, pd.DataFrame):
        rows = len(X)
        columns = list(X.columns)
    else:
        rows = len(X)
        columns = []

    metadata = {
        "model_type": model_type,
        "training_time": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "columns": columns,
        "metrics": metrics,
        "model_path": model_path,
        "label_column": label_column,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
