import logging
import joblib
import json
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error
from ml_data_loader.preprocess import build_preprocessor
from ml_data_loader.utils import explain_model

LOGGER = logging.getLogger(__name__)


def train_model(X, y):
    preprocessor = build_preprocessor(X)

    if y.nunique() < 20:
        model=RandomForestClassifier()
        model_type="classification"
    else:
        model=RandomForestRegressor()
        model_type="regression"

    pipeline=Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)
    return pipeline, model_type


def evaluate_model(model, x_test, y_test):
    preds=model.predict(x_test)
    
    if y_test.nunique()<20:
        return {"accuracy": accuracy_score(y_test, preds)}
    else:
        return {"mae": mean_absolute_error(y_test, preds)}

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
    metadata = {
        "model_type": model_type,
        "training_time": datetime.now(timezone.utc).isoformat(),
        "rows": len(X),
        "columns": list(X.columns),
        "metrics": metrics,
        "model_path": model_path,
        "label_column": label_column
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

