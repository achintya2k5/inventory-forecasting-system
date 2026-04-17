import os
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

app = FastAPI(title="Demand Forecasting API", version="1.0.0")

model = None

EXPECTED_COLUMNS = [
    "lag_1",
    "rolling_mean_3",
    "day_of_week",
    "is_holiday",
    "promotion_flag",
    "weather_temp"
]

BINARY_COLUMNS = {"is_holiday", "promotion_flag"}


def load_model():
    global model
    model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event():
    global model
    try:
        load_model()
    except Exception:
        model = None


class PredictionRequest(BaseModel):
    data: List[Dict]


def _is_classifier_model() -> bool:
    return model is not None and hasattr(model, "predict_proba")


def _validate_payload_rows(rows: List[Dict]) -> None:
    if not rows:
        raise HTTPException(status_code=400, detail="Input data must not be empty")

    expected = set(EXPECTED_COLUMNS)

    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Row {row_idx} must be an object with feature values",
            )

        row_columns = set(row.keys())
        missing_cols = sorted(expected - row_columns)
        extra_cols = sorted(row_columns - expected)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Row {row_idx} missing columns: {missing_cols}",
            )
        if extra_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Row {row_idx} has unexpected columns: {extra_cols}",
            )

        for col in EXPECTED_COLUMNS:
            value = row[col]

            if col in BINARY_COLUMNS:
                if isinstance(value, bool):
                    numeric_value = int(value)
                elif isinstance(value, (int, float)):
                    numeric_value = value
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Row {row_idx} column '{col}' must be numeric with value 0 or 1"
                        ),
                    )

                if numeric_value not in (0, 1):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Row {row_idx} column '{col}' must be 0 or 1",
                    )
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail=f"Row {row_idx} column '{col}' must be int or float",
                )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Train first or mount {MODEL_PATH}",
        )

    _validate_payload_rows(request.data)

    df = pd.DataFrame(request.data)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
    if extra_cols:
        raise HTTPException(status_code=400, detail=f"Unexpected columns: {extra_cols}")

    df = df[EXPECTED_COLUMNS]
    for binary_col in BINARY_COLUMNS:
        df[binary_col] = df[binary_col].astype(int)

    preds = model.predict(df)

    if _is_classifier_model():
        probs = model.predict_proba(df)
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
        }

    return {"predictions": preds.tolist()}
