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

    df = pd.DataFrame(request.data)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing_cols}"
        )

    df = df[EXPECTED_COLUMNS]
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
