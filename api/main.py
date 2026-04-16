from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd

app=FastAPI()

model=joblib.load("trained_model.pkl")

EXPECTED_COLUMNS = [
    "lag_1",
    "rolling_mean_3",
    "day_of_week",
    "is_holiday",
    "promotion_flag",
    "weather_temp"
]

class PredictionRequest(BaseModel):
    data: List[Dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    df=pd.DataFrame(request.data)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing_cols}"
        )
    preds=model.predict(df)
    return {"predictions": preds.tolist()}

