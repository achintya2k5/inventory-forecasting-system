import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import os
import joblib
import json
import logging
from datetime import datetime, timezone

LOGGER = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        LOGGER.info("Converting timestamp column to datetime...")

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception as e:
        LOGGER.exception("Failed to convert timestamp: %s", e)

    null_count = df["timestamp"].isnull().sum()
    if null_count > 0:
        df = df.dropna(subset=["timestamp"])

    LOGGER.info("Timestamp conversion done.")

    if "sales" in df.columns:
        df.loc[:, "sales"] = df["sales"].fillna(0)
        df = df[df["sales"] >= 0]

    if "inventory_level" in df.columns:
        df.loc[:, "inventory_level"] = df["inventory_level"].ffill().bfill()

    if "day_of_week" in df.columns:
        df["day_of_week"] = df["day_of_week"].fillna(df["day_of_week"].mode()[0])

    for col in ["is_holiday", "promotion_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(False)

    if "weather_temp" in df.columns:
        df["weather_temp"] = df["weather_temp"].fillna(df["weather_temp"].mean())

    df = df.drop_duplicates()

    df = df.sort_values(by="timestamp")

    df = df.reset_index(drop=True)
    return df


def run_eda(df: pd.DataFrame) -> None:
    LOGGER.info("Shape: %s", df.shape)
    LOGGER.info("Data Types:\n%s", df.dtypes)
    LOGGER.info("Missing Values:\n%s", df.isnull().sum())
    LOGGER.info("Summary Stats:\n%s", df.describe())
    LOGGER.info("Date Range: %s to %s", df["timestamp"].min(), df["timestamp"].max())
    LOGGER.info("Unique Products: %s", df["item_id"].nunique())
    LOGGER.info("Avg Sales by Day of Week:")
    if "day_of_week" in df.columns:
        LOGGER.info("\n%s", df.groupby("day_of_week")["sales"].mean())

    if "is_holiday" in df.columns:
        LOGGER.info("Avg Sales on Holidays:")
        LOGGER.info("\n%s", df.groupby("is_holiday")["sales"].mean())

    LOGGER.info("Correlation Matrix:")
    LOGGER.info("\n%s", df.corr(numeric_only=True))


def cut_off(cutoff_date, df):
    train = df[df["timestamp"] <= cutoff_date]
    test = df[df["timestamp"] > cutoff_date]
    return train, test


def explain_model(model, x_train, x_test):
    import shap

    # Convert booleans for SHAP compatibility.
    x_train = x_train.copy()
    x_test = x_test.copy()
    for col in x_train.select_dtypes(include="bool").columns:
        x_train[col] = x_train[col].astype(int)
        x_test[col] = x_test[col].astype(int)

    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_test)

    shap.plots.beeswarm(shap_values)
    LOGGER.info("Generated SHAP beeswarm plot.")


def recommend_inventory_actions(
    df, demand_col="predicted_sales", inventory_col="inventory_level", safety_stock=10
):
    df = df.copy()

    # Calculate difference between predicted demand and inventory
    df["reorder_flag"] = df[demand_col] + safety_stock > df[inventory_col]

    # Suggested order quantity
    df["suggested_order_qty"] = (
        (df[demand_col] + safety_stock - df[inventory_col]).clip(lower=0).round()
    )

    return df
