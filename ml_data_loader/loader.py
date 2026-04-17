import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from ml_data_loader.utils import clean_data, recommend_inventory_actions
from ml_data_loader.model import (
    evaluate_model,
    save_model,
    save_metadata,
    train_model,
)

LOGGER = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "lag_1",
    "rolling_mean_3",
    "day_of_week",
    "is_holiday",
    "promotion_flag",
    "weather_temp",
]


def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess_dataframe(df: pd.DataFrame, label: str = "sales") -> pd.DataFrame:
    df = clean_data(df)

    required_for_engineering = {"item_id", label}
    missing = required_for_engineering - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for preprocessing: {sorted(missing)}"
        )

    # Generate lag-based demand features per item.
    df["lag_1"] = df.groupby("item_id")[label].shift(1)

    df["rolling_mean_3"] = df.groupby("item_id")[label].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    df = df.dropna(subset=["lag_1", "rolling_mean_3", label])

    if "is_holiday" in df.columns:
        df["is_holiday"] = df["is_holiday"].astype(int)

    if "promotion_flag" in df.columns:
        df["promotion_flag"] = df["promotion_flag"].astype(int)

    return df


def split_features_label(df: pd.DataFrame, label: str):
    missing_feature_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing required feature columns: {missing_feature_cols}")
    if label not in df.columns:
        raise ValueError(f"Label column not found: {label}")

    X = df[FEATURE_COLUMNS]
    y = df[label]
    return X, y


def run_training_pipeline(
    file_path: str,
    label: str = "sales",
    model_path: str = "model.pkl",
    inventory_output_path: str = "inventory_recommendations.csv",
):
    df = load_csv(file_path)
    df = preprocess_dataframe(df, label=label)

    X, y = split_features_label(df, label)

    x_train, x_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model, model_type = train_model(x_train, y_train)

    metrics = evaluate_model(model, x_eval, y_eval)
    LOGGER.info("Evaluation metrics: %s", metrics)

    save_model(model, model_path)

    save_metadata(
        model_type=model_type,
        X=x_train,
        metrics=metrics,
        model_path=model_path,
        label_column=label,
    )

    if "inventory_level" in df.columns and model_type == "regression":
        preds = model.predict(X)
        preds = np.asarray(preds).ravel()
        df["predicted_sales"] = preds
        optimized_inventory = recommend_inventory_actions(df)
        optimized_inventory.to_csv(inventory_output_path, index=False)
        LOGGER.info("Inventory recommendations saved to %s", inventory_output_path)
    elif "inventory_level" in df.columns:
        LOGGER.info(
            "Skipping inventory recommendations for classification model type."
        )

    return model, metrics
