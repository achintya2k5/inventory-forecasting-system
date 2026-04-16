import pandas as pd
import logging
import numpy as np
from ml_data_loader.utils import clean_data, recommend_inventory_actions
from ml_data_loader.model import evaluate_model, save_model, save_metadata, train_model

LOGGER = logging.getLogger(__name__)


def data_loader(cutoff_date_str=None, retrain=False, explain=False):
    df = load_csv("training_data.csv")
    df = clean_data(df)

    df["lag_1"] = df.groupby("item_id")["sales"].shift(1)
    df["rolling_mean_3"] = df.groupby("item_id")["sales"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    df = df.dropna()

    X, y = split_features_label(df, label="sales")

    model, model_type = train_model(X, y)

    metrics = evaluate_model(model, X, y)
    LOGGER.info("Training metrics: %s", metrics)

    save_model(model, "trained_model.pkl")

    save_metadata(
        model_type=model_type,
        X=X,
        metrics=metrics,
        model_path="model.pkl",
        label_column="sales",
    )

    preds = model.predict(X)
    preds = np.asarray(preds).ravel()

    df["predicted_sales"] = preds

    optimized_inventory = recommend_inventory_actions(df)

    optimized_inventory.to_csv("inventory_recommendations.csv", index=False)

    LOGGER.info("Inventory recommendations saved to inventory_recommendations.csv")
    LOGGER.info(
        "\n%s",
        optimized_inventory[
            ["item_id", "inventory_level", "predicted_sales", "reorder_flag", "suggested_order_qty"]
        ].head(20),
    )

    return metrics

def load_csv(file_path):
    return pd.read_csv(file_path)

def split_features_label(df, label):
    x=df.drop(columns=[label])
    y=df[label]
    return x,y