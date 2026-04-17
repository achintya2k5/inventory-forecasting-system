import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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
        df.loc[:, "sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
        df = df[df["sales"] >= 0]

    if "inventory_level" in df.columns:
        df.loc[:, "inventory_level"] = pd.to_numeric(
            df["inventory_level"],
            errors="coerce",
        ).ffill().bfill()

    if "day_of_week" in df.columns:
        mode_values = df["day_of_week"].mode()
        mode_day = mode_values.iloc[0] if not mode_values.empty else 0
        df["day_of_week"] = df["day_of_week"].fillna(mode_day)

    for col in ["is_holiday", "promotion_flag"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean").fillna(False).astype(bool)

    if "weather_temp" in df.columns:
        weather_temp = pd.to_numeric(df["weather_temp"], errors="coerce")
        df["weather_temp"] = weather_temp.fillna(weather_temp.mean())

    df = df.drop_duplicates()

    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp")

    df = df.reset_index(drop=True)
    return df


def recommend_inventory_actions(
    df,
    demand_col="predicted_sales",
    inventory_col="inventory_level",
    safety_stock=10,
):
    df = df.copy()

    missing_cols = [col for col in [demand_col, inventory_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for inventory recommendation: {missing_cols}")

    df["reorder_flag"] = df[demand_col] + safety_stock > df[inventory_col]

    df["suggested_order_qty"] = (
        (df[demand_col] + safety_stock - df[inventory_col]).clip(lower=0).round()
    )

    return df
