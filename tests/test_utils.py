import pandas as pd

from ml_data_loader.utils import clean_data, recommend_inventory_actions


def test_clean_data_handles_timestamp_sales_and_sorting():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-05-03", "invalid-date", "2024-05-01"],
            "sales": [10, -2, None],
            "inventory_level": [None, 15, 20],
            "day_of_week": [None, 2, 0],
            "is_holiday": [None, True, None],
            "promotion_flag": [None, False, None],
            "weather_temp": [25.0, None, 20.0],
        }
    )

    cleaned = clean_data(df)

    assert cleaned["timestamp"].isna().sum() == 0
    assert (cleaned["sales"] >= 0).all()
    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned["inventory_level"].isna().sum() == 0


def test_recommend_inventory_actions_flags_and_quantities():
    df = pd.DataFrame(
        {
            "item_id": ["SKU1", "SKU2"],
            "predicted_sales": [50, 10],
            "inventory_level": [40, 30],
        }
    )

    output = recommend_inventory_actions(df, safety_stock=10)

    assert list(output["reorder_flag"]) == [True, False]
    assert list(output["suggested_order_qty"]) == [20.0, 0.0]
