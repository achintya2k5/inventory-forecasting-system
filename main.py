import argparse
import logging
import json
from pathlib import Path

import pandas as pd

from ml_data_loader.loader import (
    run_training_pipeline,
)
from ml_data_loader.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demand Forecasting CLI")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Mode: 'train' or 'predict'"
    )

    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data CSV"
    )

    parser.add_argument(
        "--label",
        type=str,
        default="sales",
        help="Target column name (default: sales)"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Prediction input as JSON string or path to JSON file"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="model.pkl",
        help="Path to saved model file (default: model.pkl)"
    )

    return parser.parse_args()


def load_prediction_input(raw_input: str):
    input_path = Path(raw_input)
    if input_path.exists():
        with input_path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    return json.loads(raw_input)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    args = parse_args()

    if args.mode == "train":
        if not args.data:
            raise ValueError("Training mode requires --data")

        _, metrics = run_training_pipeline(
            file_path=args.data,
            label=args.label,
            model_path=args.model_path,
        )

        logging.info("Training completed.")
        logging.info(f"Final metrics: {metrics}")
    elif args.mode == "predict":
        if not args.input:
            raise ValueError("Prediction mode requires --input")

        model = load_model(args.model_path)
        try:
            data = load_prediction_input(args.input)
        except Exception as e:
            raise ValueError(f"Invalid input format: {e}")

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        preds = model.predict(df)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            print(json.dumps({
                "predictions": preds.tolist(),
                "probabilities": probs.tolist(),
            }))
        else:
            print(preds.tolist())

    else:
        raise ValueError("Invalid mode. Use 'train' or 'predict'.")
