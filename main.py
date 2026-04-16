import argparse
import logging
import json
import pandas as pd

from ml_data_loader.loader import load_csv, split_features_label
from ml_data_loader.model import evaluate_model, train_model, save_model, load_model, save_metadata
from ml_data_loader.utils import clean_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run demand forecasting and inventory recommendations."
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Cutoff date in YYYY-MM-DD format for train/test split.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the model even if a saved model exists.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate SHAP beeswarm explanation plot after training.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Mode: 'train' to train the model, 'predict' to make predictions.",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Target column name"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="JSON string input for prediction"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.mode == "train":
        if not args.data or not args.label:
            raise ValueError("Training mode requires --data and --label arguments.")
        
        df=load_csv(args.data)
        df=clean_data(df)

        X, y = split_features_label(df, args.label)

        model, model_type=train_model(X, y)

        metrics=evaluate_model(model, X, y)
        logging.info(f"Training metrics: {metrics}")

        save_model(model, "model.pkl")
        save_metadata(
            model_type=model_type,
            X=X,
            metrics=metrics,
            model_path="model.pkl",
            label_column=args.label
        )

        logging.info("Model training and saving completed.")
    
    elif args.mode == "predict":
        if not args.input:
            raise ValueError("Prediction mode requires --input.")
        
        model=load_model("model.pkl")

        data=json.loads(args.input)
        df=pd.DataFrame(data)

        preds=model.predict(df)

        print(preds.tolist())
    
    else:
        raise ValueError("Invalid mode. Use 'train' or 'predict'.")
