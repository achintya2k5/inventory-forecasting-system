# AWS Hackathon Demand Forecasting

A small end-to-end Python project that forecasts item-level sales and generates inventory reorder recommendations.

## What It Does

- Loads historical sales data from `training_data.csv`
- Cleans data and creates demand forecasting features
- Trains (or loads) an XGBoost regressor
- Reports train/validation/test MAE
- Produces `inventory_recommendations.csv` with reorder flags and suggested order quantity

## Project Structure

- `main.py`: CLI entrypoint
- `ml_data_loader/loader.py`: data loading and pipeline orchestration
- `ml_data_loader/model.py`: prediction runner and evaluation
- `ml_data_loader/utils.py`: cleaning, EDA, training, metadata, inventory logic
- `docs/input_format.txt`: expected feature schema
- `docs/output_format.txt`: expected output schema

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the pipeline (non-interactive)

```powershell
python main.py --cutoff 2024-05-20 --retrain
```

You can also run without a cutoff date. It will use an automatic 80/20 temporal split:

```powershell
python main.py --retrain
```

## Outputs

- `inventory_recommendations.csv`: predicted demand and reorder suggestions
- `trained_model.pkl`: saved model artifact
- `model_metadata.json`: model metadata (feature list, train date, train/validation MAE)

## Testing

```powershell
pytest -q
```

## Notes

- Do not commit real or sensitive data files.
- `training_data.csv` should be local-only for experimentation, unless it is synthetic/public data.
