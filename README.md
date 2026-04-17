# Demand Forecasting and Inventory Recommendation System

Production-ready demand forecasting service with a reusable ML pipeline, CLI workflows, and FastAPI inference endpoints.

## Architecture Overview

The repository provides three aligned layers:

1. Training and prediction pipeline
- Data cleaning and feature engineering
- Scikit-learn preprocessing + RandomForest regression model
- Model artifact and metadata export
- Inventory recommendation generation

2. CLI interface
- Train mode for model training and artifact creation
- Predict mode for local batch inference from JSON payloads

3. API service
- Health endpoint for service readiness
- Predict endpoint for model inference via HTTP

## Features

- End-to-end demand forecasting workflow
- Feature engineering with lag and rolling demand signals
- Unified preprocessing and model pipeline with scikit-learn
- Inventory reorder decision support logic
- FastAPI service for inference deployment
- CLI-based local experimentation and automation
- Unit tests for model and utility behavior
- Docker support for API deployment
- GitHub Actions CI for test automation

## Tech Stack

- Python 3.10+
- Pandas and NumPy
- Scikit-learn
- FastAPI and Uvicorn
- Pytest
- Docker
- GitHub Actions

## Installation

1. Clone the repository

~~~bash
git clone https://github.com/achintya2k5/inventory-forecasting-system.git
cd inventory-forecasting-system
~~~

2. Create and activate a virtual environment

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

3. Install dependencies

~~~powershell
pip install -r requirements.txt
~~~

For development tooling:

~~~powershell
pip install -r requirements-dev.txt
~~~

## Usage

### CLI Training

Train a model from local CSV data and generate model artifacts.

~~~powershell
python main.py --mode train --data path\to\training_data.csv --label sales
~~~

Generated artifacts:
- model.pkl
- model_metadata.json
- inventory_recommendations.csv (when inventory_level is available in training data)

### CLI Prediction

Run local predictions from either JSON string input or JSON file path.

~~~powershell
python main.py --mode predict --model-path model.pkl --input "[{\"lag_1\": 10, \"rolling_mean_3\": 12, \"day_of_week\": 2, \"is_holiday\": 0, \"promotion_flag\": 1, \"weather_temp\": 30}]"
~~~

Or using a JSON file:

~~~powershell
python main.py --mode predict --model-path model.pkl --input data\sample_input.json
~~~

### API Usage

Start the API:

~~~powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
~~~

Health check:

~~~bash
curl http://localhost:8000/health
~~~

Prediction request:

~~~bash
curl -X POST "http://localhost:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"data": [
			{
				"lag_1": 10,
				"rolling_mean_3": 12,
				"day_of_week": 2,
				"is_holiday": 0,
				"promotion_flag": 1,
				"weather_temp": 30
			}
		]
	}'
~~~

## Project Structure

~~~text
.
├── api/
│   └── main.py
├── data/
│   └── .gitkeep
├── docs/
│   ├── input_format.txt
│   └── output_format.txt
├── ml_data_loader/
│   ├── __init__.py
│   ├── loader.py
│   ├── model.py
│   ├── preprocess.py
│   └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_model.py
│   └── test_utils.py
├── .github/workflows/
│   └── ci.yml
├── .gitignore
├── Dockerfile
├── LICENSE
├── main.py
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
└── README.md
~~~

## Input and Output Schema

Reference files:
- docs/input_format.txt
- docs/output_format.txt

API prediction input schema:

~~~json
{
	"data": [
		{
			"lag_1": 42.0,
			"rolling_mean_3": 39.5,
			"day_of_week": 2,
			"is_holiday": 0,
			"promotion_flag": 1,
			"weather_temp": 29.8
		}
	]
}
~~~

API prediction output schema:

~~~json
{
	"predictions": [43.21]
}
~~~

Inventory recommendation file schema:

~~~text
item_id, inventory_level, predicted_sales, reorder_flag, suggested_order_qty
~~~

## Model Details

- Model: RandomForestRegressor
- Preprocessing:
	- Numeric features: median imputation and scaling
	- Categorical features: mode imputation and one-hot encoding
- Engineered features:
	- lag_1
	- rolling_mean_3
	- day_of_week
	- is_holiday
	- promotion_flag
	- weather_temp
- Evaluation metric: Mean Absolute Error (MAE)

## Testing

Run all tests:

~~~powershell
pytest -q
~~~

## Docker

Build image:

~~~bash
docker build -t demand-forecasting-api .
~~~

Run container:

~~~bash
docker run --rm -p 8000:8000 demand-forecasting-api
~~~

## Future Improvements

- Add model version registry and artifact tracking
- Add automated drift and data quality monitoring
- Add cross-validation and hyperparameter tuning workflows
- Add API authentication and rate limiting
- Add contract tests for API schema validation
