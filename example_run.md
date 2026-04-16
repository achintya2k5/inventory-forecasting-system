# Example Run

Use the command below to run the pipeline non-interactively:

```powershell
python main.py --cutoff 2024-05-20 --retrain
```

Expected generated files:

- `inventory_recommendations.csv`
- `trained_model.pkl`
- `model_metadata.json`

Sample output columns in `inventory_recommendations.csv`:

- `item_id`
- `inventory_level`
- `predicted_sales`
- `reorder_flag`
- `suggested_order_qty`
