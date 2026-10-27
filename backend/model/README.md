# Model artifacts

| File | Source | Committed to Git? |
|------|--------|-------------------|
| `best_fraud_model_tuned.pkl` | Your original file, **byte-for-byte unmodified**. A `joblib` dict: `model_name`, `model` (tuned `XGBClassifier`), `threshold` (0.9084), `features` (9 names, in training order). | Yes (331 KB) |
| `preprocessing.json` | Generated from `data/Fraud.csv` with the notebook's fitted preprocessing values and verified against its saved-model results. See step 3 in the main README to regenerate it. | Yes (it only holds summary statistics, no transaction data) |

Why `preprocessing.json` is required: the model was trained on *scaled* features. Feeding it
unscaled values gives wrong scores without any error, so the API returns `503` if this file
is missing or invalid instead of guessing. The file is present in this project and must be
kept alongside the model when deploying.

Loading the `.pkl` requires `xgboost-cpu==2.1.4` (pinned in `backend/requirements.txt`).
Only load pickle files you trust: unpickling can execute code.
