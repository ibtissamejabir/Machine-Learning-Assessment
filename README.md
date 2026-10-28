# Load Rate Prediction – Machine Learning Assessment

## Overview
This repository contains the solution for predicting load rates (posted_rate) using historical freight data.

- **Training data**: 48,000 loads (January – October 2025)
- **Validation data**: 12,000 loads (November – December 2025)
- **Model**: XGBoost Regressor
- **Validation strategy**: Time-based split

## Project Structure
```
.
├── data/                          # Place original data files here (not included in repo)
│   ├── train_test.csv
│   ├── validation.csv
│   └── validation_predictions_template.csv
├── train_and_predict.py           # Main training & prediction script
├── score.py                       # Provided scoring script
├── validation_predictions.csv     # Final predictions (load_id, predicted_rate)
├── december_chart_inputs.csv      # Fixed December lane predictions for chart
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

1. Place the provided data files into the `data/` folder.

2. Run the training and prediction pipeline:
```bash
python train_and_predict.py
```
This will:
- Engineer features
- Perform time-based validation
- Retrain on full data
- Generate `validation_predictions.csv`

3. Generate the required December chart:
```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december_chart_inputs.csv \
  --output-dir scorer_results
```

The chart will be saved as `scorer_results/candidate_december.png`.

## Approach Summary

### Data Split
- **Internal validation**: Trained on Jan–Aug 2025, validated on Sep–Oct 2025 (time-based)
- **Final model**: Retrained on all available data (Jan–Oct 2025) before predicting Nov–Dec

### Feature Engineering
- Distance and log(distance)
- Date features (month, day of week, is_weekend, week of year)
- Equipment type (one-hot encoded)
- Target encoding for pickup and delivery locations
- Interaction features (distance × market_index, distance × quote_signal)
- Latitude / longitude coordinates

### Model
XGBoost Regressor with early stopping.  
Key hyperparameters: `learning_rate=0.02`, `max_depth=8`, `n_estimators=3000` (with early stopping).

### Data Quality
- Missing values in `weight` and `market_index` were imputed with the median.
- No zero or negative rates found.

## Results
- Validation MAE ≈ 143.7
- Validation RMSE ≈ 638.6
- Final predictions generated for all 12,000 validation loads.
