import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
print("Loading data...")
train = pd.read_csv("data/train_test.csv")
val = pd.read_csv("data/validation.csv")
template = pd.read_csv("data/validation_predictions_template.csv")

train["date"] = pd.to_datetime(train["date"])
val["date"] = pd.to_datetime(val["date"])

print(f"Train shape: {train.shape}")
print(f"Validation shape: {val.shape}")

# 2. Feature engineering
def engineer_features(df, is_train=True, te_maps=None):
    """Create model features. Fit target encodings only on training data."""
    df = df.copy()

    # missing values
    df["weight"] = df["weight"].fillna(df["weight"].median())
    df["market_index"] = df["market_index"].fillna(df["market_index"].median())

    # Distance features
    df["log_distance"] = np.log1p(df["distance"])

    # Date features
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # Equipment one-hot encoding
    df = pd.get_dummies(df, columns=["equipment"], prefix="eq")

    # Target encoding for pickup & delivery locations
    if is_train:
        te_maps = {}
        global_mean = df["posted_rate"].mean()
        for col in ["pickup", "delivery"]:
            te_maps[col] = df.groupby(col)["posted_rate"].mean().to_dict()
            df[f"{col}_te"] = df[col].map(te_maps[col]).fillna(global_mean)
        return df, te_maps
    else:
        global_mean = np.mean(list(te_maps["pickup"].values()))
        for col in ["pickup", "delivery"]:
            df[f"{col}_te"] = df[col].map(te_maps[col]).fillna(global_mean)
        return df

print("Engineering features...")
train_fe, te_maps = engineer_features(train, is_train=True)
val_fe = engineer_features(val, is_train=False, te_maps=te_maps)

# Interaction features
for df in [train_fe, val_fe]:
    df["dist_x_market"] = df["distance"] * df["market_index"]
    df["dist_x_quote"] = df["distance"] * df["quote_signal"]

# 3. Feature list
FEATURES = [
    "distance",
    "log_distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "dayofweek",
    "is_weekend",
    "weekofyear",
    "pickup_te",
    "delivery_te",
    "eq_Dry Van",
    "eq_Reefer",
    "eq_Flatbed",
    "dist_x_market",
    "dist_x_quote",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

# Ensure all feature columns exist
for col in FEATURES:
    if col not in train_fe.columns:
        train_fe[col] = 0
    if col not in val_fe.columns:
        val_fe[col] = 0

# 4. Time-based validation split
print("Creating time-based validation split...")
cutoff = pd.Timestamp("2025-08-31")
train_set = train_fe[train_fe["date"] < cutoff].copy()
valid_set = train_fe[train_fe["date"] >= cutoff].copy()

X_train = train_set[FEATURES]
y_train = train_set["posted_rate"]
X_valid = valid_set[FEATURES]
y_valid = valid_set["posted_rate"]

print(f"Train period : {train_set['date'].min().date()} → {train_set['date'].max().date()} ({len(train_set):,} rows)")
print(f"Valid period : {valid_set['date'].min().date()} → {valid_set['date'].max().date()} ({len(valid_set):,} rows)")


# 5. Train XGBoost with early stopping
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=8,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=150,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=100,
)

# Evaluate
preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
rmse = np.sqrt(mean_squared_error(y_valid, preds))

print(f"\nValidation MAE : {mae:.2f}")
print(f"Validation RMSE: {rmse:.2f}")

# Feature importance
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop feature importances:")
print(imp.head(10))


# 6. Retrain on full data (Jan–Oct)
print("\nRetraining on full training data...")
X_full = train_fe[FEATURES]
y_full = train_fe["posted_rate"]

final_model = xgb.XGBRegressor(
    n_estimators=model.best_iteration + 50,
    learning_rate=0.02,
    max_depth=8,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X_full, y_full, verbose=False)

# 7. Predict on real validation set (Nov–Dec)
print("Generating final predictions...")
final_preds = final_model.predict(val_fe[FEATURES])

submission = template.copy()
submission["predicted_rate"] = final_preds
submission.to_csv("validation_predictions.csv", index=False)

print("Saved: validation_predictions.csv")
print(submission["predicted_rate"].describe())


# 8. Create December chart inputs (fixed lane)
print("\nCreating December chart inputs...")

december_dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")

december_raw = pd.DataFrame({
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "pickup_lat": 38.0406,
    "pickup_lon": -84.5037,
    "delivery_lat": 41.0793,
    "delivery_lon": -85.1394,
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32000.0,
    "date": december_dates,
    "market_index": train["market_index"].median(),
    "quote_signal": train["quote_signal"].median(),
})

december_fe = engineer_features(december_raw, is_train=False, te_maps=te_maps)
december_fe["dist_x_market"] = december_fe["distance"] * december_fe["market_index"]
december_fe["dist_x_quote"] = december_fe["distance"] * december_fe["quote_signal"]

for col in FEATURES:
    if col not in december_fe.columns:
        december_fe[col] = 0

december_preds = final_model.predict(december_fe[FEATURES])

december_output = pd.DataFrame({
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32000.0,
    "date": december_dates,
    "predicted_rate": december_preds,
})

december_output.to_csv("december_chart_inputs.csv", index=False)
print("Saved: december_chart_inputs.csv")
print(f"December predicted rate range: {december_preds.min():.2f} → {december_preds.max():.2f}")

print("\n=== All done ===")
print("Next step: run the scorer")
print("python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv --output-dir scorer_results")
