
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "features.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["datetimeUtc"])
df = df.sort_values("datetimeUtc").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

feature_cols = [
    "pm25", "humidity_pct", "temperature_c", "pm03_count",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_12h", "pm25_lag_24h", "pm25_lag_48h",
    "pm25_rolling_mean_6h", "pm25_rolling_mean_12h", "pm25_rolling_mean_24h", "pm25_rolling_mean_48h",
    "hour", "day_of_week", "month",
]
target_col = "target_pm25_next24h_avg"

X_train, y_train = train[feature_cols], train[target_col]
X_val, y_val = val[feature_cols], val[target_col]
X_test, y_test = test[feature_cols], test[target_col]

model = RandomForestRegressor(
    n_estimators=200,   
    max_depth=10,       
    random_state=42,    
    n_jobs=-1,           
)
model.fit(X_train, y_train)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

pred_val = model.predict(X_val)
pred_test = model.predict(X_test)

print("--- Random Forest ---")
print(f"Validation: MAE={mae(y_val, pred_val):.2f}, RMSE={rmse(y_val, pred_val):.2f}")
print(f"Test:       MAE={mae(y_test, pred_test):.2f}, RMSE={rmse(y_test, pred_test):.2f}")

print("\n--- Для сравнения ---")
print("Baseline Moving average (24ч): MAE=4.09")
print("Ridge Regression (test): MAE=8.04")

print("\n--- Feature importance (Random Forest) ---")
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance_df.to_string(index=False))

