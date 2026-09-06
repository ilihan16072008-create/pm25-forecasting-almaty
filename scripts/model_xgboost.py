import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

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

model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
)


model.fit(X_train, y_train)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

pred_val = model.predict(X_val)
pred_test = model.predict(X_test)

print("--- XGBoost ---")
print(f"Validation: MAE={mae(y_val, pred_val):.2f}, RMSE={rmse(y_val, pred_val):.2f}")
print(f"Test:       MAE={mae(y_test, pred_test):.2f}, RMSE={rmse(y_test, pred_test):.2f}")

print("\n--- Итоговое сравнение всех моделей (test) ---")
print("Persistence baseline:        MAE=7.83")
print("Moving average baseline:     MAE=4.09  <- лучший результат")
print("Seasonal average baseline:   MAE=45.69")
print("Ridge Regression:            MAE=8.04")
print(f"Random Forest:                MAE=7.85")
print(f"XGBoost:                      MAE={mae(y_test, pred_test):.2f}")
