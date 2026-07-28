


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

pred_val = model.predict(X_val_scaled)
pred_test = model.predict(X_test_scaled)

print("--- Ridge Regression ---")
print(f"Validation: MAE={mae(y_val, pred_val):.2f}, RMSE={rmse(y_val, pred_val):.2f}")
print(f"Test:       MAE={mae(y_test, pred_test):.2f}, RMSE={rmse(y_test, pred_test):.2f}")

print("\nдля сравнения, baseline на test:")
print("Persistence: MAE=7.83")
print("Moving average (24ч): MAE=4.09")


print("\n--- коэффициенты Ridge (важность признаков) ---")
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))