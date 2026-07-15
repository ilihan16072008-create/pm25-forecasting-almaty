
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "features.csv"


df = pd.read_csv(INPUT_FILE, parse_dates=["datetimeUtc"])
df = df.sort_values("datetimeUtc").reset_index(drop=True)

# 70% 15% разбиение
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"Train: {len(train)} строк ({train['datetimeUtc'].min()} -> {train['datetimeUtc'].max()})")
print(f"Val:   {len(val)} строк ({val['datetimeUtc'].min()} -> {val['datetimeUtc'].max()})")
print(f"Test:  {len(test)} строк ({test['datetimeUtc'].min()} -> {test['datetimeUtc'].max()})")

# ррасчет метрик
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

y_test = test["target_pm25_next24h_avg"].values

# последнее известное значение PM2.5
pred_persistence = test["pm25"].values

#Baseline 2прогноз  сред PM2.5 за ласт 24ч
pred_moving_avg = test["pm25_rolling_mean_24h"].values

hourly_avg_train = train.groupby("hour")["pm25"].mean()
pred_seasonal = test["hour"].map(hourly_avg_train).values

print("\n--- Baseline метрики (на test) ---")
for name, preds in [
    ("Persistence (последнее значение)", pred_persistence),
    ("Moving average (24ч)", pred_moving_avg),
    ("Seasonal average (по месяцу)", pred_seasonal),
]:
    print(f"{name}: MAE={mae(y_test, preds):.2f}, RMSE={rmse(y_test, preds):.2f}")