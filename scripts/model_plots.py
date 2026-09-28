import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "features.csv"
PLOTS_DIR = DATA_DIR / "eda_plots"

df = pd.read_csv(INPUT_FILE, parse_dates=["datetimeUtc"])
df = df.sort_values("datetimeUtc").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
test = df.iloc[val_end:]

feature_cols = [
    "pm25", "humidity_pct", "temperature_c", "pm03_count",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_12h", "pm25_lag_24h", "pm25_lag_48h",
    "pm25_rolling_mean_6h", "pm25_rolling_mean_12h", "pm25_rolling_mean_24h", "pm25_rolling_mean_48h",
    "hour", "day_of_week", "month",
]
target_col = "target_pm25_next24h_avg"

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]

model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
pred_test = model.predict(X_test)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test["datetimeUtc"], y_test.values, label="Actual", color="steelblue", linewidth=1)
ax.plot(test["datetimeUtc"], pred_test, label="Predicted (XGBoost)", color="darkorange", linewidth=1, alpha=0.8)
ax.set_title("XGBoost: Predicted vs Actual PM2.5 (Test Set)")
ax.set_xlabel("Date")
ax.set_ylabel("PM2.5, µg/m³")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "xgboost_predicted_vs_actual.png", dpi=120)
plt.close(fig)
print(f"Saved plot: {PLOTS_DIR / 'xgboost_predicted_vs_actual.png'}")

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, pred_test, alpha=0.4, color="steelblue", s=15)
max_val = max(y_test.max(), pred_test.max())
ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Perfect prediction")
ax.set_xlabel("Actual PM2.5, µg/m³")
ax.set_ylabel("Predicted PM2.5, µg/m³")
ax.set_title("XGBoost: Predicted vs Actual (Scatter Plot)")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "xgboost_scatter.png", dpi=120)
plt.close(fig)
print(f"Saved plot: {PLOTS_DIR / 'xgboost_scatter.png'}")