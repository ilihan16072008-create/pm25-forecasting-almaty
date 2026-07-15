

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "openaq_merged.csv"
OUTPUT_FILE = DATA_DIR / "features.csv"

#чтение данных
df = pd.read_csv(INPUT_FILE, parse_dates=["datetimeUtc"])
df = df.sort_values("datetimeUtc").reset_index(drop=True)

print(f"Загружено строк: {len(df)}")

# сдвиг колоны вниз чтобы полцчить знач за час
lag_hours = [1, 3, 6, 12, 24, 48]
for lag in lag_hours:
    df[f"pm25_lag_{lag}h"] = df["pm25"].shift(lag)

# не включайй текущ час в расчет брать подряд
rolling_windows = [6, 12, 24, 48]
for window in rolling_windows:
    df[f"pm25_rolling_mean_{window}h"] = df["pm25"].shift(1).rolling(window).mean()

# --- Календарные признаки ---
df["hour"] = df["datetimeUtc"].dt.hour
df["day_of_week"] = df["datetimeUtc"].dt.dayofweek  # 0 = понедельник
df["month"] = df["datetimeUtc"].dt.month

# свдиг данных на 24 стрк
df["target_pm25_next24h_avg"] = df["pm25"].shift(-24).rolling(24).mean()

# в датасете отсвт 48 час данные поэтому из убираем
before = len(df)
df = df.dropna()
print(f"Строк до очистки: {before}, после: {len(df)} (удалено {before - len(df)})")

# --- Сохраняем результат ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"Сохранено: {OUTPUT_FILE}")
print(f"Колонки: {list(df.columns)}")