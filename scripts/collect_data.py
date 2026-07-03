

import os
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

API_KEY = os.environ.get("IQAIR_API_KEY")
CITY = "Almaty"
STATE = "Almaty Oblysy"
COUNTRY = "Kazakhstan"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "air_quality_history.csv"

CSV_HEADERS = [
    "timestamp_utc",
    "pm25_conc",       
    "aqi_us",          
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_ms",
    "wind_direction_deg",
]


def fetch_air_quality() -> dict:
    """Делает запрос к IQAir API и возвращает нужные поля."""
    if not API_KEY:
        raise RuntimeError(
            "IQAIR_API_KEY не найден в переменных окружения. "
            "Проверьте GitHub Secrets или локальный .env"
        )

    url = "https://api.airvisual.com/v2/city"
    params = {
        "city": CITY,
        "state": STATE,
        "country": COUNTRY,
        "key": API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(f"API вернул ошибку: {payload}")

    current = payload["data"]["current"]
    pollution = current["pollution"]
    weather = current["weather"]

    pm25_conc = None
    if "p2" in pollution and isinstance(pollution["p2"], dict):
        pm25_conc = pollution["p2"].get("conc")

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pm25_conc": pm25_conc,
        "aqi_us": pollution.get("aqius"),
        "temperature_c": weather.get("tp"),
        "humidity_pct": weather.get("hu"),
        "pressure_hpa": weather.get("pr"),
        "wind_speed_ms": weather.get("ws"),
        "wind_direction_deg": weather.get("wd"),
    }
    return row


def append_row(row: dict) -> None:
    """Дописывает строку в CSV, создавая файл с заголовком, если его ещё нет."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = DATA_FILE.exists()

    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    try:
        row = fetch_air_quality()
    except Exception as e:
        print(f"Ошибка при получении данных: {e}", file=sys.stderr)
        sys.exit(1)

    append_row(row)
    print(f"Записано: {row}")


if __name__ == "__main__":
    main()