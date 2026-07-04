

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PART_FILES = [
    "dataopenaq_part1.csv",
    "dataopenaq_part2.csv",
    "dataopenaq_part3.csv",
    "dataopenaq_part4.csv",
    "dataopenaq_part5.csv",
    "dataopenaq_part6.csv",
]

# соотвеств назв
PARAMETER_MAP = {
    "pm25": "pm25",
    "temperature": "temperature_c",
    "relativehumidity": "humidity_pct",
    "rh": "humidity_pct",
    "um003": "pm03_count",
}


def load_all_parts() -> pd.DataFrame:
    """Читает все части и объединяет их в один длинный датафрейм."""
    frames = []
    for filename in PART_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Пропускаю: файл не найден — {path}")
            continue
        df = pd.read_csv(path)
        frames.append(df)
        print(f"{filename}: {len(df)} строк")

    if not frames:
        raise RuntimeError("Ни один файл частей не найден в data/")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nВсего строк до очистки: {len(combined)}")
    return combined


def clean_and_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает дубликаты, разворачивает в широкий формат по времени."""

    # убирает дубликаты
    before = len(df)
    df = df.drop_duplicates(subset=["datetimeUtc", "parameter", "value"])
    print(f"Убрано дубликатов: {before - len(df)}")

    
    df["parameter"] = df["parameter"].str.lower()
    df["parameter"] = df["parameter"].map(lambda p: PARAMETER_MAP.get(p, p))

    
    pivoted = df.pivot_table(
        index="datetimeUtc",
        columns="parameter",
        values="value",
        aggfunc="mean",  # усред знач
    )

    pivoted = pivoted.reset_index()
    pivoted["datetimeUtc"] = pd.to_datetime(pivoted["datetimeUtc"])
    pivoted = pivoted.sort_values("datetimeUtc").reset_index(drop=True)

    return pivoted


def main():
    combined = load_all_parts()
    result = clean_and_pivot(combined)

    print(f"\nИтоговый датасет: {len(result)} строк, период с "
          f"{result['datetimeUtc'].min()} по {result['datetimeUtc'].max()}")
    print(f"Колонки: {list(result.columns)}")

    # проверка пропусков
    expected_hours = pd.date_range(
        start=result["datetimeUtc"].min(),
        end=result["datetimeUtc"].max(),
        freq="h",
    )
    missing_hours = len(expected_hours) - len(result)
    print(f"Ожидалось часов в диапазоне: {len(expected_hours)}, "
          f"реально строк: {len(result)}, пропущено часов: {missing_hours}")

    output_path = DATA_DIR / "openaq_merged.csv"
    result.to_csv(output_path, index=False)
    print(f"\nСохранено: {output_path}")


if __name__ == "__main__":
    main()