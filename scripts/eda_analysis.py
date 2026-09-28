import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PLOTS_DIR = DATA_DIR / "eda_plots"
PLOTS_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "openaq_merged.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetimeUtc"])
    df = df.sort_values("datetimeUtc").reset_index(drop=True)
    return df


def analyze_gaps(df: pd.DataFrame) -> None:
    full_range = pd.date_range(
        start=df["datetimeUtc"].min(),
        end=df["datetimeUtc"].max(),
        freq="h",
    )
    existing = set(df["datetimeUtc"])
    missing = [ts for ts in full_range if ts not in existing]

    print(f"Total missing hours: {len(missing)}")

    if not missing:
        print("No missing hours.")
        return

    missing_series = pd.Series(missing).sort_values().reset_index(drop=True)
    gaps = []
    block_start = missing_series[0]
    prev = missing_series[0]

    for ts in missing_series[1:]:
        if ts - prev > pd.Timedelta(hours=1):
            gaps.append((block_start, prev))
            block_start = ts
        prev = ts
    gaps.append((block_start, prev))

    print(f"\nMissing hours grouped into {len(gaps)} block(s):")
    for start, end in gaps:
        duration_hours = int((end - start).total_seconds() / 3600) + 1
        print(f"  {start} -> {end}  ({duration_hours} h)")

def plot_pm25_overview(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetimeUtc"], df["pm25"], linewidth=0.5, color="steelblue")
    ax.set_title("PM2.5 in Almaty: Full Observation Period")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5, µg/m³")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pm25_full_period.png", dpi=120)
    plt.close(fig)
    print(f"Saved plot: {PLOTS_DIR / 'pm25_full_period.png'}")

def plot_daily_pattern(df: pd.DataFrame) -> None:
    df = df.copy()
    df["hour"] = df["datetimeUtc"].dt.hour
    hourly_avg = df.groupby("hour")["pm25"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_avg.index, hourly_avg.values, marker="o", color="darkorange")
    ax.set_title("Average PM2.5 by Hour of Day")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Average PM2.5, µg/m³")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pm25_daily_pattern.png", dpi=120)
    plt.close(fig)
    print(f"Saved plot: {PLOTS_DIR / 'pm25_daily_pattern.png'}")

def plot_monthly_distribution(df: pd.DataFrame) -> None:
    df = df.copy()
    df["month"] = df["datetimeUtc"].dt.to_period("M").astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    df.boxplot(column="pm25", by="month", ax=ax)
    ax.set_title("PM2.5 Distribution by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5, µg/m³")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pm25_monthly_boxplot.png", dpi=120)
    plt.close(fig)
    print(f"Saved plot: {PLOTS_DIR / 'pm25_monthly_boxplot.png'}")

def print_basic_stats(df: pd.DataFrame) -> None:
    print("\nBasic statistics for PM2.5:")
    print(df["pm25"].describe())
    print(f"\nMissing values (NaN) per column:")
    print(df.isna().sum())


def main():
    df = load_data()
    print(f"Rows loaded: {len(df)}")
    print(f"Period: {df['datetimeUtc'].min()} -> {df['datetimeUtc'].max()}\n")

    analyze_gaps(df)
    print_basic_stats(df)

    plot_pm25_overview(df)
    plot_daily_pattern(df)
    plot_monthly_distribution(df)

    print("\nDone. Plots saved in data/eda_plots/")

if __name__ == "__main__":
    main()