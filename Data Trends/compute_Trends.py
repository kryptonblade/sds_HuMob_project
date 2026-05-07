#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# FILE PATHS
# =========================================================
FILES = {
    "A": "/home/kryptonblade/Downloads/city_A_alldata.parquet",
    "B": "/home/kryptonblade/Downloads/city_B_alldata.parquet",
    "C": "/home/kryptonblade/Downloads/city_C_alldata.parquet",
    "D": "/home/kryptonblade/Downloads/city_D_alldata.parquet",
}


# =========================================================
# LOAD
# =========================================================
def load_data(path):
    print(f"📥 Loading {path}")
    return pd.read_parquet(path)


# =========================================================
# DAILY AVG RECORDS PER USER
# =========================================================
def daily_avg_per_user(df):
    total_records = df.groupby('d').size()
    unique_users = df.groupby('d')['uid'].nunique()

    avg = total_records / unique_users
    return avg


# =========================================================
# TIME-OF-DAY AVG (MATCHES PAPER FIGURE B)
# =========================================================
def time_of_day_avg(df):
    total_records = df.groupby('t').size()

    num_days = df['d'].nunique()
    num_users = df['uid'].nunique()

    avg = total_records / (num_days * num_users)
    return avg


# =========================================================
# PLOTTING
# =========================================================
def plot_trends(city_data):

    # ---------------- FIGURE A ----------------
    plt.figure()

    for city, df in city_data.items():
        daily_avg = daily_avg_per_user(df)
        plt.plot(daily_avg.index, daily_avg.values, marker='o', label=city)

    plt.title("Average Mobility Records per User (Daily)")
    plt.xlabel("Day")
    plt.ylabel("# Records per User")
    plt.legend(title="City")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_A_daily_trend.png")

    print("Saved: figure_A_daily_trend.png")


    # ---------------- FIGURE B ----------------
    plt.figure()

    for city, df in city_data.items():
        tod_avg = time_of_day_avg(df)
        plt.plot(tod_avg.index, tod_avg.values, marker='o', label=city)

    plt.title("Average Mobility Records per User (Time of Day)")
    plt.xlabel("Time Slot (0–48)")
    plt.ylabel("# Records per User")
    plt.legend(title="City")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_B_time_of_day.png")

    print("Saved: figure_B_time_of_day.png")


# =========================================================
# MAIN
# =========================================================
def main():
    city_data = {}

    for city, path in FILES.items():
        path = Path(path)

        if not path.exists():
            print(f" Missing: {path}")
            continue

        city_data[city] = load_data(path)

    plot_trends(city_data)


if __name__ == "__main__":
    main()