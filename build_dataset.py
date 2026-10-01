import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve

import pandas as pd

# ---- settings you can tweak ----
MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
LAT, LON = 40.7128, -74.0060          # New York City
TIMEZONE = "America/New_York"          # TLC timestamps are local NYC time
DATA_DIR = "data"
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"
RAIN_THRESHOLD_MM = 0.5                # hour counts as "rainy" at or above this


def download_month(ym):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "yellow_tripdata_%s.parquet" % ym)
    if not os.path.exists(path):
        print("Downloading %s ..." % ym)
        urlretrieve(TRIP_URL.format(ym=ym), path)
    return path


def hourly_counts(path, ym):
    """Number of taxi pickups in each hour of one month's file."""
    df = pd.read_parquet(path, columns=["tpep_pickup_datetime"])
    ts = df["tpep_pickup_datetime"]
    ts = ts[ts.dt.strftime("%Y-%m") == ym]          # drop stray dates in the file
    return ts.dt.floor("60min").value_counts()


def fetch_weather(start, end):
    params = urlencode({
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "precipitation,temperature_2m",
        "timezone": TIMEZONE,
    })
    url = "https://archive-api.open-meteo.com/v1/archive?" + params
    with urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    h = data["hourly"]
    weather = pd.DataFrame({
        "hour": pd.to_datetime(h["time"]),
        "rain_mm": h["precipitation"],
        "temp_c": h["temperature_2m"],
    })
    return weather.set_index("hour")


def build(counts, weather, first_day, last_day):
    """Join hourly trip counts with weather on a complete hourly index."""
    idx = pd.date_range(first_day, last_day + pd.Timedelta(hours=23), freq="60min")
    df = pd.DataFrame(index=idx)
    df.index.name = "hour"
    df["trips"] = counts.reindex(idx, fill_value=0).astype(int)
    df = df.join(weather)
    df["rain_mm"] = df["rain_mm"].fillna(0.0)
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek          # Monday = 0
    df["is_rainy"] = (df["rain_mm"] >= RAIN_THRESHOLD_MM).astype(int)
    return df


def main():
    all_counts = []
    for ym in MONTHS:
        path = download_month(ym)
        all_counts.append(hourly_counts(path, ym))
    counts = pd.concat(all_counts)

    first_day = pd.Timestamp(MONTHS[0] + "-01")
    last_day = pd.Timestamp(MONTHS[-1] + "-01") + pd.offsets.MonthEnd(0)

    print("Fetching weather ...")
    weather = fetch_weather(first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d"))

    df = build(counts, weather, first_day, last_day)
    out = os.path.join(DATA_DIR, "hourly_demand_weather.csv")
    df.to_csv(out)

    print("\nSaved %s" % out)
    print("Rows (hours): %d" % len(df))
    print("Rainy hours:  %d (%.1f%%)" % (df["is_rainy"].sum(), 100 * df["is_rainy"].mean()))
    print("\nAverage trips per hour:")
    print(df.groupby("is_rainy")["trips"].mean().round(0).rename({0: "dry", 1: "rainy"}))


if __name__ == "__main__":
    main()