"""Step 4: how much could ride pooling save, and what does a women-only pool cost?

Idea: take real NYC yellow-taxi trips. If single riders start in the same pickup
zone, within a few minutes of each other, and go to the same dropoff zone, put
them in one cab. The cab drives the longest of their trips (plus a detour
premium) and the fare is split in proportion to what each rider would have paid
solo. Then repeat with only a subset of riders eligible (women-only pool).

Put this next to the other scripts and run:  python3 pooling_sim.py
(Takes a minute or two. For a faster run, set MONTHS = ["2024-05"].)
"""
import numpy as np
import pandas as pd

# ---- assumptions you can tweak (state them in your README) ----
MONTHS = ["2024-05", "2024-06"]
DATA_DIR = "data"
WEATHER_CSV = "data/hourly_demand_weather.csv"
PEAK_HOURS = [8, 9, 10, 17, 18, 19, 20, 21]
DETOUR_PREMIUM = 0.15     # shared cab costs the longest solo fare x 1.15
POOL_SIZE = 2             # max riders per cab
HEADLINE_WINDOW = 5       # minutes, used for the rain / peak breakdown
WINDOWS = [2, 5, 10]      # minutes, for the sensitivity sweep
WOMEN_SHARES = [0.5, 0.3] # ASSUMED share of riders who are women (no gender in the data)
SEED = 0


def load_trips():
    cols = ["tpep_pickup_datetime", "passenger_count", "trip_distance",
            "PULocationID", "DOLocationID", "fare_amount"]
    frames = []
    for ym in MONTHS:
        d = pd.read_parquet("%s/yellow_tripdata_%s.parquet" % (DATA_DIR, ym), columns=cols)
        frames.append(d[d["tpep_pickup_datetime"].dt.strftime("%Y-%m") == ym])
    df = pd.concat(frames, ignore_index=True)

    # Only single riders can be pooled; drop junk rows, same-zone hops, unknown zones
    df = df[(df["passenger_count"] == 1) & (df["fare_amount"] > 0) & (df["trip_distance"] > 0)]
    df = df[df["PULocationID"] != df["DOLocationID"]]
    df = df[~df["PULocationID"].isin([264, 265]) & ~df["DOLocationID"].isin([264, 265])]
    df = df.reset_index(drop=True)

    # Tag each trip with hour-of-day and whether that hour was rainy
    wx = pd.read_csv(WEATHER_CSV, parse_dates=["hour"]).set_index("hour")
    df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")
    df["is_rainy"] = df["hour"].map(wx["is_rainy"]).fillna(0).astype(int)
    df["is_peak"] = df["tpep_pickup_datetime"].dt.hour.isin(PEAK_HOURS).astype(int)

    # One fixed random draw per trip so women-only groups are nested (30% is inside 50%)
    df["u"] = np.random.default_rng(SEED).random(len(df))
    return df.drop(columns=["passenger_count", "hour"])


def simulate(trips, window_min, k=POOL_SIZE):
    """Group trips by (pickup zone, dropoff zone, time bucket), chunk into cabs of k."""
    t = trips[["tpep_pickup_datetime", "PULocationID", "DOLocationID",
               "fare_amount", "trip_distance", "is_rainy", "is_peak"]].copy()
    t["bucket"] = t["tpep_pickup_datetime"].dt.floor("%dmin" % window_min)
    keys = ["PULocationID", "DOLocationID", "bucket"]
    t["chunk"] = t.groupby(keys).cumcount() // k
    g = t.groupby(keys + ["chunk"])
    t["n"] = g["fare_amount"].transform("size")
    t["sum_fare"] = g["fare_amount"].transform("sum")
    t["max_fare"] = g["fare_amount"].transform("max")
    t["sum_miles"] = g["trip_distance"].transform("sum")
    t["max_miles"] = g["trip_distance"].transform("max")
    t["shared_fare"] = t["max_fare"] * (1 + DETOUR_PREMIUM)
    # Riders only pool if it is actually cheaper in total
    t["pooled"] = (t["n"] >= 2) & (t["shared_fare"] < t["sum_fare"])
    return t


def summarize(f):
    if len(f) == 0:
        return {"riders": 0, "match_%": np.nan, "saving_%": np.nan,
                "cab_trips_removed_%": np.nan, "miles_saved_%": np.nan}
    p = f["pooled"]
    # Proportional split: each rider pays shared_fare * solo/sum_solo, so everyone
    # in a cab gets the same % discount = 1 - shared_fare / sum_fare
    saving = (1 - f.loc[p, "shared_fare"] / f.loc[p, "sum_fare"]).mean() * 100 if p.any() else 0.0
    removed = (p * (1 - 1 / f["n"])).sum() / len(f) * 100
    miles = (p * f["trip_distance"] * (1 - f["max_miles"] / f["sum_miles"])).sum() \
        / f["trip_distance"].sum() * 100
    return {"riders": len(f), "match_%": round(p.mean() * 100, 1),
            "saving_%": round(saving, 1), "cab_trips_removed_%": round(removed, 1),
            "miles_saved_%": round(miles, 1)}


def main():
    print("Loading trips ...")
    trips = load_trips()
    print("Usable single-rider trips: %d\n" % len(trips))

    # ---- Sweep: matching window x who is allowed in the pool ----
    rows = []
    for w in WINDOWS:
        scenarios = [("open pool (everyone)", trips)]
        for s in WOMEN_SHARES:
            scenarios.append(("women-only (%d%% of riders)" % round(s * 100), trips[trips["u"] < s]))
        for name, subset in scenarios:
            r = summarize(simulate(subset, w))
            r.update({"window_min": w, "scenario": name})
            rows.append(r)
    sweep = pd.DataFrame(rows).set_index(["window_min", "scenario"])
    print("=== Pooling results (pool size %d, detour premium %d%%) ===" % (POOL_SIZE, DETOUR_PREMIUM * 100))
    print(sweep.to_string())
    sweep.to_csv("pooling_results.csv")

    # ---- Breakdown: does pooling work better in peak / rainy hours? ----
    w = HEADLINE_WINDOW
    open_f = simulate(trips, w)
    women_f = simulate(trips[trips["u"] < WOMEN_SHARES[0]], w)
    print("\n=== Breakdown at %d-minute window (open pool vs women-only %d%%) ===" %
          (w, round(WOMEN_SHARES[0] * 100)))
    rows = []
    for rainy in (0, 1):
        for peak in (0, 1):
            fo = open_f[(open_f["is_rainy"] == rainy) & (open_f["is_peak"] == peak)]
            fw = women_f[(women_f["is_rainy"] == rainy) & (women_f["is_peak"] == peak)]
            so, sw = summarize(fo), summarize(fw)
            rows.append({"weather": "rainy" if rainy else "dry",
                         "hours": "peak" if peak else "off-peak",
                         "open riders": so["riders"],
                         "open match %": so["match_%"],
                         "women-only match %": sw["match_%"],
                         "saving %": so["saving_%"]})
    print(pd.DataFrame(rows).set_index(["weather", "hours"]).to_string())

    print("\nCaveats to put in your README:")
    print("- Matching needs the SAME pickup zone and SAME dropoff zone, so this understates")
    print("  real potential (nearby zones and en-route stops would match more).")
    print("- Yellow taxis only. Most NYC ride-hail demand is Uber/Lyft, so real density is higher.")
    print("- Gender is not in the data. Women-only groups are a random subset with an ASSUMED share.")
    print("- Detour premium, window and pool size are assumptions; see the sweep for sensitivity.")
    print("- Rainy-hour groups are small (rainy hours are rare). Check the rider counts.")


if __name__ == "__main__":
    main()