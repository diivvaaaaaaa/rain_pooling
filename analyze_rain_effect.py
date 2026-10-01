"""Step 2: how much does rain change taxi demand, and can a model predict it?

Part A: compare demand in rainy hours against the usual demand for that same
        hour of day and day of week (so 6pm is compared with 6pm).
Part B: compare a time-only baseline with a model that also knows about rain,
        tested on months the model never saw.

Setup (once):   pip install scikit-learn matplotlib
Run:            python3 analyze_rain_effect.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CSV = "data/hourly_demand_weather.csv"
TEST_START = "2024-05-01"            # train on months before this, test from here on
PEAK_HOURS = [8, 9, 10, 17, 18, 19, 20, 21]
RAIN_BINS = [0, 0.5, 2.5, 7.5, 1000]
RAIN_LABELS = ["dry (<0.5mm)", "light (0.5-2.5)", "moderate (2.5-7.5)", "heavy (>7.5)"]
def load():
    df = pd.read_csv(CSV, parse_dates=["hour"]).set_index("hour")
    df["rain_prev1"] = df["rain_mm"].shift(1).fillna(0.0)
    df["rain_next1"] = df["rain_mm"].shift(-1).fillna(0.0)
    df["rain_3h"] = df["rain_mm"].rolling(3, min_periods=1).sum()
    df["rain_level"] = pd.cut(df["rain_mm"], RAIN_BINS, labels=RAIN_LABELS, right=False)
    df["is_peak"] = df["hour_of_day"].isin(PEAK_HOURS).astype(int)
    return df


def part_a_rain_effect(df):
    """Demand in each hour divided by what that hour of the week normally sees when dry."""
    dry = df[df["is_rainy"] == 0]
    usual_dry = dry.groupby(["day_of_week", "hour_of_day"])["trips"].mean().rename("usual")
    d = df.join(usual_dry, on=["day_of_week", "hour_of_day"])
    d["ratio"] = d["trips"] / d["usual"]

    print("=== Part A: demand vs. the usual for the same hour of the week ===")
    table = d.groupby("rain_level", observed=True)["ratio"].agg(["mean", "count"])
    table["change_%"] = ((table["mean"] - 1) * 100).round(1)
    print(table[["count", "change_%"]].to_string())

    print("\nSame comparison, peak hours only (8-10am, 5-9pm):")
    peak = d[d["is_peak"] == 1].groupby("rain_level", observed=True)["ratio"].agg(["mean", "count"])
    peak["change_%"] = ((peak["mean"] - 1) * 100).round(1)
    print(peak[["count", "change_%"]].to_string())
    return table


def mape(y, p):
    m = y > 0
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100)


def part_b_models(df):
    train = df[df.index < TEST_START]
    test = df[df.index >= TEST_START]

    # Baseline: average demand for that hour of day and day of week (no weather)
    base = train.groupby(["day_of_week", "hour_of_day"])["trips"].mean().rename("pred")
    base_pred = test.join(base, on=["day_of_week", "hour_of_day"])["pred"].values

    # Model: same time features plus rain features
    features = ["hour_of_day", "day_of_week", "rain_mm", "rain_prev1",
                "rain_next1", "rain_3h", "temp_c"]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(train[features], train["trips"])
    model_pred = model.predict(test[features])

    y = test["trips"].values
    rainy = (test["is_rainy"] == 1).values

    print("\n=== Part B: baseline vs. rain-aware model (tested on unseen months) ===")
    print("Train hours: %d | Test hours: %d (%d rainy)" % (len(train), len(test), rainy.sum()))
    rows = []
    for name, p in [("Baseline (time only)", base_pred), ("Model (time + rain)", model_pred)]:
        rows.append({
            "model": name,
            "MAE all hours": round(float(np.mean(np.abs(y - p))), 1),
            "MAE rainy hours": round(float(np.mean(np.abs(y[rainy] - p[rainy]))), 1),
            "MAPE all %": round(mape(y, p), 1),
            "MAPE rainy %": round(mape(y[rainy], p[rainy]), 1),
        })
    res = pd.DataFrame(rows).set_index("model")
    print(res.to_string())

    b, m = res.iloc[0], res.iloc[1]
    print("\nError reduction vs baseline:")
    print("  all hours:   %.1f%% (MAE)" % (100 * (b["MAE all hours"] - m["MAE all hours"]) / b["MAE all hours"]))
    print("  rainy hours: %.1f%% (MAE)" % (100 * (b["MAE rainy hours"] - m["MAE rainy hours"]) / b["MAE rainy hours"]))
    return res


def save_chart(table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed, skipping chart. pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(i) for i in table.index], table["change_%"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Change in taxi pickups vs usual (%)")
    ax.set_title("NYC taxi demand by rain intensity (same hour of week)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("rain_effect.png", dpi=150)
    print("\nSaved chart: rain_effect.png")


def main():
    df = load()
    table = part_a_rain_effect(df)
    part_b_models(df)
    save_chart(table)
    print("\nRemember: these are completed taxi trips, not unmet demand. Heavy rain can")
    print("lower trips if cabs are scarce or traffic is slow, so read the results with care.")


if __name__ == "__main__":
    main()
