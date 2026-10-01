"""Step 3: does rain actually help the model, or was it temperature / model flexibility?

Trains the SAME gradient-boosting model on four feature sets and compares them
on the same unseen months (May-June). Only the features change.

Put this file next to analyze_rain_effect.py and run:  python3 ablation.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from analyze_rain_effect import load, TEST_START

TIME = ["hour_of_day", "day_of_week"]
RAIN = ["rain_mm", "rain_prev1", "rain_next1", "rain_3h"]
TEMP = ["temp_c"]

FEATURE_SETS = {
    "time only": TIME,
    "time + temp": TIME + TEMP,
    "time + rain": TIME + RAIN,
    "time + rain + temp": TIME + RAIN + TEMP,
}


def fit_predict(train, test, features):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    m.fit(train[features], train["trips"])
    return m.predict(test[features])


def bootstrap_diff(err_a, err_b, n=5000, seed=0):
    """Mean of (err_a - err_b) with a 95% CI. Positive = model B is better."""
    rng = np.random.default_rng(seed)
    d = err_a - err_b
    means = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    df = load()
    train = df[df.index < TEST_START]
    test = df[df.index >= TEST_START]
    y = test["trips"].values
    rainy = (test["is_rainy"] == 1).values

    errors = {}
    rows = []
    for name, feats in FEATURE_SETS.items():
        err = np.abs(y - fit_predict(train, test, feats))
        errors[name] = err
        rows.append({
            "features": name,
            "MAE all": round(float(err.mean()), 1),
            "MAE rainy": round(float(err[rainy].mean()), 1),
            "MAE dry": round(float(err[~rainy].mean()), 1),
        })

    print("=== Ablation: same model, different features (test = unseen months) ===")
    print("Test hours: %d (%d rainy)\n" % (len(test), rainy.sum()))
    print(pd.DataFrame(rows).set_index("features").to_string())

    print("\n=== Does adding rain help? (paired bootstrap, 95% CI on MAE gain) ===")
    print("Positive = rain features reduce error. If the CI includes 0, we can't claim a benefit.\n")
    for label, a, b in [("time only -> time + rain", "time only", "time + rain"),
                        ("time + temp -> time + rain + temp", "time + temp", "time + rain + temp")]:
        for subset, mask in [("all hours", np.ones_like(rainy)), ("rainy hours", rainy)]:
            mean, lo, hi = bootstrap_diff(errors[a][mask], errors[b][mask])
            print("%-34s %-12s gain %+7.1f   95%% CI [%+7.1f, %+7.1f]" % (label, subset, mean, lo, hi))

    print("\nNote: hourly errors are autocorrelated and only %d test hours are rainy," % rainy.sum())
    print("so treat these intervals as optimistic. Say so in your write-up.")


if __name__ == "__main__":
    main()