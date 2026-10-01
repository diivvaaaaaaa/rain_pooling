"""Step 5: README figures + the late-night women-only question.

Reuses the simulation from pooling_sim.py (keep both files in the same folder).
Run:  python3 charts.py
Makes: pooling_window.png, pooling_hour.png, pooling_detour.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import pooling_sim as ps

WINDOWS = [2, 5, 10, 15]
HEAD = 5  # window (minutes) used for the by-hour and detour analyses


def match_by_hour(f):
    h = f["tpep_pickup_datetime"].dt.hour
    return f["pooled"].groupby(h).mean() * 100


def main():
    trips = ps.load_trips()
    scenarios = {
        "open pool": trips,
        "women-only (50%)": trips[trips["u"] < 0.5],
        "women-only (30%)": trips[trips["u"] < 0.3],
    }

    # ---- Figure 1: match rate vs matching window ----
    curves = {name: [] for name in scenarios}
    head_frames = {}
    for w in WINDOWS:
        for name, subset in scenarios.items():
            f = ps.simulate(subset, w)
            curves[name].append(f["pooled"].mean() * 100)
            if w == HEAD:
                head_frames[name] = f
        print("done window %d min" % w)

    plt.figure(figsize=(7, 4))
    for name, ys in curves.items():
        plt.plot(WINDOWS, ys, marker="o", label=name)
    plt.xlabel("Matching window (minutes)")
    plt.ylabel("Riders who find a match (%)")
    plt.title("Safety-restricted pools need a longer window to match")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pooling_window.png", dpi=150)
    plt.close()

    # ---- Figure 2: match rate by hour of day (does night hurt women-only?) ----
    open_h = match_by_hour(head_frames["open pool"])
    wom_h = match_by_hour(head_frames["women-only (50%)"])
    plt.figure(figsize=(8, 4))
    plt.plot(open_h.index, open_h.values, marker="o", label="open pool")
    plt.plot(wom_h.index, wom_h.values, marker="o", label="women-only (50%)")
    plt.xlabel("Hour of day")
    plt.ylabel("Riders who find a match (%)")
    plt.title("Match rate by hour (%d-minute window)" % HEAD)
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pooling_hour.png", dpi=150)
    plt.close()

    print("\n=== Night (22:00-05:59) vs rest of day, %d-minute window ===" % HEAD)
    rows = []
    for label, mask_fn in [("night", lambda h: (h >= 22) | (h < 6)),
                           ("rest of day", lambda h: (h >= 6) & (h < 22))]:
        row = {"period": label}
        for name in ["open pool", "women-only (50%)"]:
            f = head_frames[name]
            sub = f[mask_fn(f["tpep_pickup_datetime"].dt.hour)]
            s = ps.summarize(sub)
            row[name + " riders"] = s["riders"]
            row[name + " match %"] = s["match_%"]
        rows.append(row)
    print(pd.DataFrame(rows).set_index("period").to_string())

    # ---- Figure 3: how sensitive are the savings to the detour assumption? ----
    f = head_frames["open pool"]
    prem = [0.0, 0.10, 0.15, 0.25, 0.40]
    match, saving = [], []
    for p in prem:
        shared = f["max_fare"] * (1 + p)
        pooled = (f["n"] >= 2) & (shared < f["sum_fare"])
        match.append(pooled.mean() * 100)
        saving.append((1 - shared[pooled] / f.loc[pooled, "sum_fare"]).mean() * 100)
    print("\n=== Sensitivity to detour premium (open pool, %d-minute window) ===" % HEAD)
    print(pd.DataFrame({"detour_premium_%": [int(p * 100) for p in prem],
                        "match_%": [round(x, 1) for x in match],
                        "saving_%": [round(x, 1) for x in saving]}).to_string(index=False))

    plt.figure(figsize=(7, 4))
    x = [int(p * 100) for p in prem]
    plt.plot(x, saving, marker="o", label="avg fare saving for matched riders (%)")
    plt.plot(x, match, marker="o", label="riders matched (%)")
    plt.xlabel("Assumed detour premium (%)")
    plt.title("Savings depend on the detour assumption")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pooling_detour.png", dpi=150)
    plt.close()

    print("\nSaved: pooling_window.png, pooling_hour.png, pooling_detour.png")


if __name__ == "__main__":
    main()

