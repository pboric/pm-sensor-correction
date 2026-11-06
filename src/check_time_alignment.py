"""Check that low-cost and reference timestamps line up (both should be UTC).

For each sensor-station pair, correlate hourly PM2.5 (or PM10) at time shifts of
-3..+3 hours. If the time conversion is right, the best shift is 0 for nearly all
pairs. A systematic best shift of +-1 h would point to a time zone / daylight
saving mistake.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"


def main() -> None:
    pairs = pd.read_csv(INTERIM / "pairs_selected.csv")
    sens = pd.read_parquet(INTERIM / "sensors_hourly.parquet").set_index(["sensor_id", "time"])
    ref = pd.read_parquet(INTERIM / "reference_hourly.parquet").set_index(["station_code", "time"])
    sens_ids = set(sens.index.get_level_values(0))
    ref_ids = set(ref.index.get_level_values(0))

    rows = []
    for p in pairs.itertuples():
        if p.sensor_id not in sens_ids or p.ref_station_code not in ref_ids:
            continue
        s = sens.loc[p.sensor_id]
        r = ref.loc[p.ref_station_code]
        lc, rc = ("pm25_lc", "pm25_ref") if r["pm25_ref"].notna().sum() > 1000 else ("pm10_lc", "pm10_ref")
        corr = {}
        for lag in range(-3, 4):
            shifted = r[rc].copy()
            shifted.index = shifted.index + pd.Timedelta(hours=lag)
            j = pd.concat([s[lc], shifted], axis=1, join="inner").dropna()
            if len(j) >= 500:
                corr[lag] = j.iloc[:, 0].corr(j.iloc[:, 1])
        if corr:
            best = max(corr, key=corr.get)
            rows.append({"sensor_id": p.sensor_id, "station": p.ref_station_code, "country": p.ref_country,
                         "best_lag_h": best, "r_at_0": round(corr.get(0, float("nan")), 3),
                         "r_best": round(corr[best], 3), "n_hours": len(j)})
    res = pd.DataFrame(rows)
    res.to_csv(INTERIM / "time_alignment.csv", index=False)
    print(f"{len(res)} pairs with >=500 overlapping hours")
    print("best time shift (hours) -> number of pairs:", res["best_lag_h"].value_counts().sort_index().to_dict())
    print(f"median correlation at shift 0: {res['r_at_0'].median():.2f}")
    off = res[res["best_lag_h"] != 0]
    if len(off):
        print("pairs where another shift fits better:")
        print(off.sort_values("country").to_string(index=False))


if __name__ == "__main__":
    main()
