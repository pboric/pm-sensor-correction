"""Download hourly weather (ERA5-based) and CAMS dust/aerosol data from Open-Meteo.

Only for station grid cells and months where both low-cost and reference data
exist, to keep within Open-Meteo's free, non-commercial limits (about 10,000
weighted calls per day; a call covering more than two weeks counts as several).

Data: Open-Meteo (CC BY 4.0), based on ECMWF ERA5 and Copernicus CAMS.
Output: data/raw/meteo/<api>/<cell>_<start>_<end>.json
"""

import json
import math
import time
from collections import deque
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "data" / "raw" / "meteo"
CELL = 0.1  # degrees; stations in the same cell share one request

APIS = {
    "weather": ("https://archive-api.open-meteo.com/v1/archive",
                ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "wind_speed_10m",
                 "wind_direction_10m", "boundary_layer_height", "surface_pressure", "cloud_cover"]),
    "airquality": ("https://air-quality-api.open-meteo.com/v1/air-quality",
                   ["dust", "aerosol_optical_depth", "pm10", "pm2_5", "carbon_monoxide"]),
}
HOURLY_BUDGET, DAILY_BUDGET = 4_500, 9_000  # stay below the free limits (5,000 / 10,000)
HEADERS = {"User-Agent": "pm-sensor-correction research script (github.com/pboric)"}

_spent: deque = deque()  # (timestamp, weight)


def weight(start: pd.Timestamp, end: pd.Timestamp, n_vars: int) -> float:
    days = (end - start).days + 1
    return math.ceil(days / 14) * math.ceil(n_vars / 10)


def wait_for_budget(w: float) -> None:
    while True:
        now = time.time()
        while _spent and now - _spent[0][0] > 86_400:
            _spent.popleft()
        last_hour = sum(x for t, x in _spent if now - t <= 3_600)
        last_day = sum(x for _, x in _spent)
        if last_hour + w <= HOURLY_BUDGET and last_day + w <= DAILY_BUDGET:
            _spent.append((now, w))
            return
        time.sleep(60)


def needed_requests() -> pd.DataFrame:
    """One row per (cell, contiguous run of months) with overlapping sensor + reference data."""
    pairs = pd.read_csv(INTERIM / "pairs_selected.csv")
    sens = pd.read_parquet(INTERIM / "sensors_hourly.parquet", columns=["sensor_id", "time"])
    ref = pd.read_parquet(INTERIM / "reference_hourly.parquet", columns=["station_code", "time"])
    sens["month"] = sens["time"].dt.tz_localize(None).dt.to_period("M")
    ref["month"] = ref["time"].dt.tz_localize(None).dt.to_period("M")
    s_m = sens.drop_duplicates(["sensor_id", "month"]).merge(pairs[["sensor_id", "ref_station_code"]], on="sensor_id")
    r_m = ref.drop_duplicates(["station_code", "month"]).rename(columns={"station_code": "ref_station_code"})
    both = s_m.merge(r_m[["ref_station_code", "month"]], on=["ref_station_code", "month"])
    st = pairs.drop_duplicates("ref_station_code").set_index("ref_station_code")[["ref_lat", "ref_lon"]]
    both = both.join(st, on="ref_station_code")
    both["cell"] = ((both["ref_lat"] / CELL).round().astype(int).astype(str) + "_"
                    + (both["ref_lon"] / CELL).round().astype(int).astype(str))
    cells = both.groupby("cell").agg(lat=("ref_lat", "mean"), lon=("ref_lon", "mean"),
                                     months=("month", lambda m: sorted(set(m))))
    rows = []
    for cell, c in cells.iterrows():
        run = [c["months"][0]]
        for m in c["months"][1:] + [None]:
            if m is not None and m == run[-1] + 1:
                run.append(m)
                continue
            rows.append({"cell": cell, "lat": round(c["lat"], 4), "lon": round(c["lon"], 4),
                         "start": run[0].start_time, "end": run[-1].end_time.normalize()})
            if m is not None:
                run = [m]
    return pd.DataFrame(rows)


def main() -> None:
    req = needed_requests()
    req.to_csv(INTERIM / "meteo_requests.csv", index=False)
    total_w = sum(weight(r.start, r.end, len(v)) for r in req.itertuples() for _, v in APIS.values())
    print(f"{len(req)} cell/period requests per API, ~{total_w:,.0f} weighted calls in total "
          f"(~{total_w / DAILY_BUDGET:.1f} days at the free limit)", flush=True)

    done = 0
    for r in req.itertuples():
        for api, (url, variables) in APIS.items():
            target = OUT / api / f"{r.cell}_{r.start:%Y%m%d}_{r.end:%Y%m%d}.json"
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            params = {"latitude": r.lat, "longitude": r.lon, "hourly": ",".join(variables),
                      "start_date": f"{r.start:%Y-%m-%d}", "end_date": f"{r.end:%Y-%m-%d}", "timezone": "GMT"}
            w = weight(r.start, r.end, len(variables))
            for attempt in range(20):
                wait_for_budget(w)
                try:
                    resp = requests.get(url, params=params, headers=HEADERS, timeout=120)
                except requests.RequestException:
                    time.sleep(60)
                    continue
                if resp.status_code == 200:
                    target.write_text(resp.text)
                    break
                if resp.status_code == 429:
                    print("rate limited; pausing 15 min", flush=True)
                    time.sleep(900)
                    continue
                print(f"{api} {r.cell}: HTTP {resp.status_code} {resp.text[:200]}", flush=True)
                break
            done += 1
            if done % 50 == 0:
                print(f"{done}/{2 * len(req)} requests done", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
