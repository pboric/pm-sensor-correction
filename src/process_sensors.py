"""Turn raw Sensor.Community daily files into clean hourly PM values.

For each sensor in data/raw/sc/<sensor_id>/:
  - read all daily files (timestamps are UTC)
  - drop readings outside the physical range (negative, or >= 999 = saturated SDS011)
  - drop days when the sensor reported a location more than RELOCATION_M away from
    its current (2026) position: the sensor was somewhere else then
  - average to hourly means, keeping only hours with at least MIN_READINGS readings

Output: data/interim/sensors_hourly.parquet with columns
  sensor_id, time (UTC, hour start), pm10_lc, pm25_lc, n_readings
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sc"
PAIRS = ROOT / "data" / "interim" / "pairs_selected.csv"
OUT = ROOT / "data" / "interim" / "sensors_hourly.parquet"

MIN_READINGS = 15      # SDS011 reports every ~2.5 min (~24/hour); require ~60 % of the hour
RELOCATION_M = 200
MAX_VALID = 999.0


def process_sensor(args: tuple[int, float, float]) -> pd.DataFrame | None:
    sensor_id, lat_now, lon_now = args
    files = sorted((RAW / str(sensor_id)).glob("*.csv.gz"))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, sep=";", usecols=["lat", "lon", "timestamp", "P1", "P2"])
        except (ValueError, pd.errors.EmptyDataError, EOFError, OSError):
            continue  # empty or truncated file
        frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["time"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    for c in ("P1", "P2", "lat", "lon"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["time"])

    # Relocation check, per day, using the median reported position
    d["day"] = d["time"].dt.floor("D")
    pos = d.groupby("day")[["lat", "lon"]].median()
    dlat = np.radians(pos["lat"] - lat_now)
    dlon = np.radians(pos["lon"] - lon_now) * np.cos(np.radians(lat_now))
    dist = 6_371_000 * np.sqrt(dlat ** 2 + dlon ** 2)
    good_days = dist.index[(dist <= RELOCATION_M) | dist.isna()]
    d = d[d["day"].isin(good_days)]

    # Physical range
    for c in ("P1", "P2"):
        d.loc[(d[c] < 0) | (d[c] >= MAX_VALID), c] = np.nan

    d["hour"] = d["time"].dt.floor("h")
    h = d.groupby("hour").agg(pm10_lc=("P1", "mean"), pm25_lc=("P2", "mean"), n_readings=("P2", "count"))
    h = h[h["n_readings"] >= MIN_READINGS].reset_index().rename(columns={"hour": "time"})
    h.insert(0, "sensor_id", sensor_id)
    h["relocated_days_dropped"] = int((dist > RELOCATION_M).sum())
    return h


def main() -> None:
    pairs = pd.read_csv(PAIRS).drop_duplicates("sensor_id")
    jobs = list(pairs[["sensor_id", "lat", "lon"]].itertuples(index=False, name=None))
    with ProcessPoolExecutor() as pool:
        parts = [p for p in pool.map(process_sensor, jobs) if p is not None and len(p)]
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT, index=False)

    per = out.groupby("sensor_id").agg(hours=("time", "size"), first=("time", "min"), last=("time", "max"),
                                       relocated=("relocated_days_dropped", "first"))
    print(f"{len(out):,} sensor-hours from {out['sensor_id'].nunique()} sensors -> {OUT.relative_to(ROOT)}")
    print(f"median hours per sensor: {per['hours'].median():.0f}; "
          f"sensors with relocation days dropped: {(per['relocated'] > 0).sum()}")


if __name__ == "__main__":
    main()
