"""Find low-cost PM sensors (Sensor.Community) located next to EEA reference stations.

Inputs (downloaded into data/raw/):
  eea_stations.csv       EEA station/sampling-point metadata (DataExtract)
  sc_sensors_24h.json    Sensor.Community sensors active in the last 24 h

Output:
  data/interim/pairs.csv           one row per (low-cost sensor, reference station) within MAX_DIST_M
  data/interim/pairs_selected.csv  pairs kept for the study (see selection rule in main)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim" / "pairs.csv"
SELECTED = ROOT / "data" / "interim" / "pairs_selected.csv"

MAX_DIST_M = 500
EARTH_RADIUS_M = 6_371_000

PM_SENSOR_TYPES = {"SDS011", "SPS30", "PMS5003", "PMS7003", "PMS1003", "PMS3003", "PMS6003", "HPM", "NextPM", "SDS021"}
HUMIDITY_SENSOR_TYPES = {"BME280", "DHT22", "SHT31", "SHT30", "SHT35", "HTU21D", "SHT85"}


def load_reference_stations() -> pd.DataFrame:
    s = pd.read_csv(RAW / "eea_stations.csv", dtype=str, low_memory=False)
    s = s[
        s["Air Pollutant"].isin(["PM10", "PM2.5"])
        & (s["Year"] == "2025")
        & (s["Measurement Type"] == "automatic")
    ]
    s = s.rename(columns={
        "Air Quality Station EoI Code": "station_code",
        "Air Quality Station Name": "station_name",
        "Air Quality Station Area": "area",
        "Air Quality Station Type": "station_type",
        "Air Pollutant": "pollutant",
        "Measurement Method": "method",
        "Country": "country",
    })
    s["lat"] = s["Latitude"].astype(float)
    s["lon"] = s["Longitude"].astype(float)
    # One row per station, listing which PM fractions it measures automatically
    grouped = (
        s.groupby("station_code")
        .agg(
            station_name=("station_name", "first"),
            country=("country", "first"),
            area=("area", "first"),
            station_type=("station_type", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            pollutants=("pollutant", lambda x: ",".join(sorted(set(x)))),
            methods=("method", lambda x: ",".join(sorted(set(x.dropna().str.rsplit("/", n=1).str[-1])))),
        )
        .reset_index()
    )
    return grouped


def load_low_cost_sensors() -> pd.DataFrame:
    rows = json.load(open(RAW / "sc_sensors_24h.json"))
    recs = []
    for r in rows:
        loc = r["location"]
        recs.append({
            "sensor_id": r["sensor"]["id"],
            "sensor_type": r["sensor"]["sensor_type"]["name"],
            "location_id": loc["id"],
            "lat": float(loc["latitude"]) if loc.get("latitude") else np.nan,
            "lon": float(loc["longitude"]) if loc.get("longitude") else np.nan,
            "indoor": int(loc.get("indoor") or 0),
            "sc_country": loc.get("country"),
        })
    df = pd.DataFrame(recs).dropna(subset=["lat", "lon"]).drop_duplicates("sensor_id")
    humid_locs = set(df.loc[df["sensor_type"].isin(HUMIDITY_SENSOR_TYPES), "location_id"])
    pm = df[df["sensor_type"].isin(PM_SENSOR_TYPES) & (df["indoor"] == 0)].copy()
    pm["has_humidity_sensor"] = pm["location_id"].isin(humid_locs)
    return pm


def main() -> None:
    stations = load_reference_stations()
    sensors = load_low_cost_sensors()

    tree = BallTree(np.radians(stations[["lat", "lon"]].to_numpy()), metric="haversine")
    idx, dist = tree.query_radius(
        np.radians(sensors[["lat", "lon"]].to_numpy()), r=MAX_DIST_M / EARTH_RADIUS_M, return_distance=True
    )

    pairs = []
    for i, (st_idx, st_dist) in enumerate(zip(idx, dist)):
        for j, d in zip(st_idx, st_dist):
            pairs.append({**sensors.iloc[i].to_dict(), **stations.iloc[j].add_prefix("ref_").to_dict(),
                          "distance_m": round(d * EARTH_RADIUS_M, 1)})
    pairs = pd.DataFrame(pairs).sort_values("distance_m")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT, index=False)

    # Selection rule (chosen 2026-11-06): traffic stations have steep street-level
    # gradients, so only accept sensors within 100 m; background/industrial within 500 m.
    traffic = pairs["ref_station_type"] == "traffic"
    selected = pairs[(traffic & (pairs["distance_m"] <= 100)) | (~traffic & (pairs["distance_m"] <= 500))]
    selected.to_csv(SELECTED, index=False)
    print(f"Selected pairs (traffic <=100 m, others <=500 m): {len(selected)} "
          f"({selected['ref_station_code'].nunique()} stations) -> {SELECTED.relative_to(ROOT)}")

    print(f"Reference stations (automatic PM, 2025): {len(stations)}")
    print(f"Outdoor low-cost PM sensors active in last 24 h: {len(sensors)}")
    for limit in (50, 100, 250, 500):
        p = pairs[pairs["distance_m"] <= limit]
        print(f"  within {limit:>3} m: {len(p):>4} sensor-station pairs, "
              f"{p['ref_station_code'].nunique():>3} stations, {p['ref_country'].nunique():>2} countries, "
              f"{p['has_humidity_sensor'].sum():>4} with humidity sensor")


if __name__ == "__main__":
    main()
