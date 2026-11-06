"""Join low-cost sensor hours with reference hours (and weather/dust when available).

- applies the per-station time corrections from check_time_alignment.py
- drops pairs that do not follow the reference at all (excluded_pairs.csv)
- adds Open-Meteo weather and CAMS dust/aerosol for the station's grid cell,
  if those files have been downloaded

Output: data/processed/matched.parquet, one row per (sensor, station, hour)
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
METEO = ROOT / "data" / "raw" / "meteo"
OUT = ROOT / "data" / "processed" / "matched.parquet"
CELL = 0.1


def load_meteo() -> pd.DataFrame | None:
    frames = []
    for api in ("weather", "airquality"):
        parts = []
        for f in sorted((METEO / api).glob("*.json")):
            d = json.loads(f.read_text())
            if "hourly" not in d:
                continue
            h = pd.DataFrame(d["hourly"])
            h["time"] = pd.to_datetime(h["time"]).dt.tz_localize("UTC")
            h["cell"] = f.stem.rsplit("_", 2)[0]
            parts.append(h)
        if not parts:
            return None
        df = pd.concat(parts).drop_duplicates(["cell", "time"])
        if api == "airquality":
            df = df.rename(columns={"pm10": "pm10_cams", "pm2_5": "pm25_cams"})
        frames.append(df.set_index(["cell", "time"]))
    return frames[0].join(frames[1], how="outer").reset_index()


def main() -> None:
    pairs = pd.read_csv(INTERIM / "pairs_selected.csv")
    excluded = pd.read_csv(INTERIM / "excluded_pairs.csv").rename(columns={"station": "ref_station_code"})
    pairs = pairs.merge(excluded.assign(_bad=1), on=["sensor_id", "ref_station_code"], how="left")
    pairs = pairs[pairs["_bad"].isna()].drop(columns="_bad")
    pairs["cell"] = ((pairs["ref_lat"] / CELL).round().astype(int).astype(str) + "_"
                     + (pairs["ref_lon"] / CELL).round().astype(int).astype(str))

    sens = pd.read_parquet(INTERIM / "sensors_hourly.parquet")
    ref = pd.read_parquet(INTERIM / "reference_hourly.parquet")
    offsets = pd.read_csv(INTERIM / "station_time_offsets.csv").set_index("station_code")["offset_h"]
    ref["time"] = ref["time"] + pd.to_timedelta(ref["station_code"].map(offsets).fillna(0), unit="h")

    keep = ["sensor_id", "sensor_type", "has_humidity_sensor", "distance_m", "ref_station_code", "ref_station_name",
            "ref_country", "ref_area", "ref_station_type", "ref_lat", "ref_lon", "cell"]
    m = sens.merge(pairs[keep], on="sensor_id")
    m = m.merge(ref.rename(columns={"station_code": "ref_station_code"}), on=["ref_station_code", "time"])
    m = m[m["pm25_ref"].notna() | m["pm10_ref"].notna()]

    meteo = load_meteo()
    if meteo is not None:
        m = m.merge(meteo, on=["cell", "time"], how="left")
        print(f"weather/dust attached for {m['relative_humidity_2m'].notna().mean():.0%} of hours")
    else:
        print("no weather/dust files yet; building without them")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT, index=False)
    print(f"{len(m):,} matched hours, {m['sensor_id'].nunique()} sensors, {m['ref_station_code'].nunique()} stations, "
          f"{m['ref_country'].nunique()} countries -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
