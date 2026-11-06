"""Combine downloaded EEA files into one clean hourly reference table.

- Keep only valid values (Validity >= 1) and drop missing markers (negative values)
- Verified data (E1a) wins over unverified (E2a) for the same sampling point and hour
- Several sampling points for the same station and pollutant (e.g. a replaced
  instrument) are averaged per hour
- EEA timestamps are local standard time of the station (fixed offset, no daylight
  saving, e.g. "UTC+01" in the metadata); they are converted to UTC here. The
  offset is double-checked against the low-cost sensors in check_time_alignment.py.

Output: data/interim/reference_hourly.parquet with columns
  station_code, time (UTC, hour start), pm10_ref, pm25_ref, verified_share
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eea"
SP = ROOT / "data" / "interim" / "sampling_points.csv"
STATIONS = ROOT / "data" / "raw" / "eea_stations.csv"
OUT = ROOT / "data" / "interim" / "reference_hourly.parquet"

START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")


def utc_offset_hours(tz: str) -> int:
    m = re.fullmatch(r"UTC([+-]\d{2})(?::?(\d{2}))?", str(tz).strip())
    return int(m.group(1)) if m else 1  # nearly all our stations are UTC+01


def main() -> None:
    sp = pd.read_csv(SP)
    meta = pd.read_csv(STATIONS, dtype=str, usecols=["Air Quality Station EoI Code", "Timezone"], low_memory=False)
    tz = meta.drop_duplicates("Air Quality Station EoI Code").set_index("Air Quality Station EoI Code")["Timezone"]
    sp["offset_h"] = sp["station_code"].map(tz).map(utc_offset_hours)
    by_key = sp.set_index("key")

    frames = []
    for f in RAW.rglob("*.parquet"):
        if f.stem not in by_key.index:
            continue
        info = by_key.loc[f.stem]
        d = pd.read_parquet(f, columns=["Start", "Value", "Validity", "Verification", "AggType"])
        d = d[(d["AggType"] == "hour") & (d["Validity"] >= 1) & (d["Value"] >= 0)]
        if d.empty:
            continue
        local = pd.to_datetime(d["Start"])
        d = pd.DataFrame({
            "time": (local - pd.Timedelta(hours=int(info["offset_h"]))).dt.tz_localize("UTC"),
            "value": d["Value"].astype(float),
            "verified": f.parent.parent.name.startswith("E1a"),
            "station_code": info["station_code"],
            "pollutant": info["pollutant"],
            "sampling_point": f.stem,
        })
        frames.append(d[(d["time"] >= START) & (d["time"] < END)])

    d = pd.concat(frames, ignore_index=True)
    # Verified beats unverified for the same sampling point and hour
    d = d.sort_values("verified", ascending=False).drop_duplicates(["sampling_point", "time"])
    # Average co-located instruments per station, pollutant and hour
    agg = d.groupby(["station_code", "time", "pollutant"]).agg(value=("value", "mean"), verified=("verified", "mean"))
    wide = agg["value"].unstack("pollutant").rename(columns={"PM10": "pm10_ref", "PM2.5": "pm25_ref"})
    wide["verified_share"] = agg["verified"].groupby(["station_code", "time"]).mean()
    wide = wide.reset_index()
    wide.to_parquet(OUT, index=False)

    print(f"{len(wide):,} station-hours at {wide['station_code'].nunique()} stations -> {OUT.relative_to(ROOT)}")
    print(f"PM10 hours: {wide['pm10_ref'].notna().sum():,}; PM2.5 hours: {wide['pm25_ref'].notna().sum():,}; "
          f"verified share: {wide['verified_share'].mean():.0%}")
    print("hours by year:", wide.groupby(wide["time"].dt.year).size().to_dict())


if __name__ == "__main__":
    main()
