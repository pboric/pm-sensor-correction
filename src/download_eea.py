"""Download hourly EEA reference PM10/PM2.5 data for the stations in the selected pairs.

Uses the EEA Air Quality Download Service API: ask for the list of Parquet file
URLs per country and pollutant, then fetch only the files belonging to our
stations' sampling points. Verified data (E1a, dataset 2) is preferred; the
unverified up-to-date stream (E2a, dataset 1) is fetched as well so that recent
months not yet verified can be filled in later.

Output: data/raw/eea/<dataset>/<country>/<sampling point>.parquet
"""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "interim" / "pairs_selected.csv"
STATIONS = ROOT / "data" / "raw" / "eea_stations.csv"
OUT = ROOT / "data" / "raw" / "eea"

API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/urls"
DATASETS = {2: "E1a_verified", 1: "E2a_unverified"}
POLLUTANTS = ["PM10", "PM2.5"]


def file_key(sampling_point_id: str) -> str:
    """EEA file names are the sampling point id with unsafe characters replaced by '_'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", sampling_point_id)


def our_sampling_points() -> pd.DataFrame:
    s = pd.read_csv(STATIONS, dtype=str, low_memory=False)
    codes = set(pd.read_csv(PAIRS)["ref_station_code"])
    sp = s[
        s["Air Quality Station EoI Code"].isin(codes)
        & s["Air Pollutant"].isin(POLLUTANTS)
        & (s["Measurement Type"] == "automatic")
    ][["Air Quality Station EoI Code", "Sampling Point Id", "Air Pollutant", "Measurement Method",
       "Measurement Equipment", "Year"]].drop_duplicates("Sampling Point Id")
    sp.columns = ["station_code", "sampling_point", "pollutant", "method", "equipment", "meta_year"]
    sp["country"] = sp["station_code"].str[:2]
    sp["key"] = sp["sampling_point"].map(file_key)
    return sp


def list_urls(country: str, pollutant: str, dataset: int) -> list[str]:
    body = {"countries": [country], "cities": [], "pollutants": [pollutant], "dataset": dataset,
            "source": "API", "dateTimeStart": "2022-01-01T00:00:00Z",
            "dateTimeEnd": "2025-12-31T23:59:59Z", "aggregationType": "hour"}
    r = requests.post(API, json=body, timeout=300)
    r.raise_for_status()
    return [u.strip() for u in r.text.lstrip("﻿").splitlines()[1:] if u.strip()]


def fetch(url: str, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    tmp = target.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(target)


def main() -> None:
    sp = our_sampling_points()
    sp.to_csv(ROOT / "data" / "interim" / "sampling_points.csv", index=False)
    wanted = set(sp["key"])
    print(f"{len(sp)} sampling points at {sp['station_code'].nunique()} stations in {sp['country'].nunique()} countries")

    for dataset, label in DATASETS.items():
        jobs = []
        for country in sorted(sp["country"].unique()):
            for pol in POLLUTANTS:
                try:
                    urls = list_urls(country, pol, dataset)
                except requests.RequestException as e:
                    print(f"  {label} {country} {pol}: listing failed ({e})")
                    continue
                for u in urls:
                    name = u.rsplit("/", 1)[-1].removesuffix(".parquet")
                    if name in wanted:
                        jobs.append((u, OUT / label / country / f"{name}.parquet"))
        print(f"{label}: {len(jobs)} matching files", flush=True)
        with ThreadPoolExecutor(6) as pool:
            list(pool.map(lambda j: fetch(*j), jobs))

    got = {p.stem for p in OUT.rglob("*.parquet")}
    missing = sp[~sp["key"].isin(got)]
    print(f"done: {len(got)} sampling points downloaded; {len(missing)} with no file "
          f"(often closed or replaced instruments)")


if __name__ == "__main__":
    main()
