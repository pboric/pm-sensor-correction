"""Download daily Sensor.Community archive files for the selected low-cost PM sensors.

Files go to data/raw/sc/<sensor_id>/<date>.csv.gz. The script is resumable:
existing files are skipped, and days the archive does not have (HTTP 404) are
remembered in data/raw/sc/<sensor_id>/missing.txt so they are not requested again.

The archive is run by volunteers, so we use a small number of parallel
connections and back off on errors.
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "interim" / "pairs_selected.csv"
OUT = ROOT / "data" / "raw" / "sc"

START, END = date(2022, 1, 1), date(2025, 12, 31)
WORKERS = 6
BASE = "https://archive.sensor.community"
HEADERS = {"User-Agent": "pm-sensor-correction research script (github.com/pboric)"}

_local = threading.local()
_missing_lock = threading.Lock()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
    return _local.s


def url_for(day: date, sensor_type: str, sensor_id: int) -> str:
    d = day.isoformat()
    return f"{BASE}/{day.year}/{d}/{d}_{sensor_type.lower()}_sensor_{sensor_id}.csv.gz"


def fetch(day: date, sensor_type: str, sensor_id: int) -> str:
    folder = OUT / str(sensor_id)
    target = folder / f"{day.isoformat()}.csv.gz"
    for attempt in range(5):
        try:
            r = session().get(url_for(day, sensor_type, sensor_id), timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            tmp = target.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(target)
            return "ok"
        if r.status_code == 404:
            with _missing_lock, open(folder / "missing.txt", "a") as f:
                f.write(day.isoformat() + "\n")
            return "missing"
        time.sleep(2 ** attempt)  # 429/5xx: back off and retry
    return "failed"


def main() -> None:
    pairs = pd.read_csv(PAIRS).drop_duplicates("sensor_id")
    days = [START + timedelta(n) for n in range((END - START).days + 1)]

    jobs = []
    for s in pairs.itertuples():
        folder = OUT / str(s.sensor_id)
        folder.mkdir(parents=True, exist_ok=True)
        have = {p.name.split(".")[0] for p in folder.glob("*.csv.gz")}
        missing_file = folder / "missing.txt"
        known_missing = set(missing_file.read_text().split()) if missing_file.exists() else set()
        jobs += [(d, s.sensor_type, s.sensor_id) for d in days
                 if d.isoformat() not in have and d.isoformat() not in known_missing]

    total = len(jobs)
    print(f"{len(pairs)} sensors x {len(days)} days; {total} files still to fetch", flush=True)
    counts = {"ok": 0, "missing": 0, "failed": 0}
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(fetch, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            counts[fut.result()] += 1
            if i % 5000 == 0 or i == total:
                rate = i / (time.time() - t0)
                eta_min = (total - i) / rate / 60 if rate else 0
                print(f"{i}/{total} done ({counts}), {rate:.0f} files/s, ~{eta_min:.0f} min left", flush=True)
    print("finished:", counts, flush=True)
    sys.exit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()
