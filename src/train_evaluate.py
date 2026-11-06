"""Train and compare correction methods for low-cost PM sensors.

Validation: grouped 5-fold cross-validation where each group is a set of stations
(and the sensors attached to them), so every prediction is made for a station the
model has never seen. This mimics deploying corrected sensors in a new city.

Methods (each fitted on the training folds only):
  raw           uncorrected sensor value
  linear        ref = a + b * lc
  linear_rh     ref = a + b * lc + c * RH + d * T
  kohler        lc divided by the kappa-Koehler hygroscopic growth factor, then linear;
                kappa chosen on the training folds
  gbm_weather   LightGBM with sensor values + weather
  gbm_full      LightGBM with sensor values + weather + CAMS dust/aerosol/CO
  gbm_no_sensor LightGBM with weather + CAMS only (no sensor): what CAMS achieves alone

Episode definitions (agreed 2026-11-06):
  dust   CAMS surface dust >= 10 ug/m3 AND reference PM2.5/PM10 <= 0.5
  smoke  CAMS CO >= 1.5 x its monthly median at the site AND reference PM2.5/PM10 >= 0.75,
         May-September only (to exclude winter wood burning)
  humid  ERA5 relative humidity >= 85 %

Outputs: results/metrics.csv, results/predictions.parquet
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "matched.parquet"
RESULTS = ROOT / "results"

WEATHER = ["relative_humidity_2m", "temperature_2m", "dew_point_2m", "boundary_layer_height",
           "wind_speed_10m", "precipitation", "cloud_cover", "surface_pressure"]
CAMS = ["dust", "aerosol_optical_depth", "carbon_monoxide", "pm10_cams", "pm25_cams"]
TIME = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
TARGETS = {"pm25": ("pm25_lc", "pm25_ref"), "pm10": ("pm10_lc", "pm10_ref")}
GBM_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def station_groups(df: pd.DataFrame) -> pd.Series:
    """Connected components of the sensor-station graph, so a sensor near two stations
    never ends up in both training and test data."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, st in df[["sensor_id", "ref_station_code"]].drop_duplicates().itertuples(index=False):
        parent[find(("s", s))] = find(("st", st))
    return df["ref_station_code"].map(lambda st: find(("st", st)))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    local_hour = (df["time"].dt.hour + df["ref_lon"] / 15) % 24  # approximate solar time
    doy = df["time"].dt.day_of_year
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * local_hour / 24), np.cos(2 * np.pi * local_hour / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["lc_ratio"] = df["pm25_lc"] / df["pm10_lc"].where(df["pm10_lc"] > 0)
    df["sensor_type"] = df["sensor_type"].astype("category")

    # Episode flags (evaluation labels only, never model inputs)
    ref_ratio = df["pm25_ref"] / df["pm10_ref"].where(df["pm10_ref"] > 0)
    month = df["time"].dt.tz_localize(None).dt.to_period("M")
    co_median = df.groupby(["cell", month])["carbon_monoxide"].transform("median")
    df["is_dust"] = (df["dust"] >= 10) & (ref_ratio <= 0.5)
    df["is_smoke"] = ((df["carbon_monoxide"] >= 1.5 * co_median) & (ref_ratio >= 0.75)
                      & df["time"].dt.month.between(5, 9))
    df["is_humid"] = df["relative_humidity_2m"] >= 85
    return df


def kohler_factor(rh: pd.Series, kappa: float) -> pd.Series:
    aw = (rh.clip(upper=99) / 100)
    return 1 + (kappa / 1.65) / (1 / aw - 1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, lc: str, ref: str) -> dict[str, np.ndarray]:
    preds = {"raw": test[lc].to_numpy()}

    lin = LinearRegression().fit(train[[lc]], train[ref])
    preds["linear"] = lin.predict(test[[lc]])

    cols = [lc, "relative_humidity_2m", "temperature_2m"]
    lrh = LinearRegression().fit(train[cols], train[ref])
    preds["linear_rh"] = lrh.predict(test[cols])

    best = None
    for kappa in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        x = (train[lc] / kohler_factor(train["relative_humidity_2m"], kappa)).to_frame()
        m = LinearRegression().fit(x, train[ref])
        rmse = np.sqrt(np.mean((m.predict(x) - train[ref]) ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, kappa, m)
    _, kappa, m = best
    preds["kohler"] = m.predict((test[lc] / kohler_factor(test["relative_humidity_2m"], kappa)).to_frame())

    base = ["pm25_lc", "pm10_lc", "lc_ratio", "sensor_type"] + TIME
    # gbm_no_sensor: same inputs minus the sensor - shows what the sensor itself adds over CAMS
    for name, feats in (("gbm_weather", base + WEATHER), ("gbm_full", base + WEATHER + CAMS),
                        ("gbm_no_sensor", TIME + WEATHER + CAMS)):
        g = lgb.LGBMRegressor(**GBM_PARAMS).fit(train[feats], train[ref])
        preds[name] = np.clip(g.predict(test[feats]), 0, None)
    preds["_kappa"] = kappa
    return preds


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    if len(y) < 50:
        return {"hours": len(y)}
    return {"hours": len(y), "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
            "rmse": np.sqrt(np.mean((y - p) ** 2)), "mae": np.mean(np.abs(y - p)),
            "bias": np.mean(p - y), "ref_mean": y.mean()}


def main() -> None:
    df = add_features(pd.read_parquet(DATA))
    df = df.dropna(subset=["relative_humidity_2m", "temperature_2m", "dust"])  # need weather + CAMS
    df["group"] = station_groups(df)
    print(f"{len(df):,} hours with weather+CAMS; {df['group'].nunique()} independent station groups; "
          f"dust hours {df['is_dust'].sum():,}, smoke hours {df['is_smoke'].sum():,}, "
          f"humid hours {df['is_humid'].sum():,}", flush=True)

    rows, pred_frames = [], []
    for target, (lc, ref) in TARGETS.items():
        d = df.dropna(subset=[lc, ref]).reset_index(drop=True)
        out = {k: np.full(len(d), np.nan) for k in
               ("raw", "linear", "linear_rh", "kohler", "gbm_weather", "gbm_full", "gbm_no_sensor")}
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(d, groups=d["group"])):
            p = fit_predict(d.iloc[tr], d.iloc[te], lc, ref)
            print(f"  {target} fold {fold + 1}/5 done (kappa={p.pop('_kappa')})", flush=True)
            for k, v in p.items():
                out[k][te] = v
        y = d[ref].to_numpy()
        for subset, mask in (("all", np.ones(len(d), bool)), ("dust", d["is_dust"].to_numpy()),
                             ("smoke", d["is_smoke"].to_numpy()), ("humid", d["is_humid"].to_numpy()),
                             ("normal", ~(d["is_dust"] | d["is_smoke"] | d["is_humid"]).to_numpy())):
            for method, p in out.items():
                rows.append({"target": target, "subset": subset, "method": method, **metrics(y[mask], p[mask])})
        pf = d[["sensor_id", "ref_station_code", "ref_country", "time", ref, "is_dust", "is_smoke", "is_humid",
                "relative_humidity_2m", "dust"]].copy()
        for k, v in out.items():
            pf[f"pred_{k}"] = v
        pf["target"] = target
        pred_frames.append(pf.rename(columns={ref: "ref"}))

    RESULTS.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.3f")
    pd.concat(pred_frames).to_parquet(RESULTS / "predictions.parquet", index=False)
    show = res.pivot_table(index=["target", "subset"], columns="method", values="rmse").round(1)
    print("\nRMSE (ug/m3) on unseen stations:\n", show[["raw", "linear", "linear_rh", "kohler", "gbm_weather",
                                                       "gbm_full", "gbm_no_sensor"]].to_string())
    print("\nR2 on unseen stations:\n", res.pivot_table(index=["target", "subset"], columns="method",
                                                       values="r2").round(2).to_string())


if __name__ == "__main__":
    main()
