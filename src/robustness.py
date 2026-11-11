"""Robustness checks for the main results (run after train_evaluate.py).

1. CAMS check: correlation of CAMS PM and of the sensor with the reference, per station
   and quarter, 2022-2025. For 2022-2023 Open-Meteo serves the CAMS reanalysis, which
   assimilates EEA station data, so CAMS tracks the reference much more closely than from
   2024 on, when the values are archived forecasts. The same models are also
   cross-validated on 2022-2023 alone to show how much the reanalysis would flatter the
   CAMS-based models.
2. Grouping: one group per station instead of 50 km areas.
3. Time hold-out: train on 2024, test on 2025, and the reverse (areas held out as well).
4. No shared time: the main test holds out areas, but the training folds still contain
   the same hours from other areas, and a dust episode covers many areas at once. Here
   the weeks are split at random into two halves; the model is trained on one half and
   tested on the other (areas held out as well), so it never sees the tested episode
   anywhere. A control trained on the same half it is tested on has the same amount of
   training data, so the difference between the two is the effect of shared time alone.
5. CAMS dust false alarms: hours with CAMS dust >= 10 ug/m3 where the reference air is not
   coarse (PM2.5/PM10 > 0.5). Does the model over-correct when CAMS is wrong?
6. Uncertainty: 95 % intervals for the change in RMSE, resampling whole areas
   (cluster bootstrap, 2,000 resamples), for the main result and for check 4.

Outputs: results/cams_check.csv, results/robustness.csv, results/bootstrap.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_evaluate as te  # noqa: E402

RESULTS = te.RESULTS
METHODS = ["raw", "linear", "linear_rh", "kohler", "gbm_weather", "gbm_no_sensor", "gbm_full"]


def cams_check(min_hours: int = 300) -> pd.DataFrame:
    """Median over stations of the per-station correlation with the reference, by quarter.
    Per station, so that a change in which stations report cannot move the numbers."""
    m = pd.read_parquet(te.DATA).dropna(subset=["dust"])
    m["quarter"] = m["time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    rows = []
    for q, g in m.groupby("quarter"):
        r = {"quarter": q, "hours": len(g)}
        for a, b in (("pm25_cams", "pm25_ref"), ("pm25_lc", "pm25_ref"), ("pm10_cams", "pm10_ref"),
                     ("pm10_lc", "pm10_ref")):
            per_station = [s[a].corr(s[b]) for _, s in g[["ref_station_code", a, b]].dropna()
                           .groupby("ref_station_code") if len(s) >= min_hours]
            r[f"r_{a}"] = np.median(per_station)
            r[f"stations_{a}"] = len(per_station)
        rows.append(r)
    return pd.DataFrame(rows)


def rmse_rows(pred: pd.DataFrame, check: str, subsets: dict) -> list[dict]:
    rows = []
    for target, g in pred.groupby("target"):
        for name, mask_fn in subsets.items():
            gg = g[mask_fn(g)]
            for meth in METHODS:
                e = (gg[f"pred_{meth}"] - gg["ref"]).to_numpy()
                rows.append({"check": check, "target": target, "subset": name, "method": meth, "hours": len(gg),
                             "rmse": np.sqrt(np.mean(e ** 2)) if len(gg) else np.nan,
                             "bias": e.mean() if len(gg) else np.nan})
    return rows


def week_halves(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """0 or 1 for every row: its 7-day block, assigned at random to one of two halves."""
    week = (df["time"] - df["time"].min()).dt.days // 7
    half = np.random.default_rng(seed).permutation(week.max() + 1) % 2
    return half[week.to_numpy()]


def bootstrap(pred: pd.DataFrame, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for target, g in pred.groupby("target"):
        for subset in ("all", "dust"):
            gg = g if subset == "all" else g[g["is_dust"]]
            groups, inv = np.unique(gg["group"].to_numpy(), return_inverse=True)
            cnt = np.bincount(inv, minlength=len(groups)).astype(float)
            sse = {m: np.bincount(inv, weights=(gg[f"pred_{m}"] - gg["ref"]).to_numpy() ** 2,
                                  minlength=len(groups)) for m in METHODS}
            idx = rng.integers(0, len(groups), size=(n, len(groups)))
            boot = {m: np.sqrt(sse[m][idx].sum(1) / cnt[idx].sum(1)) for m in METHODS}
            point = {m: np.sqrt(sse[m].sum() / cnt.sum()) for m in METHODS}
            for other in ("raw", "kohler", "gbm_weather", "gbm_no_sensor"):
                change = boot["gbm_full"] / boot[other] - 1
                rows.append({"target": target, "subset": subset, "areas": len(groups), "hours": len(gg),
                             "comparison": f"gbm_full vs {other}",
                             "rmse_change": point["gbm_full"] / point[other] - 1,
                             "ci_low": np.nanpercentile(change, 2.5), "ci_high": np.nanpercentile(change, 97.5)})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    cc = cams_check()
    cc.to_csv(RESULTS / "cams_check.csv", index=False, float_format="%.3f")
    print(cc.round(2).to_string(index=False), flush=True)

    standard = {"all": lambda g: np.ones(len(g), bool), "dust": lambda g: g["is_dust"],
                "smoke": lambda g: g["is_smoke"], "humid": lambda g: g["is_humid"]}
    rows = []

    print("\n1. same models on 2022-2023 (CAMS reanalysis years)", flush=True)
    old = te.load(years=(2022, 2023))
    old["group"] = te.area_groups(old)
    _, p_old = te.cross_validate(old, log=False)
    rows += rmse_rows(p_old, "reanalysis_years_2022_23", standard)

    df = te.load()
    print("2. one group per station", flush=True)
    df["station_group"] = te.area_groups(df, km=0)
    _, p_st = te.cross_validate(df, group_col="station_group", log=False)
    rows += rmse_rows(p_st, "station_groups", standard)

    print("3. train 2024, test 2025", flush=True)
    df["group"] = te.area_groups(df)
    _, p_time = te.cross_validate(df, train_rows=lambda x: x["time"].dt.year == 2024,
                                  test_rows=lambda x: x["time"].dt.year == 2025, log=False)
    rows += rmse_rows(p_time, "train_2024_test_2025", standard)
    _, p_rev = te.cross_validate(df, train_rows=lambda x: x["time"].dt.year == 2025,
                                 test_rows=lambda x: x["time"].dt.year == 2024, log=False)
    rows += rmse_rows(p_rev, "train_2025_test_2024", standard)

    print("4. no shared time: train on one half of the weeks, test on the other", flush=True)
    df["half"] = week_halves(df)
    p_other = pd.concat([te.cross_validate(df, train_rows=lambda x, h=h: x["half"] != h,
                                           test_rows=lambda x, h=h: x["half"] == h, log=False)[1] for h in (0, 1)])
    rows += rmse_rows(p_other, "other_weeks", standard)
    p_same = pd.concat([te.cross_validate(df, train_rows=lambda x, h=h: x["half"] == h,
                                          test_rows=lambda x, h=h: x["half"] == h, log=False)[1] for h in (0, 1)])
    rows += rmse_rows(p_same, "same_weeks_control", standard)

    print("5. CAMS dust false alarms (main predictions)", flush=True)
    main_pred = pd.read_parquet(RESULTS / "predictions.parquet")
    rows += rmse_rows(main_pred, "main", standard | {
        "cams_dust_any": lambda g: g["dust"] >= 10,
        "cams_dust_false_alarm": lambda g: (g["dust"] >= 10) & (g["ref_ratio"] > 0.5)})
    rob = pd.DataFrame(rows)
    rob.to_csv(RESULTS / "robustness.csv", index=False, float_format="%.3f")

    print("6. cluster bootstrap", flush=True)
    bs = pd.concat([bootstrap(main_pred).assign(check="main"), bootstrap(p_other).assign(check="other_weeks")])
    bs.insert(0, "check", bs.pop("check"))
    bs.to_csv(RESULTS / "bootstrap.csv", index=False, float_format="%.3f")

    pd.set_option("display.width", 200)
    print("\nRMSE (ug/m3):\n", rob.pivot_table(index=["check", "target", "subset"], columns="method",
                                               values="rmse")[METHODS].round(1).to_string())
    print("\nMean bias (ug/m3):\n", rob.pivot_table(index=["check", "target", "subset"], columns="method",
                                                   values="bias")[METHODS].round(1).to_string())
    bs[["rmse_change", "ci_low", "ci_high"]] = (bs[["rmse_change", "ci_low", "ci_high"]] * 100).round(0)
    print("\nChange in RMSE with the full model, % (95 % interval, areas resampled):\n", bs.to_string(index=False))


if __name__ == "__main__":
    main()
