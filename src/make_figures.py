"""Make the README figures, each in a light and a dark version.

figures/<name>_light.png and figures/<name>_dark.png
  coarse_underestimation  how much PM10 the sensor sees vs how coarse the air is
  rmse_by_condition       error of each correction method on unseen areas
  dust_episode            one Saharan dust episode: reference vs raw vs corrected
  bias_vs_humidity        average error by relative humidity
  cams_check              why 2022-2023 is left out: CAMS reanalysis vs forecasts

Colours follow a colour-blind-checked categorical palette (validated light + dark);
each method keeps the same colour in every figure.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_evaluate import GROUP_KM, YEARS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
MATCHED = ROOT / "data" / "processed" / "matched.parquet"
PRED = ROOT / "results" / "predictions.parquet"
METRICS = ROOT / "results" / "metrics.csv"
CAMS_CHECK = ROOT / "results" / "cams_check.csv"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", raw="#898781", kohler="#eb6834", gbm_weather="#1baf7a",
                  gbm_no_sensor="#eda100", gbm_full="#2a78d6", single="#2a78d6", dust_fill="#898781"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", raw="#898781", kohler="#d95926", gbm_weather="#199e70",
                 gbm_no_sensor="#c98500", gbm_full="#3987e5", single="#3987e5", dust_fill="#898781"),
}
METHODS = [("raw", "Uncorrected sensor"), ("kohler", "κ-Köhler humidity"), ("gbm_weather", "ML: sensor + weather"),
           ("gbm_no_sensor", "ML: CAMS + weather, no sensor"), ("gbm_full", "ML: sensor + weather + CAMS")]


def setup(ax, t, ygrid=True):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.yaxis.label.set_color(t["ink2"])
    ax.xaxis.label.set_color(t["ink2"])
    if ygrid:
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)


def titles(fig, t, title, subtitle):
    fig.text(0.02, 0.97, title, color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.905, subtitle, color=t["ink2"], fontsize=9.5, va="top")


def save(fig, name, mode):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}_{mode}.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def coarse_underestimation(mode, m):
    t = THEMES[mode]
    d = m[(m["pm10_ref"] > 10) & (m["pm25_ref"] > 0)].dropna(subset=["pm10_lc"])
    ratio = d["pm25_ref"] / d["pm10_ref"]
    bins = [0, 0.3, 0.45, 0.6, 0.75, 0.9, 1.01]
    labels = ["≤0.3\nvery coarse", "0.3–0.45", "0.45–0.6", "0.6–0.75", "0.75–0.9", "≥0.9\nvery fine"]
    g = (d["pm10_lc"] / d["pm10_ref"]).groupby(pd.cut(ratio, bins), observed=True).median()

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=t["surface"])
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.08, right=0.97)
    setup(ax, t)
    x = np.arange(len(g))
    ax.bar(x, g.values * 100, width=0.62, color=t["single"], edgecolor=t["surface"], linewidth=2)
    ax.axhline(100, color=t["ink2"], linewidth=1)
    ax.text(-0.4, 103, "sensor = reference", color=t["ink2"], fontsize=9, ha="left", va="bottom")
    for i in (0, len(g) - 1):
        ax.text(x[i], g.values[i] * 100 + 3, f"{g.values[i]:.0%}", ha="center", va="bottom",
                color=t["ink"], fontsize=10, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    ax.set_ylabel("Sensor PM10 as % of reference (median)")
    ax.set_xlabel("Reference PM2.5 / PM10 ratio (how fine the particles are)")
    titles(fig, t, "Cheap sensors miss coarse particles",
           f"In very coarse air the sensor sees only {g.values[0]:.0%} of the true PM10. "
           f"{len(d):,} matched hours, {YEARS[0]}–{YEARS[-1]}.")
    save(fig, "coarse_underestimation", mode)


def rmse_by_condition(mode, met):
    t = THEMES[mode]
    conds = [("all", "All hours"), ("dust", "Dust"), ("smoke", "Smoke"), ("humid", "Humid (≥85 %)"),
             ("normal", "Other")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=t["surface"])
    fig.subplots_adjust(top=0.72, bottom=0.12, left=0.06, right=0.98, wspace=0.15)
    w = 0.16
    for ax, (target, name) in zip(axes, (("pm10", "PM10"), ("pm25", "PM2.5"))):
        setup(ax, t)
        sub = met[met["target"] == target].set_index(["subset", "method"])["rmse"]
        for i, (meth, label) in enumerate(METHODS):
            vals = [sub.get((c, meth), np.nan) for c, _ in conds]
            ax.bar(np.arange(len(conds)) + (i - 2) * w, vals, width=w, color=t[meth], label=label,
                   edgecolor=t["surface"], linewidth=1.5)
        ax.set_xticks(np.arange(len(conds)), [c for _, c in conds])
        ax.set_title(name, color=t["ink"], fontsize=11, loc="left", pad=8)
        ax.set_ylabel("RMSE (µg/m³)" if target == "pm10" else "")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=5, frameon=False, fontsize=9,
               labelcolor=t["ink2"], handlelength=1.2, columnspacing=1.4)
    titles(fig, t, "Error in areas the model never saw (lower is better)",
           f"5-fold cross-validation; stations within {GROUP_KM} km are held out together. {YEARS[0]}–{YEARS[-1]}. "
           "Dust and smoke defined from CAMS + the reference PM2.5/PM10 ratio.")
    save(fig, "rmse_by_condition", mode)


# Example episode: the well-documented Saharan dust event of April 2024 at Stuttgart-Bad Cannstatt,
# one of the days with the most dust hours in the data (24 of 24). Fixed so the example stays the same
# between runs; the summary statistics, not this example, carry the results.
EPISODE = dict(sensor=69437, station="DEBW013", day="2024-04-08")


def pick_dust_episode(pred):
    p = pred[(pred["target"] == "pm10")]
    sensor, station = EPISODE["sensor"], EPISODE["station"]
    day = pd.Timestamp(EPISODE["day"], tz="UTC")
    start, end = day - pd.Timedelta(days=3), day + pd.Timedelta(days=4)
    ep = p[(p["sensor_id"] == sensor) & (p["ref_station_code"] == station)
           & (p["time"] >= start) & (p["time"] < end)].sort_values("time")
    return ep, station


def dust_episode(mode, pred, m):
    t = THEMES[mode]
    ep, station = pick_dust_episode(pred)
    info = m.loc[m["ref_station_code"] == station, ["ref_station_name", "ref_country"]].iloc[0]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, facecolor=t["surface"],
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    fig.subplots_adjust(top=0.78, bottom=0.08, left=0.07, right=0.97)
    setup(ax, t)
    setup(ax2, t)
    series = [("ref", "Reference station", t["ink"], 2.2), ("pred_raw", "Uncorrected sensor", t["raw"], 1.6),
              ("pred_kohler", "κ-Köhler humidity", t["kohler"], 1.6),
              ("pred_gbm_full", "ML: sensor + weather + CAMS", t["gbm_full"], 2.0)]
    for col, label, color, lw in series:
        ax.plot(ep["time"], ep[col], color=color, linewidth=lw, label=label)
    # One selective annotation at the reference peak instead of labelling every line
    peak = ep.loc[ep["ref"].idxmax()]
    ax.annotate(f"At the peak: reference {peak['ref']:.0f}, ML {peak['pred_gbm_full']:.0f},\n"
                f"Köhler {peak['pred_kohler']:.0f}, uncorrected {peak['pred_raw']:.0f} µg/m³",
                xy=(peak["time"], peak["ref"]), xytext=(18, -6), textcoords="offset points",
                color=t["ink2"], fontsize=9, va="top")
    ax.set_ylabel("PM10 (µg/m³)")
    ax.set_ylim(bottom=0)
    ax2.fill_between(ep["time"], ep["dust"], color=t["dust_fill"], alpha=0.6, linewidth=0)
    ax2.set_ylabel("CAMS dust\n(µg/m³)")
    ax2.set_ylim(bottom=0)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.87), ncol=4, frameon=False, fontsize=9,
               labelcolor=t["ink2"])
    title_day = ep.loc[ep["dust"].idxmax(), "time"]
    titles(fig, t, f"A Saharan dust episode: {info['ref_station_name']}, {info['ref_country']} "
                   f"({title_day:%B %Y})",
           f"Hourly PM10. The model never saw this station, or any station within {GROUP_KM} km, during training.")
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save(fig, "dust_episode", mode)
    return info, title_day


def bias_vs_humidity(mode, pred):
    t = THEMES[mode]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=t["surface"], sharey=False)
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.07, right=0.98, wspace=0.18)
    edges = [0, 50, 60, 70, 80, 90, 95, 101]
    labels = ["<50", "50–60", "60–70", "70–80", "80–90", "90–95", "≥95"]
    for ax, (target, name) in zip(axes, (("pm10", "PM10"), ("pm25", "PM2.5"))):
        setup(ax, t)
        p = pred[pred["target"] == target]
        b = pd.cut(p["relative_humidity_2m"], edges, right=False, labels=labels)
        ax.axhline(0, color=t["ink2"], linewidth=1)
        for meth, label in (("raw", "Uncorrected sensor"), ("kohler", "κ-Köhler humidity"),
                            ("gbm_full", "ML: sensor + weather + CAMS")):
            err = (p[f"pred_{meth}"] - p["ref"]).groupby(b, observed=True).mean()
            ax.plot(range(len(err)), err.values, color=t[meth], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=t["surface"], markeredgewidth=1.5, label=label)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("Relative humidity (%)")
        ax.set_title(name, color=t["ink"], fontsize=11, loc="left", pad=8)
        ax.set_ylabel("Mean error, sensor − reference (µg/m³)" if target == "pm10" else "")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=3, frameon=False, fontsize=9,
               labelcolor=t["ink2"])
    titles(fig, t, "How the error changes with humidity", "Average error in unseen areas; 0 = no bias.")
    save(fig, "bias_vs_humidity", mode)


def cams_check(mode, cc):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=t["surface"])
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.07, right=0.97)
    setup(ax, t)
    x = np.arange(len(cc))
    split = int(np.argmax(cc["quarter"].str[:4].astype(int) >= YEARS[0]))
    ax.axvspan(-0.5, split - 0.5, color=t["grid"], alpha=0.6, linewidth=0)
    ax.text(split / 2 - 0.5, 1.0, "CAMS reanalysis (assimilates\nthe same EEA stations)", ha="center", va="top",
            color=t["ink2"], fontsize=9)
    ax.text((split + len(cc)) / 2 - 0.5, 1.0, "CAMS archived forecasts\n(used for all results)", ha="center",
            va="top", color=t["ink2"], fontsize=9)
    for col, label, color in (("r_pm25_cams", "CAMS PM2.5", t["gbm_no_sensor"]),
                              ("r_pm25_lc", "Low-cost sensor PM2.5", t["raw"])):
        ax.plot(x, cc[col], color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=t["surface"],
                markeredgewidth=1.5, label=label)
    ax.set_xticks(x, [q.replace("Q", " Q") if q.endswith("Q1") else q[-2:] for q in cc["quarter"]])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Correlation with reference PM2.5\n(median over stations)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=2, frameon=False, fontsize=9,
               labelcolor=t["ink2"])
    titles(fig, t, "Why 2022–2023 is left out: there, CAMS had already seen the reference stations",
           "Until 2023 a 10 km model matches each station as well as a sensor standing next to it. "
           "CAMS drops in January 2024, when Open-Meteo switches to forecasts.")
    save(fig, "cams_check", mode)


def main() -> None:
    m = pd.read_parquet(MATCHED)
    m = m[m["time"].dt.year.isin(YEARS)]
    pred = pd.read_parquet(PRED)
    met = pd.read_csv(METRICS)
    cc = pd.read_csv(CAMS_CHECK) if CAMS_CHECK.exists() else None
    for mode in THEMES:
        coarse_underestimation(mode, m)
        rmse_by_condition(mode, met)
        info, day = dust_episode(mode, pred, m)
        bias_vs_humidity(mode, pred)
        if cc is not None:
            cams_check(mode, cc)
    print(f"figures written to {FIG.relative_to(ROOT)}/; dust episode: {info['ref_station_name']}, "
          f"{info['ref_country']}, {day:%Y-%m-%d}")


if __name__ == "__main__":
    main()
