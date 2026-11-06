# Correcting low-cost PM sensors during Saharan dust, smoke and humid air

Low-cost optical particle sensors (like the SDS011 used by thousands of citizen-science
stations) are cheap enough to put everywhere, but their readings drift with humidity and
they largely miss coarse particles such as Saharan dust. This project measures how large
those errors are across Europe and tests which correction methods hold up on stations the
model has never seen, with a focus on dust episodes, wildfire smoke and humid air.

**Data:** 1.27 million matched hours from 74 Sensor.Community sensors next to 62 official
EEA stations in 16 countries, 2022–2025, with hourly weather (ERA5) and dust/aerosol fields
from the Copernicus Atmosphere Monitoring Service (CAMS).

## Key findings

1. **Cheap sensors miss coarse particles.** In very coarse air (reference PM2.5/PM10 ≤ 0.3,
   typical of dust) the sensor reports only **24 %** of the true PM10; in fine air it reports
   90–115 %.
2. **Humidity pushes the error both ways.** Uncorrected PM10 is 12 µg/m³ too low in dry air and
   5 µg/m³ too high above 95 % relative humidity. The standard κ-Köhler humidity correction fixes
   the middle of the range but not the extremes.
3. **During dust episodes, humidity corrections are not enough.** Uncorrected sensors read PM10
   65 % too low on average in dust hours. The error (RMSE) on unseen stations falls from
   31.6 µg/m³ uncorrected to 27.3 with κ-Köhler, but to **15.4 µg/m³** (−51 %) with a
   gradient-boosting model that also knows the CAMS dust forecast.
4. **The sensor still matters.** The same model without the sensor (CAMS + weather only) is
   clearly worse: PM2.5 RMSE 8.9 vs 6.6 µg/m³ overall, PM10 17.2 vs 15.4 in dust. Model and
   sensor complement each other.
5. **Smoke is a different problem.** Smoke particles are fine, so the sensor does see them, but its
   PM2.5 readings barely track the reference in smoke hours (R² 0.07). The full model brings the
   error down from 12.2 to 7.3 µg/m³ (R² 0.67), though with only 1,295 smoke hours this is the
   least certain result.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/dust_episode_dark.png">
  <img alt="Hourly PM10 at Stuttgart-Bad Cannstatt during the April 2024 Saharan dust episode: the reference peaks at 132, the uncorrected sensor at 34, the Köhler-corrected value at 42 and the machine-learning correction at 100 µg/m³." src="figures/dust_episode_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/coarse_underestimation_dark.png">
  <img alt="Bar chart: sensor PM10 as a percentage of reference PM10 rises from 24 % in very coarse air to 115 % in very fine air." src="figures/coarse_underestimation_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/rmse_by_condition_dark.png">
  <img alt="Grouped bar chart of RMSE by method for all hours, dust, smoke, humid and other hours, for PM10 and PM2.5. The model with sensor, weather and CAMS has the lowest error in every group." src="figures/rmse_by_condition_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/bias_vs_humidity_dark.png">
  <img alt="Line chart of mean error by relative humidity: the uncorrected sensor goes from strongly negative in dry air to positive above 95 %; the machine-learning correction stays close to zero." src="figures/bias_vs_humidity_light.png">
</picture>

### Results table

RMSE in µg/m³ on stations the model never saw (lower is better). All methods, including the two
linear calibrations, are in [`results/metrics.csv`](results/metrics.csv), which also has R², MAE
and bias.

| | Hours | Uncorrected | κ-Köhler | ML: sensor + weather | ML: CAMS + weather, no sensor | **ML: sensor + weather + CAMS** |
|---|---:|---:|---:|---:|---:|---:|
| **PM10**, all hours | 1,214,489 | 18.8 | 14.2 | 12.6 | 12.9 | **11.6** |
| PM10, dust | 7,322 | 31.6 | 27.3 | 19.8 | 17.2 | **15.4** |
| PM10, smoke | 1,295 | 19.1 | 14.2 | 9.7 | 11.3 | **8.5** |
| PM10, humid (≥ 85 %) | 469,456 | 22.2 | 13.4 | 11.7 | 11.9 | **10.7** |
| **PM2.5**, all hours | 1,102,171 | 8.8 | 7.8 | 7.6 | 8.9 | **6.6** |
| PM2.5, dust | 7,322 | 10.9 | 7.9 | 5.9 | 6.3 | **5.2** |
| PM2.5, smoke | 1,295 | 12.2 | 11.0 | 9.3 | 8.7 | **7.3** |
| PM2.5, humid (≥ 85 %) | 426,317 | 8.8 | 8.1 | 7.8 | 8.6 | **6.7** |

## Why it matters

Networks of cheap sensors are spreading fast, including in cities with few official stations.
Saharan dust episodes are frequent in southern and south-eastern Europe, and these are exactly the
days when uncorrected sensors under-report the most. Combining the sensor with freely available
CAMS forecasts recovers much of the missing PM10, which matters for public health warnings and for
judging when daily limit values are exceeded.

## Data

All data is open; nothing in this repository requires an account.

| Source | What | Used for |
|---|---|---|
| [Sensor.Community](https://sensor.community/) archive | Raw readings from citizen-science PM sensors (mostly SDS011) | The low-cost sensors to correct |
| [EEA Air Quality Download Service](https://eeadmz1-downloads-webapp.azurewebsites.net/) | Verified hourly PM10 and PM2.5 from official monitoring stations | Reference ("truth") |
| [Open-Meteo](https://open-meteo.com/) historical weather (ERA5) | Humidity, temperature, dew point, boundary layer height, wind, rain, clouds, pressure | Model inputs |
| [Open-Meteo](https://open-meteo.com/) air quality (Copernicus CAMS) | Surface dust, aerosol optical depth, carbon monoxide, modelled PM | Model inputs and episode labels |

**Pairing.** A low-cost sensor is paired with a reference station if it is within 100 m of a
traffic station or within 500 m of a background or industrial station. Traffic stations get
the stricter limit because air quality changes block by block along busy roads.
Sensor.Community rounds published positions to roughly 100 m, so distances are approximate.

**Cleaning.**
- Sensor readings are averaged to hourly means; hours with fewer than 15 readings, saturated
  values (≥ 999 µg/m³) and days when the sensor was more than 200 m from its current position
  are dropped.
- Only reference values the EEA marks as valid are used.
- Timestamps are converted to UTC. A cross-correlation check showed that a few stations label
  their hours differently (the best match is 1–2 h off, the same in summer and winter, so not a
  daylight-saving issue); those stations get a fixed offset.
- Pairs where the sensor does not follow the reference at all (r < 0.3 at any time shift) are
  excluded as broken or not measuring the same air.

**Episodes** (used only to evaluate, never as model inputs):
- **Dust:** CAMS surface dust ≥ 10 µg/m³ *and* reference PM2.5/PM10 ≤ 0.5 (the model and the
  measurement must agree, which filters out road dust and model false alarms)
- **Smoke:** CAMS carbon monoxide ≥ 1.5 × its monthly median at the site *and* reference
  PM2.5/PM10 ≥ 0.75, May–September only (so winter wood burning is not counted)
- **Humid:** ERA5 relative humidity ≥ 85 %

## Method

Six approaches are compared, from no correction to machine learning:

| Method | What it does |
|---|---|
| Uncorrected | The sensor's own value |
| Linear | `reference = a + b × sensor` |
| Linear + humidity | adds relative humidity and temperature |
| κ-Köhler humidity | divides the sensor value by the hygroscopic growth factor 1 + (κ/1.65)/(1/aw − 1), then a linear fit; κ is chosen on the training data |
| ML: weather | LightGBM gradient boosting with both sensor channels, their ratio, weather and time of day/year |
| ML: weather + CAMS dust | the same plus CAMS dust, aerosol optical depth, carbon monoxide and modelled PM |

**Validation.** Five-fold cross-validation grouped by station: every prediction is made for a
station (and sensor) the model never saw during training. Sensors that sit near two stations
are kept in one group so nothing leaks between training and test data. This mimics the real use
case: calibrating sensors in a new city that has no co-located reference.

## Limitations

- **Geography.** Most pairs are in Germany, Belgium, the Netherlands and Poland, where dust
  episodes are rarer and weaker than in southern Europe. No Croatian sensor is close enough to a
  reference station to be included.
- **Grid-scale weather.** Humidity and dust come from ERA5 and CAMS (roughly 10–25 km grids), not
  from the sensor's own location. Many sensors have their own humidity sensor, which was not used.
- **Reanalysis, not forecasts.** CAMS and ERA5 values here are after-the-fact analyses. A
  real-time system would use forecasts, which are somewhat less accurate.
- **Episode labels are proxies.** Dust and smoke hours are identified from CAMS plus the reference
  PM2.5/PM10 ratio; they are used only to evaluate, never as model inputs.
- **Sensor snapshot.** Pairs were found from sensors active in November 2026, so sensors retired
  earlier are missing. Nearly all sensors are the SDS011 model.
- **Time.** Validation holds out stations, not years; performance on future years was not tested
  separately.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). Downloads are about 1.5 GB and are kept in `data/`
(not committed).

```bash
uv sync
uv run python src/find_pairs.py            # downloads station + sensor lists, finds pairs
uv run python src/download_sensors.py      # Sensor.Community daily archive, ~30 min
uv run python src/download_eea.py          # EEA reference data
uv run python src/process_sensors.py
uv run python src/process_eea.py
uv run python src/check_time_alignment.py  # time offsets + broken-pair check
uv run python src/download_meteo.py        # Open-Meteo, paced to the free limits
uv run python src/build_dataset.py
uv run python src/train_evaluate.py        # results/metrics.csv
uv run python src/make_figures.py          # figures/
```

## Credits and licences

- Low-cost sensor data: [Sensor.Community](https://sensor.community/) contributors,
  [Open Data Commons DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/)
- Reference data: European Environment Agency, Air Quality e-Reporting, CC BY 4.0
- Weather and atmospheric composition: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0), based on
  ECMWF ERA5 and the Copernicus Atmosphere Monitoring Service (CAMS)
- Code: MIT licence

Author: [Petar Krešimir Borić](https://github.com/pboric)
