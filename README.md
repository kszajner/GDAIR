# GDAIR - Gdańsk Air Inference with Random Forest

This repository contains and powers an automated pipeline that analyzes and forecasts **air pollution risk** in Gdańsk, Poland.
It blends meteorology, pollution (PM₁₀ and PM₂.₅) and calendar effects, to estimate probability of PM exceedance in the next day using Random Forest model.

---

## Pipeline description

1. **Daily data collection (UTC→Europe/Warsaw):**
   - **Weather** (average or sum): temperature, wind speed, humidity, pressure, precipitation.
   - **Air quality** (average): PM₂.₅ and PM₁₀ from GIOŚ (Chief Inspectorate of Environmental Protection) sensors.

2. **Feature engineering (3-day window):**
   - Sequences of 3 consecutive days to capture short-term dynamics.
   - Per-variable lags: `_`, `_2`, `_3`.
   - Derived features:
     - PM statistics: `PM10_avg`, `PM10_std`, `PM10_CV`, and same for PM₂.₅
     - Trends/differences: `WindSpeed_trend`, `Humidity_diff`, `PM10_trend`, `PM2.5_trend`
   - Calendar: `Month`, `IsWeekend`, `IsHoliday` (Polish holidays).

3. **Prediction:**
   - **RandomForestClassifier** outputs probability of exceedance.
   - Risk buckets:
     - 🟢 **Low**: p < 0.30
     - 🟡 **Moderate**: 0.30 ≤ p < 0.50
     - 🔴 **High**: p ≥ 0.50

4. **Outputs & notifications:**
   - Appends a row to `prediction_log.csv` with the probability, bucket, and context.
   - Sends a **Discord** message summarizing risk and key meteorology.

---

## Data Sources

### 1) Weather — Open-Meteo API
- Endpoint: `https://api.open-meteo.com/v1/forecast`
- Location: **Gdańsk** (lat `54.3523`, lon `18.6466`)
- Requested hourly variables:
  - `temperature_2m`, `wind_speed_10m`, `relative_humidity_2m`, `pressure_msl`, `precipitation`
- Timezone: `Europe/Warsaw`
- Aggregation to daily:
  - `avg_temperature`, `avg_wind_speed`, `avg_humidity`, `avg_pressure`
  - `sum_precipitation`

### 2) Air Quality — **GIOŚ / PJP API** (Państwowy Monitoring Środowiska)
- Base: `https://api.gios.gov.pl/pjp-api/v1`
- Used endpoint for sensor measurements:
  - `/rest/data/getData/{sensorId}`
- Relevant JSON fields (Polish keys):
  - `Lista danych pomiarowych` → list of measurements
  - Each entry: `Data` (timestamp), `Wartość` (numeric value)
- Example sensor IDs used in this project:
  - `4706`, `27667` (merged/validated per day)
- Purpose:
  - Provide PM₁₀ and PM₂.₅ observations for the current day, aligned with weather aggregates.

> Note: GIOŚ values are filtered for non-null `Wartość` and restricted to the **current day** in Europe/Warsaw to compute a representative daily value.

---

## Model & Features

- **Model**: `rf_model.joblib` — `RandomForestClassifier` trained on historical data with the feature set below.
- **Core predictors** (3-day window, flattened):
  - Meteorology: `WindSpeed`, `Temperature`, `Humidity`, `Pressure`, `Precipitation` (and `_2`, `_3`)
  - Pollution: `PM10`, `PM2.5` (and `_2`, `_3`)
  - Calendar: `Month`, `IsWeekend`, `IsHoliday` (and `_2`, `_3` if used in sequences)
- **Engineered features** (examples):
  - `PM10_avg`, `PM10_std`, `PM10_CV`
  - `PM2.5_avg`, `PM2.5_std`, `PM2.5_CV`
  - `WindSpeed_trend = WindSpeed_3 - WindSpeed`
  - `Humidity_diff = Humidity_3 - Humidity`
  - `PM10_trend = (PM10_3 - PM10) / PM10` (and analogous for PM₂.₅)

**Output**: `predict_proba` returns `p(exceedance)`.  
We log the raw probability and the derived risk bucket.

---

## Data Files (schemas)

### `raw_data.csv` (daily input)
| Column              | Type    | Notes                                   |
|---------------------|---------|-----------------------------------------|
| date                | date    | Day in Europe/Warsaw                    |
| avg_temperature     | float   | °C (daily mean)                         |
| avg_humidity        | float   | % (daily mean)                          |
| avg_pressure        | float   | hPa (daily mean)                        |
| avg_wind_speed      | float   | m/s (daily mean)                        |
| sum_precipitation   | float   | mm (daily sum)                          |
| PM10                | float   | µg/m³ (from GIOŚ)                       |
| PM2_5               | float   | µg/m³ (from GIOŚ)                       |
| timestamp           | string  | ISO time when the row was appended      |

> Columns are validated/coerced to numeric where applicable; dates are normalized to Europe/Warsaw.

### `prediction_log.csv` (one row per run)
| Column                | Type   | Notes                                      |
|-----------------------|--------|--------------------------------------------|
| run_timestamp         | string | ISO timestamp of the prediction            |
| date_window_start     | date   | First day in the 3-day window              |
| date_window_end       | date   | Third day in the 3-day window              |
| pm10_exceedance_prob  | float  | 0–1 probability from the model             |
| risk_bucket           | string | `low` \| `moderate` \| `high`              |
| pm10_baseline         | float  | Latest PM₁₀ used for context (µg/m³)       |
| meteo_snapshot        | json   | {temp, wind, pressure, humidity, precip}   |
| notes                 | string | Optional free-text (e.g., diagnostics)     |

---

## Automation

A GitHub Actions workflow runs once per day at ~19:00 Warsaw and performs:

1. Fetch & aggregate **Open-Meteo** weather.
2. Fetch **GIOŚ** PM data from selected sensors and align to the day.
3. Append to `raw_data.csv` and commit.
4. Build 3-day features, load `rf_model.joblib`, and compute probability.
5. Append a row to `prediction_log.csv` and (optionally) send a **Discord** summary.

> The workflow enforces “fetch → commit → predict → commit” to ensure the model always runs on the latest committed data.

---

## Assumptions & Notes

- **Timezone**: All daily boundaries use **Europe/Warsaw**; cron scheduling uses **UTC** underneath.
- **Sensors**: GIOŚ sensor IDs may change or be temporarily unavailable; the pipeline handles missing values conservatively.
- **Validation**: Before inference, the feature matrix is checked against the model’s expected columns to prevent drift-related errors.
- **Risk thresholds**: Fixed at 30% / 50% for interpretability; can be tuned with new validation data.
- **Intended use**: Internal monitoring & research; not a public health advisory.

---

## Future Enhancements

 - To be added

---

*Operational details (secrets, tokens, deployment) are intentionally omitted. This document focuses on data, features, and predictive logic, including explicit coverage of the **GIOŚ** API used for PM₂.₅/PM₁₀ measurements.*

