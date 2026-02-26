
import pandas as pd
import numpy as np
import joblib
import json
import os
import holidays
import requests
from datetime import datetime

# ============================================================
# KONFIGURACJA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_model_v2.joblib")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "data/clean/feature_names.txt")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models/optimal_threshold.json")
RAW_DATA_PATH = os.path.join(BASE_DIR, "raw_data.csv")
PREDICTION_LOG_PATH = os.path.join(BASE_DIR, "prediction_log.csv")

# ============================================================
# WCZYTAJ MODEL I PROGI
# ============================================================
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, 'r', encoding='utf-8') as f:
    threshold_data = json.load(f)

THRESHOLD_HIGH = threshold_data['threshold_moderate_high']
THRESHOLD_LOW = threshold_data.get('threshold_low_moderate', 0.15)

with open(FEATURE_NAMES_PATH, 'r') as f:
    EXPECTED_FEATURES = [line.strip() for line in f if line.strip()]

# ============================================================
# WCZYTAJ DANE (ostatnie N dni dla lagów 1-7)
# ============================================================
df_raw = pd.read_csv(RAW_DATA_PATH)
df_raw = df_raw.drop(columns=['timestamp'], errors='ignore')
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values('date').reset_index(drop=True)
df_raw = df_raw.drop_duplicates(subset='date')

# Potrzebujemy ostatnie 10 dni (8 dla lagu + rolling 7)
df = df_raw.tail(10).copy().reset_index(drop=True)

# Uzupełnij ewentualne luki przez resample
df = df.set_index('date').resample('D').mean().interpolate(method='linear').reset_index()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
pl_holidays = holidays.PL()

df['Month'] = df['date'].dt.month
df['DayOfWeek'] = df['date'].dt.dayofweek
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['IsHoliday'] = df['date'].apply(lambda x: int(x in pl_holidays))
df['winter_season'] = df['Month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
df['temp_below_5'] = (df['avg_temperature'] < 5).astype(int)

# Lagi 1-7
base_cols = ['avg_temperature', 'avg_humidity', 'avg_pressure',
             'avg_wind_speed', 'sum_precipitation', 'PM10', 'PM2_5']
for col in base_cols:
    for lag in range(1, 8):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# Agregaty 7-dniowe
df['PM10_7day_avg'] = df['PM10'].rolling(7, min_periods=3).mean()
df['PM10_7day_std'] = df['PM10'].rolling(7, min_periods=3).std()
df['PM2_5_7day_avg'] = df['PM2_5'].rolling(7, min_periods=3).mean()
df['PM2_5_7day_std'] = df['PM2_5'].rolling(7, min_periods=3).std()
df['PM2_5_PM10_ratio'] = df['PM2_5'] / (df['PM10'] + 1)

# Trendy
df['PM10_trend'] = df['PM10'] - df['PM10'].shift(3)
df['PM2_5_trend'] = df['PM2_5'] - df['PM2_5'].shift(3)
df['pressure_trend_3d'] = df['avg_pressure'] - df['avg_pressure'].shift(3)
df['Humidity_diff'] = df['avg_humidity'] - df['avg_humidity'].shift(1)
df['WindSpeed_trend'] = df['avg_wind_speed'] - df['avg_wind_speed'].shift(1)

# Flagi
df['PM2_5_above_20'] = (df['PM2_5'] >= 20).astype(int)
df['PM10_above_40'] = (df['PM10'] >= 40).astype(int)

# Days since rain
df = df.reset_index(drop=True)
days_since_rain = []
counter = 0
for i in range(len(df)):
    if df.loc[i, 'sum_precipitation'] >= 1.0:
        counter = 0
    else:
        counter += 1
    days_since_rain.append(counter)
df['days_since_rain'] = days_since_rain

# Weź ostatni wiersz (dzisiejszy dzień — prognozujemy jutro)
latest = df.dropna().tail(1)

if len(latest) == 0:
    raise ValueError("Brak wystarczajacych danych do predykcji (za malo wierszy po usunieciu NaN)")

# ============================================================
# PREDYKCJA
# ============================================================
# Walidacja kolejności features
X = latest.reindex(columns=EXPECTED_FEATURES)

# Sprawdź czy są braki (wypełnij zerem jeśli jakichś kolumn brak)
missing_cols = [c for c in EXPECTED_FEATURES if c not in latest.columns]
if missing_cols:
    print(f"UWAGA: Brakujace kolumny (wypelnione 0): {missing_cols}")
    for col in missing_cols:
        X[col] = 0.0

X = X[EXPECTED_FEATURES]  # Upewnij się o kolejności

proba = model.predict_proba(X)[:, 1][0]
percent = round(proba * 100, 2)

# Interpretacja z nowym progiem
if proba >= THRESHOLD_HIGH:
    risk = "High threat of pollution"
elif proba >= THRESHOLD_LOW:
    risk = "Moderate chance of high pollution"
else:
    risk = "Low chance of high pollution"

# ============================================================
# WYNIK
# ============================================================
print(f"Probability of high pollution: {percent}% -> {risk}")
print(f"(Threshold HIGH: {THRESHOLD_HIGH:.2f}, Model: rf_model_v2.joblib)")

# Zapisz do logu
result = {
    "timestamp": datetime.now().isoformat(),
    "probability_high_pollution": percent,
    "risk_level": risk
}
result_df = pd.DataFrame([result])
result_df.to_csv(PREDICTION_LOG_PATH, mode="a", index=False,
                 header=not pd.io.common.file_exists(PREDICTION_LOG_PATH))
print("Prediction saved to prediction_log.csv")

# ============================================================
# WIADOMOSC DISCORD
# ============================================================
def send_to_discord(message, webhook_url):
    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        print(f"Failed to send message. Status code: {response.status_code}")
    else:
        print("Message sent successfully to Discord!")

latest_row = df_raw.iloc[-1]
timestamp = pd.to_datetime(latest_row["date"]).strftime("%Y-%m-%d")
temp = latest_row["avg_temperature"]
wind = latest_row["avg_wind_speed"]
pressure = latest_row["avg_pressure"]
current_pm10 = latest_row["PM10"]
current_pm25 = latest_row["PM2_5"]

message = f"**🌍 Air Quality Forecast - {timestamp}**\n"
message += f"📊 **Exceedance Probability:** {percent}% (threshold: PM10≥50 OR PM2.5≥25 μg/m³)\n"
message += f"🌡️ **Current Conditions:** {temp:.1f}°C, {wind:.1f} m/s wind, {pressure:.0f} hPa\n"
message += f"💨 **Current PM:** PM10={current_pm10:.0f} μg/m³, PM2.5={current_pm25:.1f} μg/m³\n\n"

if proba >= THRESHOLD_HIGH:
    message += f"🔴 **HIGH RISK ALERT**\n"
    message += f"Model v2 indicates {percent}% probability of exceeding thresholds tomorrow.\n"
    if wind < 2.0:
        message += f"⚠️ Low wind conditions ({wind:.1f} m/s) limiting dispersion.\n"
    if pressure > 1020:
        message += f"⚠️ High pressure system ({pressure:.0f} hPa) promoting accumulation.\n"
    if current_pm25 >= 20:
        message += f"⚠️ Elevated PM2.5 today ({current_pm25:.1f} μg/m³) — trend risk.\n"
    message += f"**Recommendation:** Limit outdoor exercise, vulnerable groups should stay indoors."
elif proba >= THRESHOLD_LOW:
    message += f"🟡 **MODERATE RISK**\n"
    message += f"Elevated probability ({percent}%) of air quality deterioration.\n"
    message += f"**Recommendation:** Monitor conditions, sensitive individuals exercise caution."
else:
    message += f"🟢 **LOW RISK**\n"
    message += f"Favorable atmospheric conditions predicted ({percent}% exceedance risk).\n"
    if wind > 4.0:
        message += f"✅ Good wind dispersion ({wind:.1f} m/s) supporting air quality.\n"
    message += f"**Outlook:** Air quality should remain within acceptable limits."

message += f"\n*Model: v2 (Logistic Regression) | Threshold HIGH: {THRESHOLD_HIGH:.2f} | Window: 7-day lags*"

webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
if webhook_url:
    send_to_discord(message, webhook_url)
else:
    print("No Discord webhook configured, skipping send.")
