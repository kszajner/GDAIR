
import pandas as pd
import joblib
from datetime import datetime
import holidays
df = pd.read_csv("raw_data.csv").tail(3)
def preprocess_raw(df):
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={
        "avg_wind_speed": "WindSpeed",
        "avg_temperature": "Temperature",
        "avg_humidity": "Humidity",
        "avg_pressure": "Pressure",
        "sum_precipitation": "Precipitation",
        "PM2_5": "PM2.5"
    })
    pl_holidays = holidays.Poland()
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["IsWeekend"] = df["date"].dt.weekday >= 5
    df["IsHoliday"] = df["date"].dt.date.isin(pl_holidays)
    numeric_cols = ["WindSpeed", "Temperature", "Humidity",
                    "Pressure", "Precipitation", "PM2.5", "PM10"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df
df = preprocess_raw(df)
def create_sequence_and_label(df, label=0):
    sequences = []
    i = 0
    while i < len(df):
        seq = df.iloc[i:i+3].values.flatten()
        if len(seq) == len(df.columns) * 3:
            sequences.append(seq)
        i += 3
    return pd.DataFrame(sequences).assign(label=label)
df_seq = create_sequence_and_label(df, label=0)
original_cols = df.columns.tolist()
column_names = (
    original_cols +
    [f"{col}_2" for col in original_cols] +
    [f"{col}_3" for col in original_cols] +
    ["label"]
)
df_seq.columns = column_names
def new_features(df):
    df["PM10_avg"] = df[["PM10", "PM10_2", "PM10_3"]].mean(axis=1)
    df["PM10_std"] = df[["PM10", "PM10_2", "PM10_3"]].std(axis=1)
    df["PM10_CV"] = df["PM10_std"] / df["PM10_avg"]
    df["PM2.5_avg"] = df[["PM2.5", "PM2.5_2", "PM2.5_3"]].mean(axis=1)
    df["PM2.5_std"] = df[["PM2.5", "PM2.5_2", "PM2.5_3"]].std(axis=1)
    df["PM2.5_CV"] = df["PM2.5_std"] / df["PM2.5_avg"]
    df["WindSpeed_trend"] = df["WindSpeed_3"] - df["WindSpeed"]
    df["Humidity_diff"] = df["Humidity_3"] - df["Humidity"]
    df["PM2.5_trend"] = (df["PM2.5_3"] - df["PM2.5"]) / df["PM2.5"]
    df["PM10_trend"] = (df["PM10_3"] - df["PM10"]) / df["PM10"]
    cols_to_drop = [
        "PM10", "PM10_2", "PM10_3",
        "PM2.5", "PM2.5_2", "PM2.5_3",
        "date", "date_2", "date_3",
        "timestamp", "timestamp_2", "timestamp_3"
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df
df_final = new_features(df_seq)
model = joblib.load("rf_model.joblib")
X = df_final.reindex(columns=model.feature_names_in_, fill_value=0)
proba = model.predict_proba(X)[:, 1][0]
percent = round(proba * 100, 2)
if percent < 30:
    risk = "Low chance of high pollution"
elif percent < 50:
    risk = "Moderate chance of high pollution"
else:
    risk = "High threat of pollution"
print(f"✅ Probability of high pollution: {percent}% → {risk}")
result = {
    "timestamp": datetime.now().isoformat(),
    "probability_high_pollution": percent,
    "risk_level": risk
}
result_df = pd.DataFrame([result])
result_df.to_csv("prediction_log.csv", mode="a", index=False,
                 header=not pd.io.common.file_exists("prediction_log.csv"))
print("📂 Prediction saved to prediction_log.csv")
import requests
def send_to_discord(message, webhook_url):
    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        print(f"❌ Failed to send message. Status code: {response.status_code}")
    else:
        print("✅ Message sent successfully to Discord!")
import os
webhook_url = os.getenv("DISCORD_HOOK")
latest_row = df.iloc[-1]
timestamp = latest_row["date"].strftime("%Y-%m-%d")
temp = latest_row["Temperature"]
wind = latest_row["WindSpeed"]
pressure = latest_row["Pressure"]
current_pm10 = latest_row["PM10"]
confidence_pct = percent
risk_probability = proba
message = f"**🌍 PM10 Air Quality Forecast - {timestamp}**\n"
message += f"📊 **Exceedance Probability:** {confidence_pct}% (threshold: 50 μg/m³)\n"
message += f"🌡️ **Current Conditions:** {temp:.1f}°C, {wind:.1f} m/s wind, {pressure:.0f} hPa\n"
message += f"💨 **Baseline PM10:** {current_pm10:.0f} μg/m³\n\n"
if risk_probability > 0.7:
    message += f"🔴 **HIGH RISK ALERT**\n"
    message += f"Model indicates {confidence_pct}% probability of exceeding WHO daily guidelines tomorrow.\n"
    if wind < 2.0:
        message += f"⚠️ Low wind conditions ({wind:.1f} m/s) limiting dispersion.\n"
    if pressure > 1020:
        message += f"⚠️ High pressure system ({pressure:.0f} hPa) promoting accumulation.\n"
    message += f"**Recommendation:** Limit outdoor exercise, vulnerable groups should stay indoors."
elif risk_probability > 0.4:
    message += f"🟡 **MODERATE RISK**\n"
    message += f"Elevated probability ({confidence_pct}%) of air quality deterioration.\n"
    message += f"**Recommendation:** Monitor conditions, sensitive individuals exercise caution."
else:
    message += f"🟢 **LOW RISK**\n"
    message += f"Favorable atmospheric conditions predicted ({confidence_pct}% exceedance risk).\n"
    if wind > 4.0:
        message += f"✅ Good wind dispersion ({wind:.1f} m/s) supporting air quality.\n"
    message += f"**Outlook:** Air quality should remain within acceptable limits."
message += f"\n*Model: RandomForest | Data: 72h moving window*"


if webhook_url:
    send_to_discord(message, webhook_url)
else:
    print("ℹ️ No Discord webhook configured, skipping send.")


