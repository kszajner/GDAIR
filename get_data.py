import requests
import pandas as pd
from datetime import datetime
sensor_ids = {
    "PM10": "4706",
    "PM2_5": "27667"
}
daily_averages = {}
for name, sensor_id in sensor_ids.items():
    url = f"https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/{sensor_id}"
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        measurements = []
        for entry in json_data['Lista danych pomiarowych']:
            if entry['Wartość'] is not None:
                measurements.append({
                    'date': entry['Data'],
                    'value': entry['Wartość']
                })
        df = pd.DataFrame(measurements)
        df['date'] = pd.to_datetime(df['date'])
        current_date = datetime.now().date()
        today_data = df[df['date'].dt.date == current_date]
        if len(today_data) > 0:
            daily_average = today_data['value'].max()
            daily_averages[name] = daily_average
        else:
            daily_averages[name] = None
    else:
        daily_averages[name] = None
print("PM sensor data:", daily_averages)
LAT, LON = 54.3523, 18.6466
def fetch_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation"
        "&timezone=Europe/Warsaw"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()['hourly']
def process_weather(data):
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    today = df[df['time'].dt.date == datetime.now().date()]
    result = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'avg_temperature': today['temperature_2m'].mean(),
        'avg_humidity': today['relative_humidity_2m'].mean(),
        'avg_pressure': today['pressure_msl'].mean(),
        'avg_wind_speed': today['wind_speed_10m'].mean(),
        'sum_precipitation': today['precipitation'].sum(),
        'timestamp': datetime.now().isoformat()
    }
    return result
def append_to_csv(record, filename="raw_data.csv"):
    df = pd.DataFrame([record])
    try:
        df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))
    except Exception as e:
        print("Error writing to CSV:", e)
weather_data = fetch_weather()
summary = process_weather(weather_data)
summary.update(daily_averages)
append_to_csv(summary)