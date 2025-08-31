import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

@st.cache_data
def load_data():
    pred_data = "https://raw.githubusercontent.com/kszajner/GDAIR/main/prediction_log.csv"
    raw_data = "https://raw.githubusercontent.com/kszajner/GDAIR/main/raw_data.csv"
    df_pred = pd.read_csv(pred_data, parse_dates=["timestamp"])
    df_raw = pd.read_csv(raw_data, parse_dates=["date"])
    return df_pred, df_raw
df_pred, df_raw = load_data()

df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"]).dt.strftime("%Y-%m-%d")
df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")
df = pd.merge(df_pred, df_raw, left_on="timestamp", right_on="date", how="right")

# Add probability_high_pollution_t_minus_1 (risk from previous day)
df_pred = pd.read_csv("https://raw.githubusercontent.com/kszajner/GDAIR/main/prediction_log.csv", parse_dates=["timestamp"])
df_pred["date"] = pd.to_datetime(df_pred["timestamp"]).dt.strftime("%Y-%m-%d")
df_pred["probability_high_pollution_t_minus_1"] = df_pred["probability_high_pollution"].shift(1)

# Merge t and t-1 probability into df_raw by date
df_raw = pd.read_csv("https://raw.githubusercontent.com/kszajner/GDAIR/main/raw_data.csv", parse_dates=["date"])
df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")
df_merged = pd.merge(df_raw, df_pred[["date", "probability_high_pollution", "probability_high_pollution_t_minus_1"]], on="date", how="left")

# Now df_merged contains PM10, PM2.5 for day t and risk from t and t-1
df = df_merged.copy()

numeric_columns = ["avg_temperature", "avg_humidity", "avg_pressure",
                   "avg_wind_speed", "sum_precipitation", "PM10",
                   "PM2_5", "probability_high_pollution"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

st.set_page_config(layout="wide")

st.title("GDAIR Analytical & Monitoring Dashboard")

main_col1, main_col2 = st.columns([2, 2])

with main_col1:
    st.subheader("Time Series: PM10, PM2.5, and Previous Day's Risk")
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["probability_high_pollution_t_minus_1"], name="Risk (t-1) [%]", marker_color="orange", yaxis="y2", opacity=0.4))
    fig.add_trace(go.Scatter(x=df["date"], y=df["PM10"], mode="lines+markers", name="PM10 (µg/m³)", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["PM2_5"], mode="lines+markers", name="PM2.5 (µg/m³)", line=dict(color="blue")))
    fig.update_layout(
        yaxis=dict(title="PM Concentration (µg/m³)", color="black"),
        yaxis2=dict(title="Risk (t-1) [%]", overlaying="y", side="right", color="orange"),
        legend=dict(x=0.01, y=0.99),
        title="PM10, PM2.5, and Previous Day's Risk Over Time",
        xaxis_title="Date",
        bargap=0.2,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df[["PM10", "PM2_5", "probability_high_pollution_t_minus_1", "avg_temperature", "avg_humidity", "avg_pressure", "avg_wind_speed", "sum_precipitation"]].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

with main_col2:
    st.subheader("Probability of High Pollution (t) - Barplot")
    fig_prob = px.bar(df, x="date", y="probability_high_pollution",
                      labels={"date": "Date", "probability_high_pollution": "Risk (t) [%]"},
                      title="Probability of High Pollution (Current Day)", color="probability_high_pollution",
                      color_continuous_scale="Oranges")
    st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader("Monitoring Table")
    st.dataframe(df[["date", "PM10", "PM2_5", "probability_high_pollution_t_minus_1", "avg_temperature", "avg_humidity", "avg_pressure", "avg_wind_speed", "sum_precipitation"]])

st.header("Analytics & Model Monitoring")
mlops_col1, mlops_col2 = st.columns([2, 2])

with mlops_col1:
    st.subheader("Data Completeness")
    missing = df.isnull().sum()
    st.write(missing)

with mlops_col2:
    st.subheader("Model Input Distribution Monitoring")
    fig_dist = px.box(df, y=["PM10", "PM2_5", "avg_temperature", "avg_humidity", "avg_pressure", "avg_wind_speed", "sum_precipitation"],
                      title="Input Feature Distributions")
    st.plotly_chart(fig_dist, use_container_width=True)

# Download option
st.download_button("Download Analytical Data as CSV", df.to_csv(index=False), "analytical_monitoring_data.csv")

