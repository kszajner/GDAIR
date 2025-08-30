import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

#@st.cache_data
def load_data():
    pred_data = "https://raw.githubusercontent.com/kszajner/GDAIR/main/prediction_log.csv"
    raw_data = "https://raw.githubusercontent.com/kszajner/GDAIR/main/raw_data.csv"
    df_pred = pd.read_csv(pred_data, parse_dates=["timestamp"])
    df_raw = pd.read_csv(raw_data, parse_dates=["date"])
    return df_pred, df_raw
df_pred, df_raw = load_data()

df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"]).dt.strftime("%Y-%m-%d")
#df = pd.merge(df_pred, df_raw, left_on="timestamp", right_on="date", how="right")
print(df_raw)



