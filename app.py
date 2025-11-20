import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="ESP8266 IoT Forensics", layout="wide")
st.title("📡 ESP8266 IoT Forensics Dashboard (DHT11 + Firebase)")

# 🔥 YOUR CORRECT FIREBASE PATH
FIREBASE_URL = (
    "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/"
    "forensics_logs.json"
)

def convert_ts(ts):
    try:
        return datetime.utcfromtimestamp(int(ts))
    except Exception:
        return None

def load_data():
    try:
        data = requests.get(FIREBASE_URL).json()
    except Exception:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    rows = []
    for key, val in data.items():
        if not isinstance(val, dict):
            continue

        ts = val.get("timestamp")
        if ts is None:
            continue

        rows.append({
            "id": key,
            "timestamp": convert_ts(ts),
            "temperature": val.get("temperature"),
            "humidity": val.get("humidity"),
            "anomaly": val.get("anomaly", "unknown"),
        })

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    return df

df = load_data()

if df.empty:
    st.warning("⚠ No valid log data found in Firebase yet.")
    st.stop()

st.subheader("📁 Raw Logs")
st.dataframe(df, height=300)

if "temperature" in df.columns:
    st.subheader("🌡 Temperature Over Time")
    st.line_chart(df.set_index("timestamp")["temperature"])

if "humidity" in df.columns:
    st.subheader("💧 Humidity Over Time")
    st.line_chart(df.set_index("timestamp")["humidity"])

anomaly_df = df[df["anomaly"] != "normal"]

st.subheader("🚨 Detected Anomalies")
if anomaly_df.empty:
    st.success("No anomalies detected.")
else:
    st.dataframe(anomaly_df)

csv = df.to_csv(index=False).encode()
st.download_button(
    "⬇ Download Forensic Report (CSV)",
    csv,
    "iot_forensics_report.csv",
    "text/csv",
)
