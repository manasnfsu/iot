import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ================================
# CONFIGURE FIREBASE
# ================================
FIREBASE_URL = "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/iot-logs.json"

# ================================
# HELPER FUNCTIONS
# ================================
def convert_ts(ts):
    """Convert unix timestamp to readable datetime"""
    return datetime.utcfromtimestamp(int(ts))

def load_data():
    data = requests.get(FIREBASE_URL).json()
    if data is None:
        return pd.DataFrame()

    rows = []
    for key, val in data.items():
        rows.append({
            "id": key,
            "timestamp": convert_ts(val.get("timestamp", 0)),
            "temperature": val.get("temperature"),
            "humidity": val.get("humidity"),
            "anomaly": val.get("anomaly", "unknown")
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp")
    return df

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="ESP8266 IoT Forensics", layout="wide")
st.title("📡 ESP8266 IoT Forensics Dashboard")
st.write("Realtime DHT11 + Anomaly Monitoring")

df = load_data()

# Show Raw Logs
st.subheader("📁 Raw Logs from Firebase")
st.dataframe(df, height=300)

# Temperature Chart
st.subheader("🌡 Temperature Over Time")
st.line_chart(df.set_index("timestamp")["temperature"])

# Humidity Chart
st.subheader("💧 Humidity Over Time")
st.line_chart(df.set_index("timestamp")["humidity"])

# Anomaly Table
st.subheader("🚨 Detected Anomalies")
st.dataframe(df[df["anomaly"] != "normal"])

# Download button
csv = df.to_csv(index=False).encode()
st.download_button(
    "Download Forensic Report (CSV)",
    csv,
    "iot_forensics_report.csv",
    "text/csv",
)
