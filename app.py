import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="ESP8266 IoT Forensics", layout="wide")
st.title("📡 ESP8266 IoT Forensics Dashboard (DHT11 + Firebase)")


# ---------------------------------------------------------
# FIREBASE CONFIG
# ---------------------------------------------------------
FIREBASE_URL = (
    "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/"
    "iot-logs.json"
)


# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def convert_ts(ts):
    """Convert Unix timestamp to Python datetime"""
    try:
        return datetime.utcfromtimestamp(int(ts))
    except Exception:
        return None


def load_data():
    """Load JSON data from Firebase and convert to DataFrame"""
    try:
        data = requests.get(FIREBASE_URL).json()
    except Exception:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    rows = []
    for key, val in data.items():

        # Skip entries that are not valid objects
        if not isinstance(val, dict):
            continue

        ts = val.get("timestamp")
        temp = val.get("temperature")
        hum = val.get("humidity")
        anomaly = val.get("anomaly", "unknown")

        if ts is None:
            continue  # skip entries without timestamp

        rows.append({
            "id": key,
            "timestamp": convert_ts(ts),
            "temperature": temp,
            "humidity": hum,
            "anomaly": anomaly
        })

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Drop rows where timestamp failed to convert
    df = df.dropna(subset=["timestamp"])

    # Sort by time
    df = df.sort_values("timestamp")

    return df


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = load_data()

if df.empty:
    st.warning("⚠ No valid log data found in Firebase.")
    st.info("➡ Wait for ESP8266 to send data.\n➡ Check Firebase path: `/iot-logs`")
    st.stop()


# ---------------------------------------------------------
# RAW LOG TABLE
# ---------------------------------------------------------
st.subheader("📁 Raw Logs")
st.dataframe(df, height=300)


# ---------------------------------------------------------
# TEMPERATURE CHART
# ---------------------------------------------------------
if "temperature" in df.columns:
    st.subheader("🌡 Temperature Over Time")
    st.line_chart(df.set_index("timestamp")["temperature"])


# ---------------------------------------------------------
# HUMIDITY CHART
# ---------------------------------------------------------
if "humidity" in df.columns:
    st.subheader("💧 Humidity Over Time")
    st.line_chart(df.set_index("timestamp")["humidity"])


# ---------------------------------------------------------
# ANOMALY TABLE
# ---------------------------------------------------------
anomaly_df = df[df["anomaly"] != "normal"]

st.subheader("🚨 Detected Anomalies")
if anomaly_df.empty:
    st.success("No anomalies detected.")
else:
    st.dataframe(anomaly_df)


# ---------------------------------------------------------
# DOWNLOAD FORENSIC REPORT
# ---------------------------------------------------------
csv = df.to_csv(index=False).encode()
st.download_button(
    "⬇ Download Forensic Report (CSV)",
    csv,
    "iot_forensics_report.csv",
    "text/csv",
)
