# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import altair as alt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import base64
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ============================================================
# EMAIL ALERT — HARDCODED CREDENTIALS
# ============================================================
SENDER_EMAIL = "manas.dfis242604@nfsu.ac.in"
APP_PASSWORD = "euozfdlazplbmtkd"              # Google App Password
RECEIVER_EMAIL = "manas.dfis242604@nfsu.ac.in"


# ============================================================
# EMAIL SENDING FUNCTION
# ============================================================
def send_email_alert(subject, message):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        return True

    except Exception as e:
        st.error(f"❌ Email failed: {e}")
        return False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AI IoT Forensics", layout="wide")
st.title("🔎 IoT Forensics — AI Anomaly Detection")
st.caption("ESP8266 + Firebase + AI + Streamlit + Email Alerts")


# ============================================================
# FIREBASE URL
# ============================================================
FIREBASE_URL = (
    "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/"
    "forensics_logs.json"
)


# ============================================================
# FETCH DATA FROM FIREBASE
# ============================================================
@st.cache_data(ttl=10)
def fetch_raw_data(url):
    try:
        data = requests.get(url).json()
    except:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    rows = []
    for key, v in data.items():
        if not isinstance(v, dict):
            continue

        ts = v.get("timestamp")
        if ts is None:
            continue

        rows.append({
            "id": key,
            "ts": datetime.utcfromtimestamp(int(ts)),
            "temperature": float(v.get("temperature", np.nan)),
            "humidity": float(v.get("humidity", np.nan)),
            "anomaly_tag": v.get("anomaly", "unknown")
        })

    df = pd.DataFrame(rows)
    return df.sort_values("ts")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineer(df, window=5):
    tmp = df.copy().set_index("ts")
    tmp["temperature"] = tmp["temperature"].interpolate().ffill().bfill()
    tmp["humidity"] = tmp["humidity"].interpolate().ffill().bfill()

    tmp["temp_diff"] = tmp["temperature"].diff().fillna(0)
    tmp["hum_diff"] = tmp["humidity"].diff().fillna(0)

    tmp["temp_ma"] = tmp["temperature"].rolling(window, 1).mean()
    tmp["hum_ma"] = tmp["humidity"].rolling(window, 1).mean()
    tmp["temp_std"] = tmp["temperature"].rolling(window, 1).std().fillna(0)
    tmp["hum_std"] = tmp["humidity"].rolling(window, 1).std().fillna(0)

    tmp["temp_z"] = (tmp["temperature"] - tmp["temp_ma"]) / tmp["temp_std"].replace(0, 1)
    tmp["hum_z"] = (tmp["humidity"] - tmp["hum_ma"]) / tmp["hum_std"].replace(0, 1)

    tmp["hour"] = tmp.index.hour

    return tmp.reset_index()


# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================
def train_and_score(df_feat, model_type="iforest", contamination=0.02, features=None):
    X = df_feat[features].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if model_type == "iforest":
        model = IsolationForest(contamination=contamination, random_state=42)
    else:
        model = LocalOutlierFactor(novelty=True, contamination=contamination)

    model.fit(Xs)
    scores = -model.decision_function(Xs)
    labels = (model.predict(Xs) == -1).astype(int)

    df_feat["anomaly_score"] = scores
    df_feat["is_anomaly"] = labels
    return df_feat, model, scaler


# ============================================================
# LOAD LIVE DATA
# ============================================================
df_raw = fetch_raw_data(FIREBASE_URL)

if df_raw.empty:
    st.warning("⚠ Waiting for sensor data...")
    st.stop()

# last 12 hours window
start_dt = df_raw["ts"].max() - timedelta(hours=12)
df_window = df_raw[df_raw["ts"] >= start_dt]

df_feat = feature_engineer(df_window)
features = ["temperature", "humidity", "temp_diff", "hum_diff", "temp_z", "hum_z", "hour"]

# ============================================================
# RETRAIN BUTTON
# ============================================================
if st.button("🔁 Retrain Model Live"):
    st.cache_resource.clear()
    st.success("Model retrained using latest data!")


# ============================================================
# TRAIN MODEL
# ============================================================
scored_df, model, scaler = train_and_score(df_feat, "iforest", 0.02, features)


# ============================================================
# TOTAL COUNTS — NEW FEATURE
# ============================================================
total_events = len(df_raw)
total_anomalies = scored_df["is_anomaly"].sum()

st.subheader("📊 System Summary")
m1, m2 = st.columns(2)

m1.metric("📡 Total Events Received", total_events)
m2.metric("🚨 Total Anomalies Detected", int(total_anomalies))


# ============================================================
# AUTOMATIC EMAIL ALERT (for newest anomaly)
# ============================================================
latest_anomaly_rows = scored_df[scored_df["is_anomaly"] == 1]

if not latest_anomaly_rows.empty:
    latest_anomaly = latest_anomaly_rows.iloc[-1]
    last_anom_id = str(latest_anomaly["ts"])

    if "last_alert_sent_id" not in st.session_state:
        st.session_state["last_alert_sent_id"] = None

    if st.session_state["last_alert_sent_id"] != last_anom_id:
        latest = df_raw.iloc[-1]

        subject = "🚨 AI IoT ALERT — Anomaly Detected"

        message = f"""
===============================
🔴 **AI IoT Forensics Alert**
===============================

📌 **Anomaly Detected**
Timestamp : {latest_anomaly['ts']}
Anomaly Score : {latest_anomaly['anomaly_score']:.4f}

📡 **Sensor Values at Anomaly**
Temperature : {latest_anomaly['temperature']} °C
Humidity    : {latest_anomaly['humidity']} %

📡 **Current Live Sensor Values**
Live Temperature : {latest['temperature']} °C
Live Humidity    : {latest['humidity']} %

Please investigate the IoT device immediately.
"""

        send_email_alert(subject, message)
        st.session_state["last_alert_sent_id"] = last_anom_id

        st.success("📧 Automatic alert sent for latest anomaly!")


# ============================================================
# MANUAL ALERT BUTTON
# ============================================================
if st.button("📤 Send Manual Alert"):
    if not latest_anomaly_rows.empty:
        latest = df_raw.iloc[-1]
        msg = f"""
Manual IoT Alert Triggered

Latest Anomaly:
Timestamp : {latest_anomaly['ts']}
Temperature : {latest_anomaly['temperature']}
Humidity : {latest_anomaly['humidity']}
Anomaly Score : {latest_anomaly['anomaly_score']}

Current Live:
Temperature : {latest['temperature']}
Humidity : {latest['humidity']}
"""
        send_email_alert("Manual IoT Alert", msg)
        st.info("📨 Manual alert sent.")
    else:
        st.warning("No anomalies to alert.")


# ============================================================
# LIVE SENSOR VALUES
# ============================================================
latest = df_raw.iloc[-1]

st.subheader("📡 Live Sensor Status")
c1, c2, c3 = st.columns(3)

c1.metric("🌡 Temperature", f"{latest['temperature']:.2f} °C")
c2.metric("💧 Humidity", f"{latest['humidity']:.2f} %")
c3.metric("⏱ Last Update", latest["ts"].strftime("%Y-%m-%d %H:%M:%S"))

st.markdown("---")


# ============================================================
# CHARTS — Temperature + Humidity with Anomalies
# ============================================================
st.subheader("📈 Temperature (with anomalies)")
base = alt.Chart(scored_df).encode(x="ts:T")

st.altair_chart(
    base.mark_line().encode(y="temperature:Q") +
    base.transform_filter("datum.is_anomaly == 1")
    .mark_circle(color="red", size=70).encode(y="temperature:Q"),
    use_container_width=True
)

st.subheader("📉 Humidity (with anomalies)")
st.altair_chart(
    base.mark_line(color="green").encode(y="humidity:Q") +
    base.transform_filter("datum.is_anomaly == 1")
    .mark_circle(color="red", size=70).encode(y="humidity:Q"),
    use_container_width=True
)


# ============================================================
# PCA ANOMALY MAP
# ============================================================
st.subheader("🔵 PCA Anomaly Map")
X_vis = scaler.transform(df_feat[features].fillna(0))
p = PCA(n_components=2).fit_transform(X_vis)
p_df = pd.DataFrame({"pc1": p[:, 0], "pc2": p[:, 1], "is_anomaly": scored_df["is_anomaly"]})

st.altair_chart(
    alt.Chart(p_df).mark_circle(size=60).encode(
        x="pc1:Q", y="pc2:Q",
        color=alt.condition("datum.is_anomaly==1", alt.value("red"), alt.value("blue"))
    ).interactive(),
    use_container_width=True
)


# ============================================================
# ANOMALY TABLE
# ============================================================
st.subheader("📜 Anomaly Table")
st.dataframe(scored_df.sort_values("anomaly_score", ascending=False))
