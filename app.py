# app_next_level.py
# Enhanced Streamlit OT-IoT Threat Monitoring Console
# Built as an upgraded version of the user's provided app.py

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
import matplotlib.pyplot as plt
from email.mime.image import MIMEImage
import matplotlib.dates as mdates
import zipfile
import os
import hashlib

# -----------------------------
# CONFIG / SENSITIVE ITEMS
# -----------------------------
# NOTE: It's strongly recommended to store credentials in Streamlit secrets or environment variables
# For convenience we still provide defaults but the sidebar allows you to override them at runtime.
DEFAULT_SENDER = "manas.dfis242604@nfsu.ac.in"
DEFAULT_RECEIVER = "manas.dfis242604@nfsu.ac.in"
DEFAULT_APP_PASSWORD = ""  # leave blank by default — fill via sidebar or secrets

MODEL_PATH = "saved_iforest.joblib"
SCALER_PATH = "saved_scaler.joblib"

# -----------------------------
# Email utilities (unchanged behavior preserved)
# -----------------------------
SENDER_EMAIL = DEFAULT_SENDER
APP_PASSWORD = DEFAULT_APP_PASSWORD
RECEIVER_EMAIL = DEFAULT_RECEIVER


def send_email_alert_text(subject, message):
    """Simple text email using global SENDER_EMAIL / APP_PASSWORD — kept for fallback."""
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

# Attachments-capable email (kept compatible with previous code)

def send_email_alert_with_graphs(subject, message, attachments):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        for fname, data in attachments:
            try:
                img = MIMEImage(data)
                img.add_header('Content-Disposition', 'attachment', filename=fname)
                msg.attach(img)
            except Exception as e:
                print("Failed to attach image", fname, e)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Email (with graphs) failed: {e}")
        return False

# -----------------------------
# Graph creation helper (keeps original functionality)
# -----------------------------

def create_graph_images(scored_df, df_feat, features):
    images = []
    try:
        # 1) temp/hum with anomalies
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(scored_df['ts'], scored_df['temperature'], label='Temperature', linewidth=1.5)
        ax1.set_ylabel('Temperature (°C)')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2 = ax1.twinx()
        ax2.plot(scored_df['ts'], scored_df['humidity'], label='Humidity', linewidth=1.0, linestyle='--')
        ax2.set_ylabel('Humidity (%)')

        anomalies = scored_df[scored_df['is_anomaly'] == 1]
        if not anomalies.empty:
            ax1.scatter(anomalies['ts'], anomalies['temperature'], marker='o', s=70, facecolors='none', edgecolors='r', label='Anomaly Temp')
            ax2.scatter(anomalies['ts'], anomalies['humidity'], marker='x', s=70, color='red', label='Anomaly Hum')

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        images.append(("temp_hum_with_anomalies.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("Error creating temp/hum plot:", e)

    # rolling stats & others — same as before
    try:
        win = 5
        rtemp = df_feat.set_index('ts')['temperature'].rolling(win).mean()
        rhum = df_feat.set_index('ts')['humidity'].rolling(win).mean()

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(df_feat['ts'], df_feat['temperature'], alpha=0.4, label='Temperature raw')
        ax[0].plot(rtemp.index, rtemp.values, linewidth=2, label=f'{win}-pt MA')
        ax[0].set_ylabel('Temperature (°C)')
        ax[0].legend()

        ax[1].plot(df_feat['ts'], df_feat['humidity'], alpha=0.4, label='Humidity raw')
        ax[1].plot(rhum.index, rhum.values, linewidth=2, label=f'{win}-pt MA')
        ax[1].set_ylabel('Humidity (%)')
        ax[1].legend()

        fig.autofmt_xdate()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        images.append(("rolling_stats.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("Error creating rolling stats:", e)

    try:
        if 'hour' not in scored_df.columns:
            scored_df['hour'] = scored_df['ts'].dt.hour
        byhour = scored_df.groupby('hour')['is_anomaly'].sum().reindex(range(24), fill_value=0)

        fig, ax = plt.subplots(figsize=(8,3.5))
        ax.bar(byhour.index, byhour.values)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Anomalies Count')
        ax.set_title('Anomalies by Hour (last window)')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        images.append(("anomalies_by_hour.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("Error creating anomalies by hour:", e)

    try:
        ordered = scored_df.sort_values('ts')
        ordered['cum_anom'] = ordered['is_anomaly'].cumsum()
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(ordered['ts'], ordered['cum_anom'], marker='o')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Anomalies')
        ax.set_title('Cumulative Anomalies Over Time')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        images.append(("cumulative_anomalies.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("Error creating cumulative anomalies:", e)

    # Simple OT ladder diagram
    try:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        rung_y = [9, 7, 5, 3]
        for y in rung_y:
            ax.hlines(y, 1, 9, linewidth=3, color='black')
        ax.vlines(1, 2, 10, linewidth=4)
        ax.vlines(9, 2, 10, linewidth=4)
        ax.text(2, 8.6, "I: Sensor OK", fontsize=10)
        ax.text(2, 6.6, "I: Manual Stop", fontsize=10)
        ax.text(5, 4.6, "M: Safety Interlock", fontsize=10)
        ax.text(6.5, 2.6, "Q: Alarm Output", fontsize=10, color='red', fontweight='bold')
        ax.text(4.5, 1.2, "OT Ladder (Simplified) — rungs represent logic flow", fontsize=9, ha='center')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        images.append(("ladder_logic.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("Error creating ladder logic image:", e)

    return images

# -----------------------------
# Data fetching and feature engineering (same but robust)
# -----------------------------

FIREBASE_URL = (
    "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/"
    "forensics_logs.json"
)

@st.cache_data(ttl=15)
def fetch_raw_data(url):
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("Fetch error", e)
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
        try:
            ts_dt = datetime.utcfromtimestamp(int(ts))
        except Exception:
            # accept ISO timestamps too
            try:
                ts_dt = pd.to_datetime(ts)
            except Exception:
                continue
        rows.append({
            "id": key,
            "ts": ts_dt,
            "temperature": float(v.get("temperature", np.nan)),
            "humidity": float(v.get("humidity", np.nan)),
            "anomaly_tag": v.get("anomaly", "unknown")
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("ts").reset_index(drop=True)
    return df


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

# -----------------------------
# Model train/score with persistence option and explanation helper
# -----------------------------

def train_and_score(df_feat, model_type="iforest", contamination=0.02, features=None, save_model=False):
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

    df_feat = df_feat.copy()
    df_feat["anomaly_score"] = scores
    df_feat["is_anomaly"] = labels

    if save_model and model_type == "iforest":
        try:
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
        except Exception as e:
            st.warning(f"Failed to save model: {e}")

    return df_feat, model, scaler


def load_saved_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler
        except Exception as e:
            st.warning("Could not load saved model: " + str(e))
    return None, None


def explain_anomaly(row, df_feat, features, top_n=3):
    # Use normalized z-style delta features to pick top contributors
    contributions = {}
    for f in features:
        if f in row.index:
            # fall back to feature Z if available
            contributions[f] = abs(row.get(f, 0))
    # sort by absolute value
    ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# -----------------------------
# Reporting: in-memory zip containing images + summary
# -----------------------------

def create_report_zip(scored_df, df_feat, features):
    images = create_graph_images(scored_df, df_feat, features)
    summary = []
    summary.append("OT-IoT Forensics Report")
    summary.append("Generated: " + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))
    summary.append("")

    total_events = len(scored_df)
    total_anoms = int(scored_df['is_anomaly'].sum())
    summary.append(f"Total Events: {total_events}")
    summary.append(f"Total Anomalies: {total_anoms}")
    summary.append("")

    latest_anoms = scored_df[scored_df['is_anomaly'] == 1].sort_values('ts')
    if not latest_anoms.empty:
        la = latest_anoms.iloc[-1]
        summary.append("Latest anomaly:")
        summary.append(f" - ts: {la['ts']}")
        summary.append(f" - score: {la['anomaly_score']:.4f}")
        # list feature contributors
        contribs = explain_anomaly(la, df_feat, features, top_n=5)
        for k, v in contribs:
            summary.append(f"   * {k}: {v:.4f}")

    # build zip in memory
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w') as zf:
        # add images
        for name, data in images:
            zf.writestr(name, data)
        # add summary text
        zf.writestr('report_summary.txt', '\n'.join(summary))
        # add csv of anomalies
        zf.writestr('anomalies.csv', scored_df[scored_df['is_anomaly'] == 1].to_csv(index=False))
    mem.seek(0)
    return mem

# -----------------------------
# Streamlit UI & logic
# -----------------------------

st.set_page_config(page_title="OT-IoT Threat Monitoring Console — Next Level", layout="wide")
st.title("🔎 OT-IoT Threat Monitoring Console — Next Level")
st.caption("Adds model persistence, alert cooldowns, report export, anomaly explanations, and runtime controls.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox("Model", ["iforest", "lof"], index=0)
    contamination = st.slider("Contamination (expected anomaly fraction)", 0.001, 0.2, 0.02, 0.001)
    window_hours = st.slider("Window (hours of data to analyze)", 1, 48, 12)
    save_model = st.checkbox("Save model after training", value=False)
    load_saved = st.checkbox("Try loading saved model on start", value=False)
    enable_email = st.checkbox("Enable Email Alerts", value=False)
    alert_cooldown_minutes = st.number_input("Alert cooldown (minutes)", min_value=0, max_value=1440, value=30)

    st.markdown("---")
    st.subheader("Email credentials (optional)")
    SENDER_EMAIL = st.text_input("Sender email", value=DEFAULT_SENDER)
    RECEIVER_EMAIL = st.text_input("Receiver email", value=DEFAULT_RECEIVER)
    APP_PASSWORD = st.text_input("App password (or leave blank to use secrets)", type="password")
    st.caption("Tip: for production store these in Streamlit Secrets or environment variables.")

# fetch data
FIREBASE_URL_input = FIREBASE_URL
raw = fetch_raw_data(FIREBASE_URL_input)
if raw.empty:
    st.warning("⚠ Waiting for sensor data or Firebase unreachable — check URL and network.")
    st.stop()

# windowing
start_dt = raw['ts'].max() - timedelta(hours=window_hours)
df_window = raw[raw['ts'] >= start_dt]

if df_window.empty:
    st.warning("No data in the selected window.")
    st.stop()

# feature engineer
df_feat = feature_engineer(df_window)
features = ["temperature", "humidity", "temp_diff", "hum_diff", "temp_z", "hum_z", "hour"]

# model load option
model = None
scaler = None
if load_saved:
    model, scaler = load_saved_model()

# train model
scored_df, model, scaler = train_and_score(df_feat, model_type=model_type, contamination=contamination, features=features, save_model=save_model)

# persist model objects in session for UI use
st.session_state.setdefault('model_hash', None)
if model is not None:
    # compute a light fingerprint for the model to know if it changed
    try:
        mh = hashlib.sha256(str(contamination).encode() + model.__class__.__name__.encode()).hexdigest()
        st.session_state['model_hash'] = mh
    except Exception:
        pass

# Summary metrics
st.subheader("📊 System Summary")
col1, col2, col3 = st.columns(3)
col1.metric("📡 Total Events Received", len(raw))
col2.metric("🚨 Total Anomalies Detected", int(scored_df['is_anomaly'].sum()))
col3.metric("Events (window)", len(scored_df))

# Retrain button
if st.button("🔁 Retrain Model Now"):
    scored_df, model, scaler = train_and_score(df_feat, model_type=model_type, contamination=contamination, features=features, save_model=save_model)
    st.success("Model retrained.")

# Automatic alert logic with cooldown
latest_anomaly_rows = scored_df[scored_df['is_anomaly'] == 1]
if not latest_anomaly_rows.empty:
    latest_anomaly = latest_anomaly_rows.sort_values('ts').iloc[-1]
    last_anom_id = str(latest_anomaly['ts'])

    if 'last_alert_sent_id' not in st.session_state:
        st.session_state['last_alert_sent_id'] = None
    if 'last_alert_time' not in st.session_state:
        st.session_state['last_alert_time'] = datetime.utcfromtimestamp(0)

    # determine cooldown
    cooldown = timedelta(minutes=int(alert_cooldown_minutes))
    now = datetime.utcnow()
    if st.session_state['last_alert_sent_id'] != last_anom_id and (now - st.session_state['last_alert_time']) > cooldown:
        # prepare message + explanation
        latest = raw.iloc[-1]
        subject = "🚨 AI IoT ALERT — Anomaly Detected"
        message = f"Anomaly at {latest_anomaly['ts']}\nScore: {latest_anomaly['anomaly_score']:.4f}\nTemperature: {latest_anomaly['temperature']}\nHumidity: {latest_anomaly['humidity']}"

        # explain top contributors
        explain = explain_anomaly(latest_anomaly, df_feat, features, top_n=5)
        explain_text = '\n'.join([f" - {k}: {v:.4f}" for k, v in explain])
        message += "\nTop contributions:\n" + explain_text

        # attachments
        attachments = create_graph_images(scored_df, df_feat, features)

        if enable_email and APP_PASSWORD:
            try:
                send_email_alert_with_graphs(subject, message, attachments)
                st.success("📧 Automatic alert sent (with graphs).")
            except Exception as e:
                st.error("Email send failed: " + str(e))
        else:
            st.info("Email disabled or app password missing; alert composed but not sent.\n" + message)

        st.session_state['last_alert_sent_id'] = last_anom_id
        st.session_state['last_alert_time'] = now

# Manual alert + download report
st.markdown("---")
if st.button("📤 Send Manual Alert (Preview)"):
    if not latest_anomaly_rows.empty:
        la = latest_anomaly_rows.sort_values('ts').iloc[-1]
        subject = "Manual IoT Alert"
        msg = f"Manual Trigger\nTime: {la['ts']}\nScore: {la['anomaly_score']:.4f}\nTemp: {la['temperature']}\nHum: {la['humidity']}"
        attachments = create_graph_images(scored_df, df_feat, features)
        if enable_email and APP_PASSWORD:
            send_email_alert_with_graphs(subject, msg, attachments)
            st.info("Manual alert sent (with graphs).")
        else:
            st.info("Manual alert preview (email disabled).\n" + msg)
    else:
        st.warning("No anomalies to send.")

# Downloadable report
st.markdown("### 🗂️ Export")
if st.button("Generate report (zip)"):
    zmem = create_report_zip(scored_df, df_feat, features)
    st.download_button("Download report (ZIP)", data=zmem.getvalue(), file_name=f"iot_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.zip")

# Live sensor values
latest = raw.iloc[-1]
st.subheader("📡 Live Sensor Status")
c1, c2, c3 = st.columns(3)
c1.metric("🌡 Temperature", f"{latest['temperature']:.2f} °C")
c2.metric("💧 Humidity", f"{latest['humidity']:.2f} %")
c3.metric("⏱ Last Update", latest['ts'].strftime("%Y-%m-%d %H:%M:%S"))

st.markdown("---")
# Charts (Altair + images)
st.subheader("📈 Temperature (with anomalies)")
base = alt.Chart(scored_df).encode(x="ts:T")
st.altair_chart(
    base.mark_line().encode(y="temperature:Q") +
    base.transform_filter("datum.is_anomaly == 1").mark_circle(color="red", size=70).encode(y="temperature:Q"),
    use_container_width=True
)

st.subheader("📉 Humidity (with anomalies)")
st.altair_chart(
    base.mark_line(color="green").encode(y="humidity:Q") +
    base.transform_filter("datum.is_anomaly == 1").mark_circle(color="red", size=70).encode(y="humidity:Q"),
    use_container_width=True
)

# PCA map
st.subheader("🔵 PCA Anomaly Map")
try:
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
except Exception as e:
    st.warning("Could not compute PCA visualization: " + str(e))

# Anomaly table + explanation
st.subheader("📜 Anomaly Table & Explanation")
st.dataframe(scored_df.sort_values("anomaly_score", ascending=False))

if not latest_anomaly_rows.empty:
    la = latest_anomaly_rows.sort_values('ts').iloc[-1]
    st.markdown("**Latest Anomaly — Explanation**")
    top_contribs = explain_anomaly(la, df_feat, features, top_n=5)
    for f, v in top_contribs:
        st.write(f"- {f}: {v:.4f}")

# OT visuals (images)
st.markdown("---")
st.subheader("🧭 OT Ladder & Additional Diagnostics")
try:
    images = create_graph_images(scored_df, df_feat, features)
    cols = st.columns(3)
    for idx, (fname, bdata) in enumerate(images):
        b64 = base64.b64encode(bdata).decode()
        img_md = f"data:image/png;base64,{b64}"
        with cols[idx % 3]:
            st.image(img_md, caption=fname, use_column_width=True)
except Exception as e:
    st.warning("Could not create additional visuals: " + str(e))

# Diagnostics
st.markdown("### 🔧 Diagnostics")
col1, col2, col3 = st.columns(3)
col1.metric("Anomaly Rate (window)", f"{(scored_df['is_anomaly'].mean()*100):.2f}%")
col2.metric("Latest Anomaly Score", f"{scored_df['anomaly_score'].max():.4f}")
col3.metric("Events in window", len(scored_df))

st.caption("Next-level features: model persistence, alert cooldown, per-anomaly explanation, downloadable zipped report, runtime controls for contamination/window and email creds input.")

# EOF
