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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import matplotlib.pyplot as plt
from email.mime.image import MIMEImage
import matplotlib.dates as mdates
import re


# ============================================================
# EMAIL CONFIG
# ============================================================
SENDER_EMAIL = "manas.dfis242604@nfsu.ac.in"
APP_PASSWORD = "euozfdlazplbmtkd"
RECEIVER_EMAIL = "manas.dfis242604@nfsu.ac.in"


# ============================================================
# NORMALIZE EMAIL LIST
# ============================================================
def _normalize_recipient_input(receiver_emails):
    if receiver_emails is None:
        return [RECEIVER_EMAIL]
    if isinstance(receiver_emails, str):
        emails = [e.strip() for e in receiver_emails.split(",") if e.strip()]
        return emails or [RECEIVER_EMAIL]
    if isinstance(receiver_emails, list):
        emails = [e.strip() for e in receiver_emails if e.strip()]
        return emails or [RECEIVER_EMAIL]
    return [RECEIVER_EMAIL]


# ============================================================
# SEND EMAIL ALERT (with optional attachments)
# ============================================================
def send_email_alert(subject, message, attachments=None, receiver_emails=None):
    receiver_list = _normalize_recipient_input(receiver_emails)

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ", ".join(receiver_list)
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        # Attach graphs
        if attachments:
            for fname, data in attachments:
                img = MIMEImage(data)
                img.add_header('Content-Disposition', 'attachment', filename=fname)
                msg.attach(img)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_list, msg.as_string())
        server.quit()
        return True

    except Exception as e:
        st.error(f"Email failed: {e}")
        return False


# ============================================================
# CREATE PLOT IMAGES
# ============================================================
def create_graph_images(scored_df, df_feat, features):
    images = []

    # ----- Temperature & Humidity -----
    try:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(scored_df['ts'], scored_df['temperature'])
        ax1.set_ylabel("Temperature (°C)")
        ax2 = ax1.twinx()
        ax2.plot(scored_df['ts'], scored_df['humidity'], color="green")
        ax2.set_ylabel("Humidity (%)")

        anomalies = scored_df[scored_df['is_anomaly'] == 1]
        if not anomalies.empty:
            ax1.scatter(anomalies['ts'], anomalies['temperature'], color='red')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        images.append(("temp_hum.png", buf.read()))
        plt.close(fig)
    except:
        pass

    # ----- Rolling Stats -----
    try:
        win = 5
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        ax[0].plot(df_feat['ts'], df_feat['temperature'], alpha=0.5)
        ax[1].plot(df_feat['ts'], df_feat['humidity'], alpha=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        images.append(("rolling_stats.png", buf.read()))
        plt.close(fig)
    except:
        pass

    return images


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="OT-IoT Threat Monitoring Console", layout="wide")
st.title("🔎 OT-IoT Threat Monitoring Console")
st.caption("ESP8266 + Firebase + AI + Streamlit + Email Alerts")


# ============================================================
# FIREBASE FETCH
# ============================================================
FIREBASE_URL = (
    "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/"
    "forensics_logs.json"
)


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
        if ts:
            rows.append({
                "id": key,
                "ts": datetime.utcfromtimestamp(int(ts)),
                "temperature": float(v.get("temperature", np.nan)),
                "humidity": float(v.get("humidity", np.nan)),
            })

    return pd.DataFrame(rows).sort_values("ts")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineer(df, window=5):
    df = df.copy().set_index("ts")
    df["temperature"] = df["temperature"].interpolate().ffill()
    df["humidity"] = df["humidity"].interpolate().ffill()

    df["temp_diff"] = df["temperature"].diff().fillna(0)
    df["hum_diff"] = df["humidity"].diff().fillna(0)

    df["hour"] = df.index.hour

    return df.reset_index()
# ============================================================
# MODEL TRAINING
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
    st.warning("⚠ Waiting for sensor data…")
    st.stop()

start_dt = df_raw["ts"].max() - timedelta(hours=12)
df_window = df_raw[df_raw["ts"] >= start_dt]

df_feat = feature_engineer(df_window)

features = [
    "temperature", "humidity",
    "temp_diff", "hum_diff",
    "hour"
]

# ============================================================
# TRAIN MODEL
# ============================================================
scored_df, model, scaler = train_and_score(df_feat, "iforest", 0.02, features)

# ============================================================
# SUMMARY SECTION
# ============================================================
st.subheader("📊 System Summary")

m1, m2 = st.columns(2)
m1.metric("📡 Total Events", len(df_raw))
m2.metric("🚨 Total Anomalies", int(scored_df["is_anomaly"].sum()))

# ============================================================
# AUTOMATIC EMAIL ALERT — CLEAN FIXED VERSION
# ============================================================
latest_anomaly_rows = scored_df[scored_df["is_anomaly"] == 1]

if not latest_anomaly_rows.empty:

    # latest anomaly
    latest_anomaly = latest_anomaly_rows.iloc[-1]
    last_id = str(latest_anomaly["ts"])

    # session tracking
    if "last_alert_sent_id" not in st.session_state:
        st.session_state["last_alert_sent_id"] = None

    # Only send if NEW anomaly
    if st.session_state["last_alert_sent_id"] != last_id:

        # Build anomaly history text
        history_text = "Previous Anomalies (History):\n"
        for _, row in latest_anomaly_rows.iterrows():
            history_text += (
                f" - {row['ts']} | Temp: {row['temperature']}°C | "
                f"Hum: {row['humidity']}% | Score: {row['anomaly_score']:.4f}\n"
            )

        latest = df_raw.iloc[-1]

        subject = "AI IoT ALERT — New Anomaly Detected"

        message = f"""
===============================
      AI IoT Forensics Alert
===============================

A NEW anomaly has been detected.

🔴 Latest Anomaly
Timestamp : {latest_anomaly['ts']}
Score     : {latest_anomaly['anomaly_score']:.4f}
Temp      : {latest_anomaly['temperature']} °C
Humidity  : {latest_anomaly['humidity']} %

📡 Current Sensor Status
Temperature : {latest['temperature']} °C
Humidity    : {latest['humidity']} %

{history_text}

This alert is triggered ONLY once per anomaly.
"""

        # Attach graphs
        attachments = create_graph_images(scored_df, df_feat, features)

        # Send
        ok = send_email_alert(subject, message, attachments, RECEIVER_EMAIL)

        if ok:
            st.session_state["last_alert_sent_id"] = last_id
            st.success("📧 Automatic alert sent (with graphs)")
        else:
            st.error("Failed to send alert")
# ============================================================
# MANUAL ALERT UI — contacts, quick-buttons, add-new, send
# ============================================================
st.markdown("---")
st.subheader("📤 Manual Alert (send to any email)")

# initialize contacts
if "contacts" not in st.session_state:
    st.session_state["contacts"] = [
        {"name": "Me (Manas)", "email": "manas.dfis242604@nfsu.ac.in"},
    ]

QUICK_CONTACTS = [
    {"name": "Nandini", "email": "nandini.dfis242606@nfsu.ac.in"},
    {"name": "Jayendra", "email": "jayendra.dfis242605@nfsu.ac.in"},
    {"name": "Ujjaval", "email": "ujjaval.patel@nfsu.ac.in"},
]

# contact search
search_query = st.text_input("Search contacts (type email to filter)", key="contact_search")
if search_query:
    filtered = [c for c in st.session_state["contacts"] if search_query.lower() in c["email"].lower()]
else:
    filtered = st.session_state["contacts"]

st.write("**Contacts** — click to add to recipients")
if filtered:
    cols = st.columns(min(len(filtered), 4))
    for i, c in enumerate(filtered):
        with cols[i % len(cols)]:
            if st.button(f"Add {c['email']}", key=f"add_contact_{i}"):
                if "manual_recipients" not in st.session_state or not st.session_state["manual_recipients"]:
                    st.session_state["manual_recipients"] = c["email"]
                else:
                    existing = [e.strip() for e in st.session_state["manual_recipients"].split(",") if e.strip()]
                    if c["email"] not in existing:
                        existing.append(c["email"])
                        st.session_state["manual_recipients"] = ", ".join(existing)
                st.success(f"Added {c['email']}")
else:
    st.info("No contacts match. Use 'Add new email' below.")

# quick contacts
st.write("**Quick Contacts** — one-click add")
qcols = st.columns(len(QUICK_CONTACTS))
for i, qc in enumerate(QUICK_CONTACTS):
    with qcols[i]:
        if st.button(f"Add {qc['name']}", key=f"quick_add_{i}"):
            if "manual_recipients" not in st.session_state or not st.session_state["manual_recipients"]:
                st.session_state["manual_recipients"] = qc["email"]
            else:
                existing = [e.strip() for e in st.session_state["manual_recipients"].split(",") if e.strip()]
                if qc["email"] not in existing:
                    existing.append(qc["email"])
                    st.session_state["manual_recipients"] = ", ".join(existing)
            if not any(c.get("email") == qc["email"] for c in st.session_state["contacts"]):
                st.session_state["contacts"].append({"name": qc["name"], "email": qc["email"]})
            st.success(f"Added {qc['email']}")

# add new email
with st.expander("➕ Add new email"):
    new_email = st.text_input("Email address", key="new_contact_email_only")
    if st.button("Save email", key="save_new_email"):
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not new_email:
            st.error("Email required.")
        elif not re.match(pattern, new_email):
            st.error("Invalid email format.")
        else:
            st.session_state["contacts"].append({"name": new_email, "email": new_email})
            st.session_state["manual_recipients"] = new_email
            st.success("Saved. Use buttons to add to the recipient list.")

# recipient input
if "manual_recipients" not in st.session_state:
    st.session_state["manual_recipients"] = ""

recipient_input = st.text_input(
    "Recipient email(s) (comma-separated). You can add via contact buttons above.",
    value=st.session_state["manual_recipients"],
    key="recipient_input"
)

include_graphs_manual = st.checkbox("Include graphs/attachments", value=True, key="include_graphs_manual")

def is_valid_email_list(s):
    if not s:
        return False
    emails = [e.strip() for e in s.split(",") if e.strip()]
    if not emails:
        return False
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return all(re.match(pattern, e) for e in emails)

if st.button("📤 Send Manual Alert"):
    if not is_valid_email_list(recipient_input):
        st.error("Please provide at least one valid recipient email.")
    else:
        if not latest_anomaly_rows.empty:
            latest_anom = latest_anomaly_rows.iloc[-1]
            latest_row = df_raw.iloc[-1]
            subject = "MANUAL ALERT — AI IoT – Anomaly Detected"
            message = (
                "MANUAL ALERT — AI IoT Forensics\n\n"
                f"Anomaly Detected\nTimestamp : {latest_anom['ts']}\n"
                f"Anomaly Score : {latest_anom['anomaly_score']:.4f}\n\n"
                f"Sensor Values at Anomaly\nTemperature : {latest_anom['temperature']} °C\n"
                f"Humidity    : {latest_anom['humidity']} %\n\n"
                f"Current Live Sensor Values\nLive Temperature : {latest_row['temperature']} °C\n"
                f"Live Humidity    : {latest_row['humidity']} %\n\n"
                "Note: Triggered from Streamlit console.\n"
            )
        else:
            latest_row = df_raw.iloc[-1]
            subject = "MANUAL ALERT — AI IoT – Status Update (no anomaly)"
            message = (
                "MANUAL ALERT — AI IoT Forensics\n\n"
                "No anomalies detected in current window.\n"
                f"Live Temperature : {latest_row['temperature']} °C\n"
                f"Live Humidity    : {latest_row['humidity']} %\n\n"
                "Note: Triggered from Streamlit console.\n"
            )

        attachments = []
        if include_graphs_manual:
            attachments = create_graph_images(scored_df, df_feat, features)

        ok = send_email_alert(subject, message, attachments=attachments if include_graphs_manual else None, receiver_emails=recipient_input)
        if ok:
            st.success(f"📨 Manual alert sent to: {recipient_input}")
        else:
            st.error("Failed to send manual alert.")

# ============================================================
# LIVE SENSOR + CHARTS + DIAGNOSTICS
# ============================================================
latest = df_raw.iloc[-1]

st.subheader("📡 Live Sensor Status")
c1, c2, c3 = st.columns(3)
c1.metric("🌡 Temperature", f"{latest['temperature']:.2f} °C")
c2.metric("💧 Humidity", f"{latest['humidity']:.2f} %")
c3.metric("⏱ Last Update", latest["ts"].strftime("%Y-%m-%d %H:%M:%S"))

st.markdown("---")

# Temperature chart
st.subheader("📈 Temperature (with anomalies)")
base = alt.Chart(scored_df).encode(x="ts:T")
st.altair_chart(
    base.mark_line().encode(y="temperature:Q") +
    base.transform_filter("datum.is_anomaly == 1")
    .mark_circle(color="red", size=60).encode(y="temperature:Q"),
    use_container_width=True
)

# Humidity chart
st.subheader("📉 Humidity (with anomalies)")
st.altair_chart(
    base.mark_line().encode(y="humidity:Q") +
    base.transform_filter("datum.is_anomaly == 1")
    .mark_circle(color="red", size=60).encode(y="humidity:Q"),
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
except Exception:
    st.info("PCA visualization not available for current data.")

# Anomaly table
st.subheader("📜 Anomaly Table")
st.dataframe(scored_df.sort_values("anomaly_score", ascending=False))

# OT visuals
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
except Exception:
    st.warning("Could not create additional visuals.")

# Quick diagnostics
st.subheader("🔧 Quick Diagnostics")
col1, col2, col3 = st.columns(3)
col1.metric("Anomaly Rate (window)", f"{(scored_df['is_anomaly'].mean()*100):.2f}%")
col2.metric("Latest Anomaly Score", f"{scored_df['anomaly_score'].max():.4f}")
col3.metric("Events in window", len(scored_df))

st.markdown("---")
st.caption("Graphs are attached automatically to alerts when requested. Automatic alerts are only sent once per new anomaly.")
