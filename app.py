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

# ========== ADDED IMPORTS ==========
import matplotlib.pyplot as plt
from email.mime.image import MIMEImage
import matplotlib.dates as mdates
# ===================================

# ============================================================
# EMAIL ALERT — HARDCODED CREDENTIALS
# ============================================================
SENDER_EMAIL = "manas.dfis242604@nfsu.ac.in"
APP_PASSWORD = "euozfdlazplbmtkd"              # Google App Password
RECEIVER_EMAIL = "manas.dfis242604@nfsu.ac.in"


# ============================================================
# EMAIL SENDING FUNCTION (original, preserved)
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

# ========== ENHANCEMENT: keep original function under a new name,
# then override send_email_alert with a richer version that includes graphs.
# This is non-destructive: original behavior is preserved as send_email_alert_text.
send_email_alert_text = send_email_alert
# ===================================


# ============================================================
# ADDITIONAL HELPERS — create plots/images and attach to email
# ============================================================
def create_graph_images(scored_df, df_feat, features):
    """
    Return list of tuples (filename, bytes_data) containing PNG images:
    - temp_hum_with_anomalies.png : Matplotlib line chart with anomaly markers
    - rolling_stats.png : Rolling mean & std
    - anomalies_by_hour.png : Count of anomalies per hour
    - cumulative_anomalies.png : cumulative anomalies over time
    - ladder_logic.png : simple ladder-like diagram for OT flavor
    """
    images = []

    # ---------- 1) Temperature & Humidity with anomaly markers ----------
    try:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(scored_df['ts'], scored_df['temperature'], label='Temperature', linewidth=1.5)
        ax1.set_ylabel('Temperature (°C)')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2 = ax1.twinx()
        ax2.plot(scored_df['ts'], scored_df['humidity'], label='Humidity', linewidth=1.0, linestyle='--')
        ax2.set_ylabel('Humidity (%)')

        # Mark anomalies
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

    # ---------- 2) Rolling means and std ----------
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

    # ---------- 3) Anomaly count by hour ----------
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

    # ---------- 4) Cumulative anomaly curve ----------
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

    # ---------- 5) Simple Ladder Logic Diagram (OT flavor) ----------
    try:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        # draw few rungs
        rung_y = [9, 7, 5, 3]
        for y in rung_y:
            ax.hlines(y, 1, 9, linewidth=3, color='black')
            # left and right vertical rails
        ax.vlines(1, 2, 10, linewidth=4)
        ax.vlines(9, 2, 10, linewidth=4)
        # put some "contacts" and "coils"
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


def send_email_alert_with_graphs(subject, message, attachments):
    """
    attachments: list of (filename, bytes)
    """
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


# Override send_email_alert to attach graphs automatically.
# We keep the original as send_email_alert_text (above) and now point send_email_alert
# to the enhanced version so existing calls will include graphs.
def _enhanced_send_email_alert(subject, message, scored_df_local=None, df_feat_local=None, features_local=None):
    # First send the plain-text email (preserve original behavior / logs)
    try:
        send_email_alert_text(subject, message)
    except Exception:
        pass

    # Create attachments if we have data
    attachments = []
    if scored_df_local is not None and df_feat_local is not None:
        try:
            attachments = create_graph_images(scored_df_local, df_feat_local, features_local or [])
        except Exception as e:
            print("Failed to create attachments:", e)

    # Send full email with attachments
    return send_email_alert_with_graphs(subject + " (with graphs)", message, attachments)

# Rebind the name so later calls to send_email_alert() send graphs
send_email_alert = _enhanced_send_email_alert
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="OT-IoT Threat Monitoring Console", layout="wide")
st.title("🔎 OT-IoT Threat Monitoring Console")
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

        subject = "AI IoT ALERT — Anomaly Detected"

        message = f"""
===============================
   **AI IoT Forensics Alert**
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

        # === EXISTING CALL: sends plain text originally, now invokes enhanced function that also attaches graphs
        try:
            # We call the enhanced send_email_alert with data so attachments are created
            send_email_alert(subject, message, scored_df_local=scored_df, df_feat_local=df_feat, features_local=features)
            st.session_state["last_alert_sent_id"] = last_anom_id
            st.success("📧 Automatic alert sent for latest anomaly (with graphs)!")
        except Exception as e:
            # fallback: attempt text-only send
            send_email_alert_text(subject, message)
            st.session_state["last_alert_sent_id"] = last_anom_id
            st.success("📧 Automatic alert sent (text-only fallback).")


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
        # Call enhanced send (will attach graphs)
        try:
            send_email_alert("Manual IoT Alert", msg, scored_df_local=scored_df, df_feat_local=df_feat, features_local=features)
            st.info("📨 Manual alert sent (with graphs).")
        except Exception as e:
            send_email_alert_text("Manual IoT Alert", msg)
            st.info("📨 Manual alert sent (text-only fallback).")
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

# ============================================================
# ========== NEW: Additional OT-style Visuals and Graphs ==========
# ============================================================
st.markdown("---")
st.subheader("🧭 OT Ladder & Additional Diagnostics (added)")

# Display the ladder logic image and other images produced for the email in the UI
try:
    images = create_graph_images(scored_df, df_feat, features)
    cols = st.columns(3)
    for idx, (fname, bdata) in enumerate(images):
        # convert bytes to displayable PNG via base64
        b64 = base64.b64encode(bdata).decode()
        img_md = f"data:image/png;base64,{b64}"
        with cols[idx % 3]:
            st.image(img_md, caption=fname, use_column_width=True)
except Exception as e:
    st.warning("Could not create additional visuals: " + str(e))

# Extra interactive chart: anomalies over time (Altair)
try:
    st.subheader("📊 Anomalies Over Time (Altair)")
    anomaly_ts = scored_df[scored_df['is_anomaly'] == 1][['ts', 'anomaly_score']]
    if not anomaly_ts.empty:
        st.altair_chart(
            alt.Chart(anomaly_ts).mark_circle(size=80, color="red").encode(
                x='ts:T', y='anomaly_score:Q', tooltip=['ts', 'anomaly_score']
            ).interactive(),
            use_container_width=True
        )
    else:
        st.info("No anomalies in current window to plot (Altair).")
except Exception as e:
    st.warning("Could not show Altair anomaly chart: " + str(e))

# Extra gauge-like metrics for OT feel
try:
    st.subheader("🔧 OT-style Quick Diagnostics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly Rate (window)", f"{(scored_df['is_anomaly'].mean()*100):.2f}%")
    col2.metric("Latest Anomaly Score", f"{scored_df['anomaly_score'].max():.4f}")
    col3.metric("Events in window", len(scored_df))
except Exception as e:
    st.warning("Could not compute diagnostics: " + str(e))

st.markdown("---")
st.caption("Added visuals include rolling stats, anomaly histograms by hour, cumulative curve, and a simplified ladder-logic diagram to give an OT flavor. Graphs are attached automatically to alerts.")

# End of file
