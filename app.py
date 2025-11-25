"""
app.py - OT-IoT Threat Monitoring Console (Futuristic Dark Blue IoT Theme)

Features (updated & advanced):
- All original functionality: fetch from Firebase, feature engineering, IsolationForest anomaly detection
- New UI theme (Futuristic Dark Blue / Neon) with animated header, animated banner, wallpaper
- Advanced sidebar controls: model params, contamination, rolling window size, alert options
- Live retrain button, manual train button, and model persistence in session_state
- Enhanced visuals: Matplotlib images, Altair charts, PCA map, animated cards
- Alerting: automatic one-time-per-anomaly email alerts + manual alerts with attachments
- Report generation: PDF report builder (uses reportlab if available; falls back to zipped PNGs)
- Downloads: CSV export of scored data, ZIP of images, PDF report (when supported)
- Contact management in session_state (Me + Add new)
- Better logging and error handling
- Lightweight performance improvements (caching, minimal redraws)

Note: For email sending via Gmail, either set APP_PASSWORD appropriately or
use an app-specific password. You may also set environment variables
SENDER_EMAIL and APP_PASSWORD to avoid hardcoding credentials.

Place this file in the same folder as your existing project and run:
streamlit run app.py
"""

# ============================
# Imports
# ============================
import os
import io
import re
import time
import zipfile
import base64
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional imports for PDF generation; handled gracefully
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ============================
# CONFIG (edit if needed or set via env)
# ============================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "manas.dfis242604@nfsu.ac.in")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "euozfdlazplbmtkd")  # consider using env vars
DEFAULT_RECEIVER = os.environ.get("DEFAULT_RECEIVER", SENDER_EMAIL)

# Firebase endpoint for the Realtime DB (example)
FIREBASE_URL = os.environ.get(
    "FIREBASE_URL",
    "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/forensics_logs.json"
)

# App settings
PAGE_TITLE = "OT-IoT Threat Monitoring Console"
PAGE_ICON = "🔐"
CACHE_TTL = 15  # seconds for data fetch caching

# ============================
# Helper utils
# ============================
def normalize_recipients(receiver_emails):
    if receiver_emails is None:
        return [DEFAULT_RECEIVER]
    if isinstance(receiver_emails, str):
        emails = [e.strip() for e in receiver_emails.split(",") if e.strip()]
        return emails or [DEFAULT_RECEIVER]
    if isinstance(receiver_emails, (list, tuple)):
        emails = [str(e).strip() for e in receiver_emails if str(e).strip()]
        return emails or [DEFAULT_RECEIVER]
    return [DEFAULT_RECEIVER]

def is_valid_email_list(s):
    if not s:
        return False
    emails = [e.strip() for e in s.split(",") if e.strip()]
    if not emails:
        return False
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return all(re.match(pattern, e) for e in emails)

# ============================
# Email sender (keeps original MIME attachments behavior)
# ============================
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

def send_email_with_attachments(subject, message, attachments=None, receiver_emails=None, smtp_host="smtp.gmail.com", smtp_port=587, timeout=30):
    receivers = normalize_recipients(receiver_emails)
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ", ".join(receivers)
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        if attachments:
            for fname, b in attachments:
                try:
                    # If bytes are image-like, attach as image; otherwise attach generic
                    img = MIMEImage(b)
                    img.add_header("Content-Disposition", "attachment", filename=fname)
                    msg.attach(img)
                except Exception:
                    # fallback: attach as generic payload
                    part = MIMEText(base64.b64encode(b).decode(), "plain")
                    part.add_header("Content-Disposition", "attachment", filename=fname + ".b64")
                    msg.attach(part)

        server = smtplib.SMTP(smtp_host, smtp_port, timeout=timeout)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, receivers, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email send error: {e}")
        return False

# ============================
# Theming & CSS (Futuristic Dark Blue / Neon)
# ============================
def inject_theme_css():
    css = f"""
    <style>
    /* Page background */
    [data-testid="stAppViewContainer"] {{
        background-image: url("https://i.ibb.co/KV0x5Zn/iot-bg-dark-blue.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(0,6,30,0.85), rgba(0,14,40,0.9));
    }}
    /* Glass panels for metrics */
    .glass-card {{
        background: rgba(10, 20, 30, 0.55);
        border-radius: 12px;
        padding: 12px;
        color: #eafcff;
        box-shadow: 0 8px 30px rgba(0, 200, 255, 0.06);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(0,255,240,0.05);
    }}
    /* Glowing title */
    .glow-title {{
        font-size: 34px;
        color: #00eaff;
        text-align: left;
        font-weight: 700;
        text-shadow: 0 0 8px #00eaff, 0 0 18px rgba(0,234,255,0.4), 0 0 30px rgba(0,200,255,0.12);
        animation: glow 2.2s ease-in-out infinite alternate;
        margin-bottom: 6px;
    }}
    @keyframes glow {{
        from {{ text-shadow: 0 0 8px #00eaff; }}
        to   {{ text-shadow: 0 0 20px #00eaff, 0 0 30px rgba(0,234,255,0.2); }}
    }}
    /* Sensor cards */
    .sensor-card {{
        padding: 14px;
        border-radius: 10px;
        background: linear-gradient(180deg, rgba(0,0,0,0.22), rgba(0,0,0,0.18));
        color: #dffcff;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,150,200,0.06);
    }}
    /* Small text tweaks */
    .muted {{
        color: rgba(220,255,255,0.65);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def header_banner():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="flex:1;">
                <div class="glow-title">🔐 OT–IoT Threat Monitoring Console</div>
                <div class="muted">ESP8266 · Firebase · AI Anomaly Detection · Email Alerts</div>
            </div>
            <div style="width:340px; text-align:right;">
                <img src="https://i.ibb.co/VNQKfFn/iot-animation.gif" width="280" style="margin-right:6px; border-radius:8px; box-shadow:0 12px 30px rgba(0,0,0,0.5);" />
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# ============================
# Data fetch + feature engineering
# ============================
@st.cache_data(ttl=CACHE_TTL)
def fetch_raw_data(url):
    """
    Fetch JSON from Firebase realtime DB endpoint (assumes structure
    { record_id: { timestamp: <epoch>, temperature: .., humidity: .. }, ... })
    Returns DataFrame with id, ts (datetime UTC), temperature, humidity.
    """
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as e:
        st.error(f"Failed to fetch data from Firebase: {e}")
        return pd.DataFrame(columns=["id", "ts", "temperature", "humidity"])

    if not data:
        return pd.DataFrame(columns=["id", "ts", "temperature", "humidity"])

    rows = []
    for rid, v in (data.items() if isinstance(data, dict) else []):
        if not isinstance(v, dict):
            continue
        ts = v.get("timestamp")
        if ts is None:
            continue
        # Accept epoch seconds (int/str) or ISO timestamp
        dt = None
        try:
            dt = datetime.utcfromtimestamp(int(ts))
        except Exception:
            try:
                dt = pd.to_datetime(ts, utc=True).to_pydatetime()
            except Exception:
                continue
        temp = v.get("temperature")
        hum = v.get("humidity")
        try:
            temp = float(temp) if temp is not None else np.nan
        except Exception:
            temp = np.nan
        try:
            hum = float(hum) if hum is not None else np.nan
        except Exception:
            hum = np.nan
        rows.append({"id": str(rid), "ts": dt, "temperature": temp, "humidity": hum})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("ts").reset_index(drop=True)

def feature_engineer(df, window=5):
    """
    Input df must have columns: id, ts, temperature, humidity
    Returns df with additional engineered features and preserved id.
    """
    if df.empty:
        return df.copy()
    tmp = df.copy().set_index("ts")
    if "id" in df.columns:
        tmp["id"] = df.set_index("ts")["id"]
    tmp["temperature"] = tmp["temperature"].interpolate().ffill().bfill()
    tmp["humidity"] = tmp["humidity"].interpolate().ffill().bfill()
    tmp["temp_diff"] = tmp["temperature"].diff().fillna(0)
    tmp["hum_diff"] = tmp["humidity"].diff().fillna(0)
    tmp["temp_ma"] = tmp["temperature"].rolling(window, min_periods=1).mean()
    tmp["hum_ma"] = tmp["humidity"].rolling(window, min_periods=1).mean()
    tmp["temp_std"] = tmp["temperature"].rolling(window, min_periods=1).std().fillna(0)
    tmp["hum_std"] = tmp["humidity"].rolling(window, min_periods=1).std().fillna(0)
    tmp["temp_z"] = (tmp["temperature"] - tmp["temp_ma"]) / tmp["temp_std"].replace(0, 1)
    tmp["hum_z"] = (tmp["humidity"] - tmp["hum_ma"]) / tmp["hum_std"].replace(0, 1)
    tmp["hour"] = tmp.index.hour
    out = tmp.reset_index()
    if "id" not in out.columns:
        out["id"] = np.nan
    return out

# ============================
# Model training / scoring
# ============================
def train_and_score(df_feat, model_type="iforest", contamination=0.02, features=None):
    """
    Returns: scored_df, model, scaler
    scored_df contains anomaly_score & is_anomaly (0/1)
    """
    if df_feat.empty or not features:
        return df_feat.copy(), None, None
    X = df_feat[features].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if model_type == "iforest":
        model = IsolationForest(contamination=contamination, random_state=42)
    else:
        # fallback to iforest for now
        model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(Xs)
    # higher scores -> more anomalous; keep original sign convention
    try:
        scores = -model.decision_function(Xs)
    except Exception:
        try:
            scores = model.score_samples(Xs) * -1.0
        except Exception:
            scores = np.zeros(Xs.shape[0])
    labels = (model.predict(Xs) == -1).astype(int)
    df = df_feat.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"] = labels
    return df, model, scaler

# ============================
# Plot & image creation functions
# ============================
def create_graph_images(scored_df, df_feat, features, show_ladder=True):
    """
    Return list of (filename, bytes) images (PNG)
    """
    images = []
    try:
        # 1) Temp & Hum with anomalies (matplotlib)
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(scored_df["ts"], scored_df["temperature"], label="Temperature", linewidth=1.6)
        ax1.set_ylabel("Temperature (°C)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2 = ax1.twinx()
        ax2.plot(scored_df["ts"], scored_df["humidity"], label="Humidity", linewidth=1.0, linestyle="--")
        ax2.set_ylabel("Humidity (%)")

        anomalies = scored_df[scored_df["is_anomaly"] == 1]
        if not anomalies.empty:
            ax1.scatter(anomalies["ts"], anomalies["temperature"], edgecolors="r", facecolors="none", s=90, label="Anom Temp")
            ax2.scatter(anomalies["ts"], anomalies["humidity"], color="red", marker="x", s=70, label="Anom Hum")

        ax1.set_title("Temperature & Humidity (with anomalies)")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        images.append(("temp_hum_with_anomalies.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("temp_hum plot error:", e)

    try:
        # 2) Rolling means
        win = 5
        tmp = df_feat.set_index("ts")
        rtemp = tmp["temperature"].rolling(win).mean()
        rhum = tmp["humidity"].rolling(win).mean()
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(df_feat["ts"], df_feat["temperature"], alpha=0.35, label="Temp raw")
        axs[0].plot(rtemp.index, rtemp.values, linewidth=2, label=f"{win}-pt MA")
        axs[0].set_ylabel("Temperature (°C)")
        axs[0].legend()
        axs[1].plot(df_feat["ts"], df_feat["humidity"], alpha=0.35, label="Hum raw")
        axs[1].plot(rhum.index, rhum.values, linewidth=2, label=f"{win}-pt MA")
        axs[1].set_ylabel("Humidity (%)")
        axs[1].legend()
        fig.autofmt_xdate()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        images.append(("rolling_stats.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("rolling plot error:", e)

    try:
        # 3) Anomalies by hour
        if "hour" not in scored_df.columns:
            scored_df["hour"] = scored_df["ts"].dt.hour
        byhour = scored_df.groupby("hour")["is_anomaly"].sum().reindex(range(24), fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(byhour.index, byhour.values)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Anomalies Count")
        ax.set_title("Anomalies by Hour (last window)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        images.append(("anomalies_by_hour.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("byhour plot error:", e)

    try:
        # 4) Cumulative curve
        ordered = scored_df.sort_values("ts").copy()
        ordered["cum_anom"] = ordered["is_anomaly"].cumsum()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ordered["ts"], ordered["cum_anom"], marker="o")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Anomalies")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        images.append(("cumulative_anomalies.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("cumulative plot error:", e)

    # 5) Ladder logic (simple)
    if show_ladder:
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")
            rung_y = [9, 7, 5, 3]
            for y in rung_y:
                ax.hlines(y, 1, 9, linewidth=3, color="white")
            ax.vlines(1, 2, 10, linewidth=4, color="white")
            ax.vlines(9, 2, 10, linewidth=4, color="white")
            ax.text(2, 8.6, "I: Sensor OK", fontsize=10, color="#00eaff")
            ax.text(2, 6.6, "I: Manual Stop", fontsize=10, color="#00eaff")
            ax.text(5, 4.6, "M: Safety Interlock", fontsize=10, color="#00eaff")
            ax.text(6.5, 2.6, "Q: Alarm Output", fontsize=10, color="red", fontweight="bold")
            ax.text(4.5, 1.2, "OT Ladder (Simplified)", fontsize=9, ha="center", color="#eafcff")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            images.append(("ladder_logic.png", buf.read()))
            plt.close(fig)
        except Exception as e:
            print("ladder plot error:", e)

    return images

# ============================
# PDF report generator
# ============================
def create_pdf_report(scored_df, df_raw, images, report_title="OT-IoT Report"):
    """
    Create a simple PDF report containing a header, key metrics and embedded PNG images.
    Uses reportlab if available; otherwise returns None.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(36, height - 36, report_title)
    c.setFont("Helvetica", 10)
    c.drawString(36, height - 54, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Key metrics
    total_events = len(df_raw)
    total_anoms = int(scored_df["is_anomaly"].sum()) if not scored_df.empty else 0
    latest_ts = df_raw["ts"].max() if not df_raw.empty else None
    c.drawString(36, height - 76, f"Total events: {total_events}")
    c.drawString(36, height - 92, f"Total anomalies: {total_anoms}")
    c.drawString(36, height - 108, f"Latest sample timestamp: {latest_ts}")

    y = height - 140
    for fname, b in images:
        try:
            # write image to temporary buffer and draw
            img_buf = io.BytesIO(b)
            c.drawImage(img_buf, 36, y - 180, width=520, height=140, preserveAspectRatio=True, mask='auto')
            y -= 180 + 12
            if y < 120:
                c.showPage()
                y = height - 60
        except Exception:
            continue

    c.save()
    buffer.seek(0)
    return buffer.read()

# ============================
# Utility: prepare zip of images
# ============================
def images_to_zip_bytes(images):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname, data in images:
            z.writestr(fname, data)
    buf.seek(0)
    return buf.read()

# ============================
# Main Streamlit UI
# ============================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
inject_theme_css()
header_banner()

# Sidebar controls
st.sidebar.header("Controls & Model")
with st.sidebar.expander("Data & Window"):
    firebase_input = st.text_input("Firebase URL", value=FIREBASE_URL)
    window_hours = st.number_input("Window (hours to analyze)", min_value=1, max_value=168, value=12)
    rolling_window = st.slider("Rolling window (points)", 1, 30, 5)

with st.sidebar.expander("Model & Detection"):
    model_type = st.selectbox("Model", ["iforest"], index=0)
    contamination = st.slider("Contamination (expected fraction of anomalies)", 0.001, 0.2, 0.02, step=0.001, format="%.3f")
    anomaly_threshold = st.number_input("Anomaly score threshold (optional, <=0 to auto)", value=0.0, format="%.4f")

with st.sidebar.expander("Alerts & Email"):
    auto_alerts = st.checkbox("Enable automatic email alerts (one per anomaly id)", value=True)
    include_graphs_in_alert = st.checkbox("Include graphs in email alerts", value=True)
    sender_edit = st.text_input("Sender email", value=SENDER_EMAIL)
    receiver_default = st.text_input("Default receiver email", value=DEFAULT_RECEIVER)

with st.sidebar.expander("Advanced"):
    enable_pca = st.checkbox("Enable PCA visualization", value=True)
    show_ladder = st.checkbox("Show ladder-logic visual", value=True)
    use_reportlab = st.checkbox("Use reportlab for PDF reports", value=REPORTLAB_AVAILABLE and True)

# Update globals from sidebar if needed
SENDER_EMAIL = sender_edit or SENDER_EMAIL
DEFAULT_RECEIVER = receiver_default or DEFAULT_RECEIVER

# Fetch data
with st.spinner("Fetching data..."):
    df_raw = fetch_raw_data(firebase_input)

if df_raw.empty:
    st.warning("⚠ Waiting for sensor data (no records found). Add data to Firebase or verify the URL.")
    st.stop()

# windowing
start_dt = df_raw["ts"].max() - timedelta(hours=window_hours)
df_window = df_raw[df_raw["ts"] >= start_dt].copy()
df_feat = feature_engineer(df_window, window=rolling_window)
features = ["temperature", "humidity", "temp_diff", "hum_diff", "temp_z", "hum_z", "hour"]

# training controls
if "model_state" not in st.session_state:
    st.session_state["model_state"] = {"model": None, "scaler": None, "features": features, "trained_at": None}
if "last_alert_sent_id" not in st.session_state:
    st.session_state["last_alert_sent_id"] = None
if "contacts" not in st.session_state:
    st.session_state["contacts"] = [{"name": "Me", "email": DEFAULT_RECEIVER}]

# Live retrain logic
col_train1, col_train2 = st.columns([1, 1])
with col_train1:
    if st.button("Retrain Model Live"):
        scored_df, model, scaler = train_and_score(df_feat, model_type, contamination, features)
        st.session_state["model_state"] = {"model": model, "scaler": scaler, "features": features, "trained_at": datetime.utcnow().isoformat()}
        st.success("Model retrained using latest window.")
with col_train2:
    if st.button("Train & Persist (session)"):
        scored_df, model, scaler = train_and_score(df_feat, model_type, contamination, features)
        st.session_state["model_state"] = {"model": model, "scaler": scaler, "features": features, "trained_at": datetime.utcnow().isoformat()}
        st.success("Model trained and stored in session.")

# If model exists in session, use it; otherwise, train
if st.session_state["model_state"]["model"] is not None:
    try:
        model = st.session_state["model_state"]["model"]
        scaler = st.session_state["model_state"]["scaler"]
        scored_df, model_unused, scaler_unused = train_and_score(df_feat, model_type, contamination, features)
    except Exception:
        scored_df, model, scaler = train_and_score(df_feat, model_type, contamination, features)
else:
    scored_df, model, scaler = train_and_score(df_feat, model_type, contamination, features)

# Show top summary metrics
st.markdown("---")
st.subheader("System Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Total Events Received", len(df_raw))
m2.metric("Events in Window", len(scored_df))
m3.metric("Anomalies Detected", int(scored_df["is_anomaly"].sum() if not scored_df.empty else 0))

# Live sensor status cards (neon)
latest = df_raw.iloc[-1]
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    st.markdown(f'<div class="sensor-card"><h4>🌡 Temperature</h4><h2>{latest["temperature"]:.2f} °C</h2></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="sensor-card"><h4>💧 Humidity</h4><h2>{latest["humidity"]:.2f} %</h2></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="sensor-card"><h4>⏱ Last Update</h4><h3>{latest["ts"].strftime("%Y-%m-%d %H:%M:%S")}</h3></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="sensor-card"><h4>⚙️ Model Trained</h4><h4>{st.session_state["model_state"].get("trained_at","Not trained")}</h4></div>', unsafe_allow_html=True)

# Altair charts for temperature & humidity with anomaly markers
st.markdown("---")
st.subheader("Live Charts")
base = alt.Chart(scored_df).encode(x="ts:T")
temp_line = base.mark_line().encode(y="temperature:Q").properties(title="Temperature (window)")
temp_points = base.transform_filter("datum.is_anomaly == 1").mark_point(size=80, filled=False, color="red").encode(y="temperature:Q")
st.altair_chart(temp_line + temp_points, use_container_width=True)

hum_line = base.mark_line().encode(y="humidity:Q").properties(title="Humidity (window)")
hum_points = base.transform_filter("datum.is_anomaly == 1").mark_point(size=80, filled=True, color="red").encode(y="humidity:Q")
st.altair_chart(hum_line + hum_points, use_container_width=True)

# PCA map
if enable_pca:
    st.subheader("PCA Anomaly Map")
    try:
        X_vis = scaler.transform(df_feat[features].fillna(0))
        p = PCA(n_components=2).fit_transform(X_vis)
        p_df = pd.DataFrame({"pc1": p[:, 0], "pc2": p[:, 1], "is_anomaly": scored_df["is_anomaly"].astype(int)})
        p_df["label"] = p_df["is_anomaly"].apply(lambda x: "anomaly" if x == 1 else "normal")
        chart = alt.Chart(p_df).mark_circle(size=60).encode(
            x="pc1:Q", y="pc2:Q",
            color=alt.Color("label:N", scale=alt.Scale(range=["#00a7ff", "#ff3b3b"]))
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.info("PCA visualization not available for current data.")

# Anomaly table & controls
st.subheader("Anomaly Table & Diagnostics")
st.dataframe(scored_df.sort_values("anomaly_score", ascending=False).reset_index(drop=True))

# Generate images and show OT visuals
st.markdown("---")
st.subheader("OT Ladder & Diagnostic Images")
try:
    images = create_graph_images(scored_df, df_feat, features, show_ladder=show_ladder)
    cols = st.columns(3)
    for idx, (fname, bdata) in enumerate(images):
        b64 = base64.b64encode(bdata).decode()
        img_md = f"data:image/png;base64,{b64}"
        with cols[idx % 3]:
            st.image(img_md, caption=fname, use_column_width=True)
except Exception:
    st.warning("Could not create additional visuals.")

# Quick diagnostics metrics
st.markdown("---")
st.subheader("Quick Diagnostics")
diag_c1, diag_c2, diag_c3 = st.columns(3)
diag_c1.metric("Anomaly Rate (window)", f"{(scored_df['is_anomaly'].mean()*100):.2f}%")
diag_c2.metric("Latest Anomaly Score", f"{scored_df['anomaly_score'].max():.4f}")
diag_c3.metric("Events in window", len(scored_df))

# ============================
# Automatic Alerting (one email per new anomaly id)
# ============================
if auto_alerts and not scored_df.empty:
    if "id" not in scored_df.columns:
        scored_df = scored_df.merge(df_window[["ts", "id"]], on="ts", how="left")

    latest_anomaly_rows = scored_df[scored_df["is_anomaly"] == 1].copy()
    if not latest_anomaly_rows.empty:
        latest_row = latest_anomaly_rows.sort_values("ts").iloc[-1]
        latest_anom_id = str(latest_row.get("id", latest_row["ts"]))

        if st.session_state["last_alert_sent_id"] != latest_anom_id:
            # Prepare alert
            history_lines = []
            for _, r in latest_anomaly_rows.sort_values("ts").iterrows():
                rid = r.get("id", "")
                history_lines.append(f"{rid} | {r['ts']} | Temp: {r['temperature']}°C | Hum: {r['humidity']}% | Score: {r['anomaly_score']:.4f}")
            history_text = "\n".join(history_lines)
            subject = "AI IoT ALERT — New Anomaly Detected"
            message = (
                "AI IoT Forensics Alert\n\n"
                f"Latest Anomaly (id={latest_anom_id})\n"
                f"Timestamp : {latest_row['ts']}\n"
                f"Score     : {latest_row['anomaly_score']:.4f}\n"
                f"Temp      : {latest_row['temperature']} °C\n"
                f"Humidity  : {latest_row['humidity']} %\n\n"
                f"Current sensor values: Temp {latest['temperature']} °C, Hum {latest['humidity']} %\n\n"
                "Anomaly history:\n" + history_text + "\n\n"
                "This alert is triggered ONLY once per anomaly id."
            )
            attachments = images if include_graphs_in_alert else None
            sent = False
            try:
                sent = send_email_with_attachments(subject, message, attachments=attachments, receiver_emails=DEFAULT_RECEIVER)
            except Exception as e:
                st.error(f"Auto alert send failed: {e}")
            if sent:
                st.session_state["last_alert_sent_id"] = latest_anom_id
                st.success("Automatic anomaly alert sent.")
            else:
                st.warning("Automatic alert failed to send (check SMTP settings).")

# ============================
# Manual Alert UI
# ============================
st.markdown("---")
st.subheader("Manual Alert (Send to any email)")

# Contacts & quick-add
search_query = st.text_input("Search contacts (type email to filter)", key="contact_search")
if search_query:
    filtered = [c for c in st.session_state["contacts"] if search_query.lower() in c["email"].lower()]
else:
    filtered = st.session_state["contacts"]

st.write("**Contacts** — click a button to add to recipient field")
if filtered:
    cols = st.columns(min(len(filtered), 4))
    for i, c in enumerate(filtered):
        with cols[i % len(cols)]:
            if st.button(f"Add {c['name']}", key=f"add_contact_{i}"):
                if "manual_recipients" not in st.session_state or not st.session_state["manual_recipients"]:
                    st.session_state["manual_recipients"] = c["email"]
                else:
                    existing = [e.strip() for e in st.session_state["manual_recipients"].split(",") if e.strip()]
                    if c["email"] not in existing:
                        existing.append(c["email"])
                        st.session_state["manual_recipients"] = ", ".join(existing)
                st.success(f"Added {c['email']} to recipients")
else:
    st.info("No contacts match. You can add a new email below.")

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
            st.success("Saved new contact.")

if "manual_recipients" not in st.session_state:
    st.session_state["manual_recipients"] = ""

recipient_input = st.text_input(
    "Recipient email(s) (comma-separated). You can add via contact buttons above.",
    value=st.session_state["manual_recipients"],
    key="recipient_input"
)
include_graphs_manual = st.checkbox("Include graphs/attachments in manual alert", value=True, key="include_graphs_manual")

if st.button("Send Manual Alert"):
    if not is_valid_email_list(recipient_input):
        st.error("Please provide at least one valid recipient email.")
    else:
        anoms = scored_df[scored_df["is_anomaly"] == 1] if not scored_df.empty else pd.DataFrame()
        if not anoms.empty:
            latest_anom = anoms.sort_values("ts").iloc[-1]
            history_lines = []
            for _, r in anoms.sort_values("ts").iterrows():
                history_lines.append(f"{r.get('id','')} | {r['ts']} | Temp: {r['temperature']}°C | Hum: {r['humidity']}% | Score: {r['anomaly_score']:.4f}")
            history_text = "\n".join(history_lines)
            subject = "MANUAL ALERT — AI IoT – Anomaly Detected"
            message = (
                "MANUAL ALERT — AI IoT Forensics\n\n"
                f"Anomaly Detected\nTimestamp : {latest_anom['ts']}\n"
                f"Anomaly Score : {latest_anom['anomaly_score']:.4f}\n\n"
                f"Sensor Values at Anomaly\nTemperature : {latest_anom['temperature']} °C\n"
                f"Humidity    : {latest_anom['humidity']} %\n\n"
                f"Current Live Sensor Values\nLive Temperature : {latest['temperature']} °C\n"
                f"Live Humidity    : {latest['humidity']} %\n\n"
                "Full anomaly history:\n" + history_text + "\n"
            )
        else:
            subject = "MANUAL ALERT — AI IoT – Status Update (no anomaly)"
            message = (
                "MANUAL ALERT — AI IoT Forensics\n\n"
                "No anomalies detected in current window.\n"
                f"Live Temperature : {latest['temperature']} °C\n"
                f"Live Humidity    : {latest['humidity']} %\n"
            )

        attachments = images if include_graphs_manual else None
        sent = send_email_with_attachments(subject, message, attachments=attachments, receiver_emails=recipient_input)
        if sent:
            st.success(f"Manual alert sent to: {recipient_input}")
        else:
            st.error("Failed to send manual alert. Check SMTP settings and credentials.")

# ============================
# Reporting & Downloads
# ============================
st.markdown("---")
st.subheader("Reports & Downloads")

# CSV download of scored data
csv_buf = io.StringIO()
scored_df.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode()

b64_csv = base64.b64encode(csv_bytes).decode()
st.markdown(f"[Download scored CSV](data:file/csv;base64,{b64_csv})", unsafe_allow_html=True)

# Zip of images
try:
    zip_images = images_to_zip_bytes(images)
    b64_zip = base64.b64encode(zip_images).decode()
    st.markdown(f"[Download diagnostic images (ZIP)](data:application/zip;base64,{b64_zip})", unsafe_allow_html=True)
except Exception:
    st.info("Image ZIP unavailable.")

# PDF report
if REPORTLAB_AVAILABLE:
    pdf_bytes = create_pdf_report(scored_df, df_raw, images, report_title="OT-IoT Threat Report")
    if pdf_bytes:
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        st.markdown(f"[Download PDF report](data:application/pdf;base64,{b64_pdf})", unsafe_allow_html=True)
    else:
        st.info("PDF report generation failed.")
else:
    st.info("PDF report (reportlab) not available in environment. Enable reportlab to create PDFs.")

# ============================
# Footer notes
# ============================
st.markdown("---")
st.caption("Automatic alerts are sent only once per anomaly id and include charts when available.")
st.info("Theme: Futuristic Dark Blue IoT · Animations & wallpaper added. Modify CSS in inject_theme_css() to adjust visuals.")

# Optional: small debug / state panel for advanced users
with st.expander("🔧 Debug / Session State"):
    st.write("Model trained at:", st.session_state["model_state"].get("trained_at"))
    st.write("Last alert sent id:", st.session_state.get("last_alert_sent_id"))
    st.write("Contacts:", st.session_state["contacts"])
    st.write("Session keys:", list(st.session_state.keys()))
