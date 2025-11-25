# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import altair as alt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

# ============================================================
# CONFIG - edit if needed
# ============================================================
SENDER_EMAIL = "manas.dfis242604@nfsu.ac.in"
APP_PASSWORD = "euozfdlazplbmtkd"   # Google App Password
DEFAULT_RECEIVER = "manas.dfis242604@nfsu.ac.in"

FIREBASE_URL = (
    "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/"
    "forensics_logs.json"
)

# ============================================================
# EMAIL HELPERS
# ============================================================
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


def send_email_with_attachments(subject, message, attachments=None, receiver_emails=None):
    """
    Send single email with optional attachments.
    attachments: list of (filename, bytes)
    receiver_emails: None | str (comma separated) | list
    """
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
                    img = MIMEImage(b)
                    img.add_header("Content-Disposition", "attachment", filename=fname)
                    msg.attach(img)
                except Exception as e:
                    print("Attach failed:", fname, e)

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, receivers, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email send error: {e}")
        return False

# ============================================================
# PLOT / IMAGE CREATION
# ============================================================
def create_graph_images(scored_df, df_feat, features):
    images = []

    # 1) Temp & Hum with anomalies
    try:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(scored_df["ts"], scored_df["temperature"], label="Temperature", linewidth=1.5)
        ax1.set_ylabel("Temperature (°C)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2 = ax1.twinx()
        ax2.plot(scored_df["ts"], scored_df["humidity"], label="Humidity", linewidth=1.0, linestyle="--")
        ax2.set_ylabel("Humidity (%)")

        anomalies = scored_df[scored_df["is_anomaly"] == 1]
        if not anomalies.empty:
            ax1.scatter(anomalies["ts"], anomalies["temperature"], edgecolors="r", facecolors="none", s=80, label="Anom Temp")
            ax2.scatter(anomalies["ts"], anomalies["humidity"], color="red", marker="x", s=60, label="Anom Hum")

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        images.append(("temp_hum_with_anomalies.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("temp_hum plot error:", e)

    # 2) Rolling means
    try:
        win = 5
        tmp = df_feat.set_index("ts")
        rtemp = tmp["temperature"].rolling(win).mean()
        rhum = tmp["humidity"].rolling(win).mean()

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(df_feat["ts"], df_feat["temperature"], alpha=0.4, label="Temp raw")
        axs[0].plot(rtemp.index, rtemp.values, linewidth=2, label=f"{win}-pt MA")
        axs[0].set_ylabel("Temperature (°C)")
        axs[0].legend()

        axs[1].plot(df_feat["ts"], df_feat["humidity"], alpha=0.4, label="Hum raw")
        axs[1].plot(rhum.index, rhum.values, linewidth=2, label=f"{win}-pt MA")
        axs[1].set_ylabel("Humidity (%)")
        axs[1].legend()

        fig.autofmt_xdate()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        images.append(("rolling_stats.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("rolling plot error:", e)

    # 3) Anomalies by hour
    try:
        if "hour" not in scored_df.columns:
            scored_df["hour"] = scored_df["ts"].dt.hour
        byhour = scored_df.groupby("hour")["is_anomaly"].sum().reindex(range(24), fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(byhour.index, byhour.values)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Anomalies Count")
        ax.set_title("Anomalies by Hour (last window)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        images.append(("anomalies_by_hour.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("byhour plot error:", e)

    # 4) Cumulative curve
    try:
        ordered = scored_df.sort_values("ts").copy()
        ordered["cum_anom"] = ordered["is_anomaly"].cumsum()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ordered["ts"], ordered["cum_anom"], marker="o")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Anomalies")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        images.append(("cumulative_anomalies.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("cumulative plot error:", e)

    # 5) Ladder logic (simple)
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        rung_y = [9, 7, 5, 3]
        for y in rung_y:
            ax.hlines(y, 1, 9, linewidth=3, color="black")
        ax.vlines(1, 2, 10, linewidth=4)
        ax.vlines(9, 2, 10, linewidth=4)
        ax.text(2, 8.6, "I: Sensor OK", fontsize=10)
        ax.text(2, 6.6, "I: Manual Stop", fontsize=10)
        ax.text(5, 4.6, "M: Safety Interlock", fontsize=10)
        ax.text(6.5, 2.6, "Q: Alarm Output", fontsize=10, color="red", fontweight="bold")
        ax.text(4.5, 1.2, "OT Ladder (Simplified)", fontsize=9, ha="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        images.append(("ladder_logic.png", buf.read()))
        plt.close(fig)
    except Exception as e:
        print("ladder plot error:", e)

    return images

# ============================================================
# DATA FETCH + FEATURE ENGINEERING
# ============================================================
@st.cache_data(ttl=10)
def fetch_raw_data(url):
    """
    Fetch JSON from Firebase realtime DB endpoint (assumes structure
    { record_id: { timestamp: <epoch>, temperature: .., humidity: .. }, ... })
    Returns a DataFrame with columns: id, ts (datetime), temperature, humidity
    """
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as e:
        print("fetch error:", e)
        return pd.DataFrame(columns=["id", "ts", "temperature", "humidity"])

    if not data:
        return pd.DataFrame(columns=["id", "ts", "temperature", "humidity"])

    rows = []
    for rid, v in data.items():
        if not isinstance(v, dict):
            continue
        ts = v.get("timestamp")
        if ts is None:
            continue
        try:
            dt = datetime.utcfromtimestamp(int(ts))
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
    Keep id column intact. df MUST contain columns: id, ts, temperature, humidity
    """
    if df.empty:
        return df.copy()

    tmp = df.copy().set_index("ts")
    # preserve id
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
    # ensure id column exists
    if "id" not in out.columns:
        out["id"] = np.nan
    return out

# ============================================================
# MODEL - train and score
# ============================================================
def train_and_score(df_feat, model_type="iforest", contamination=0.02, features=None):
    if df_feat.empty:
        return df_feat.copy(), None, None

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
# STREAMLIT PAGE
# ============================================================
st.set_page_config(page_title="OT-IoT Threat Monitoring Console", layout="wide")
st.title("OT-IoT Threat Monitoring Console")
st.caption("ESP8266 + Firebase + AI + Streamlit + Email Alerts")

# Load data
df_raw = fetch_raw_data(FIREBASE_URL)

if df_raw.empty:
    st.warning("⚠ Waiting for sensor data (no records found).")
    st.stop()

# window = last 12 hours
start_dt = df_raw["ts"].max() - timedelta(hours=12)
df_window = df_raw[df_raw["ts"] >= start_dt].copy()

df_feat = feature_engineer(df_window)
features = ["temperature", "humidity", "temp_diff", "hum_diff", "temp_z", "hum_z", "hour"]

# retrain button (clears caches to force retrain)
if st.button("Retrain Model Live"):
    st.cache_resource.clear()
    st.success("Model retrained using latest data!")

scored_df, model, scaler = train_and_score(df_feat, "iforest", contamination=0.02, features=features)

# show summary
st.subheader("System Summary")
m1, m2 = st.columns(2)
m1.metric("Total Events Received", len(df_raw))
m2.metric("Total Anomalies Detected", int(scored_df["is_anomaly"].sum() if not scored_df.empty else 0))

# ============================================================
# LIVE SENSOR STATUS & CHARTS
# ============================================================
latest = df_raw.iloc[-1]
st.subheader("Live Sensor Status")
c1, c2, c3 = st.columns(3)
c1.metric("Temperature", f"{latest['temperature']:.2f} °C")
c2.metric("Humidity", f"{latest['humidity']:.2f} %")
c3.metric("Last Update", latest["ts"].strftime("%Y-%m-%d %H:%M:%S"))

# ============================================================
# AUTOMATIC ALERT (Option A) - one email per NEW anomaly id
# ============================================================
if not scored_df.empty:
    # ensure id exists in scored_df (should be preserved via feature_engineer)
    if "id" not in scored_df.columns:
        # try to merge from df_window by ts
        scored_df = scored_df.merge(df_window[["ts", "id"]], on="ts", how="left")

    latest_anomaly_rows = scored_df[scored_df["is_anomaly"] == 1].copy()

    if not latest_anomaly_rows.empty:
        # newest anomaly (last by timestamp)
        latest_row = latest_anomaly_rows.sort_values("ts").iloc[-1]
        latest_anom_id = str(latest_row.get("id", latest_row["ts"]))

        # initialize session store
        if "last_alert_sent_id" not in st.session_state:
            st.session_state["last_alert_sent_id"] = None

        # only send when anomaly id differs from last sent id
        if st.session_state["last_alert_sent_id"] != latest_anom_id:
            # build history text
            history_lines = []
            for _, r in latest_anomaly_rows.sort_values("ts").iterrows():
                rid = r.get("id", "")
                history_lines.append(
                    f"{rid} | {r['ts']} | Temp: {r['temperature']}°C | Hum: {r['humidity']}% | Score: {r['anomaly_score']:.4f}"
                )
            history_text = "Previous Anomalies (History):\n" + "\n".join(history_lines)

            # Compose message
            subject = "AI IoT ALERT — New Anomaly Detected"
            message = (
                "===============================\n"
                "      AI IoT Forensics Alert\n"
                "===============================\n\n"
                "A NEW anomaly has been detected.\n\n"
                f"Latest Anomaly (id={latest_anom_id})\n"
                f"Timestamp : {latest_row['ts']}\n"
                f"Score     : {latest_row['anomaly_score']:.4f}\n"
                f"Temp      : {latest_row['temperature']} °C\n"
                f"Humidity  : {latest_row['humidity']} %\n\n"
                "📡 Current Sensor Status (latest sample)\n"
                f"Temperature : {df_raw.iloc[-1]['temperature']} °C\n"
                f"Humidity    : {df_raw.iloc[-1]['humidity']} %\n\n"
                f"{history_text}\n\n"
                "This alert is triggered ONLY once per anomaly id.\n"
            )

            # create attachments and send
            attachments = create_graph_images(scored_df, df_feat, features)
            sent = send_email_with_attachments(subject, message, attachments=attachments, receiver_emails=DEFAULT_RECEIVER)
            if sent:
                st.session_state["last_alert_sent_id"] = latest_anom_id
                st.success("New anomaly alert sent (with graphs).")
            else:
                st.error("Automatic alert failed to send.")

# ============================================================
# MANUAL ALERT UI - contacts + quick-add + send
# ============================================================
st.markdown("---")
st.subheader("Manual Alert (send to any email)")

# initialize contacts
if "contacts" not in st.session_state:
    st.session_state["contacts"] = [{"name": "Me", "email": DEFAULT_RECEIVER}]

# search/filter
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
            st.success("Saved new contact.")

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

if st.button("Send Manual Alert"):
    if not is_valid_email_list(recipient_input):
        st.error("Please provide at least one valid recipient email.")
    else:
        # prepare message — include all anomalies history
        anoms = scored_df[scored_df["is_anomaly"] == 1] if not scored_df.empty else pd.DataFrame()
        if not anoms.empty:
            # newest anomaly
            latest_anom = anoms.sort_values("ts").iloc[-1]
            history_lines = []
            for _, r in anoms.sort_values("ts").iterrows():
                history_lines.append(
                    f"{r.get('id','')} | {r['ts']} | Temp: {r['temperature']}°C | Hum: {r['humidity']}% | Score: {r['anomaly_score']:.4f}"
                )
            history_text = "\n".join(history_lines)
            subject = "MANUAL ALERT — AI IoT – Anomaly Detected"
            message = (
                "MANUAL ALERT — AI IoT Forensics\n\n"
                f"Anomaly Detected\nTimestamp : {latest_anom['ts']}\n"
                f"Anomaly Score : {latest_anom['anomaly_score']:.4f}\n\n"
                f"Sensor Values at Anomaly\nTemperature : {latest_anom['temperature']} °C\n"
                f"Humidity    : {latest_anom['humidity']} %\n\n"
                f"Current Live Sensor Values\nLive Temperature : {df_raw.iloc[-1]['temperature']} °C\n"
                f"Live Humidity    : {df_raw.iloc[-1]['humidity']} %\n\n"
                "Full anomaly history:\n"
                f"{history_text}\n"
            )
        else:
            subject = "MANUAL ALERT — AI IoT – Status Update (no anomaly)"
            message = (
                "MANUAL ALERT — AI IoT Forensics\n\n"
                "No anomalies detected in current window.\n"
                f"Live Temperature : {df_raw.iloc[-1]['temperature']} °C\n"
                f"Live Humidity    : {df_raw.iloc[-1]['humidity']} %\n"
            )

        attachments = create_graph_images(scored_df, df_feat, features) if include_graphs_manual else None
        sent = send_email_with_attachments(subject, message, attachments=attachments, receiver_emails=recipient_input)
        if sent:
            st.success(f"Manual alert sent to: {recipient_input}")
        else:
            st.error("Failed to send manual alert.")

# Temperature chart
st.subheader("Temperature (with anomalies)")
base = alt.Chart(scored_df).encode(x="ts:T")
st.altair_chart(
    base.mark_line().encode(y="temperature:Q") +
    base.transform_filter("datum.is_anomaly == 1")
    .mark_circle(color="red", size=70).encode(y="temperature:Q"),
    use_container_width=True
)

# Humidity chart
st.subheader("Humidity (with anomalies)")
st.altair_chart(
    base.mark_line().encode(y="humidity:Q") +
    base.transform_filter("datum.is_anomaly == 1")
    .mark_circle(color="red", size=70).encode(y="humidity:Q"),
    use_container_width=True
)

# PCA map (if available)
st.subheader("PCA Anomaly Map")
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
st.subheader("Anomaly Table")
st.dataframe(scored_df.sort_values("anomaly_score", ascending=False))

# OT visuals (images created earlier)
st.markdown("---")
st.subheader("OT Ladder & Additional Diagnostics (added)")
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
st.subheader("Quick Diagnostics")
col1, col2, col3 = st.columns(3)
col1.metric("Anomaly Rate (window)", f"{(scored_df['is_anomaly'].mean()*100):.2f}%")
col2.metric("Latest Anomaly Score", f"{scored_df['anomaly_score'].max():.4f}")
col3.metric("Events in window", len(scored_df))

st.markdown("---")
st.caption("Automatic alerts are sent only once per anomaly id and include charts when available.")
