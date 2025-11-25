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
import re
# ===================================

# ============================================================
# EMAIL ALERT — HARDCODED CREDENTIALS (leave as-is or edit)
# ============================================================
SENDER_EMAIL = "manas.dfis242604@nfsu.ac.in"
APP_PASSWORD = "euozfdlazplbmtkd"              # Google App Password
# default receiver (used for auto-alerts if no manual recipient provided)
RECEIVER_EMAIL = "manas.dfis242604@nfsu.ac.in"


# ============================================================
# EMAIL SENDING FUNCTION (single-email behavior with optional graphs)
# ============================================================
def _normalize_recipient_input(receiver_emails):
    if receiver_emails is None:
        return [RECEIVER_EMAIL]
    if isinstance(receiver_emails, str):
        emails = [e.strip() for e in receiver_emails.split(",") if e.strip()]
        return emails if emails else [RECEIVER_EMAIL]
    if isinstance(receiver_emails, list):
        emails = [e.strip() for e in receiver_emails if e.strip()]
        return emails if emails else [RECEIVER_EMAIL]
    return [RECEIVER_EMAIL]


def send_email_alert(subject, message, attachments=None, receiver_emails=None):
    """
    Send a single email (plain text body + optional image attachments).
    attachments: list of (filename, bytes)
    receiver_emails: None | str (comma separated) | list
    Returns True on success, False on failure.
    """
    receiver_list = _normalize_recipient_input(receiver_emails)

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ", ".join(receiver_list)
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        # Attach images (if provided)
        if attachments:
            for fname, data in attachments:
                try:
                    img = MIMEImage(data)
                    img.add_header('Content-Disposition', 'attachment', filename=fname)
                    msg.attach(img)
                except Exception as e:
                    print(f"Warning: failed to attach {fname}: {e}")

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_list, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Email failed: {e}")
        return False


# ============================================================
# ADDITIONAL HELPERS — create plots/images and return list of (filename, bytes)
# ============================================================
def create_graph_images(scored_df, df_feat, features):
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


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="OT-IoT Threat Monitoring Console", layout="wide")
st.title("OT-IoT Threat Monitoring Console")
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
if st.button("Retrain Model Live 🔁"):
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

st.subheader("System Summary")
m1, m2 = st.columns(2)

m1.metric("Total Events Received", total_events)
m2.metric("Total Anomalies Detected", int(total_anomalies))


# ============================================================
# AUTOMATIC EMAIL ALERT (for newest anomaly) — SINGLE EMAIL ONLY
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
   AI IoT Forensics Alert
===============================

📌 Anomaly Detected
Timestamp : {latest_anomaly['ts']}
Anomaly Score : {latest_anomaly['anomaly_score']:.4f}

📡 Sensor Values at Anomaly
Temperature : {latest_anomaly['temperature']} °C
Humidity    : {latest_anomaly['humidity']} %

📡 Current Live Sensor Values
Live Temperature : {latest['temperature']} °C
Live Humidity    : {latest['humidity']} %

Please investigate the IoT device immediately.
"""

        # create attachments and send single email (graphs attached)
        attachments = []
        try:
            attachments = create_graph_images(scored_df, df_feat, features)
        except Exception as e:
            print("Failed to create attachments:", e)

        try:
            ok = send_email_alert(subject, message, attachments=attachments, receiver_emails=RECEIVER_EMAIL)
            if ok:
                st.session_state["last_alert_sent_id"] = last_anom_id
                st.success("Automatic alert sent for latest anomaly ")
            else:
                st.error("Failed to send automatic alert. See logs.")
        except Exception as e:
            st.error(f"Automatic alert failed: {e}")


# ============================================================
# MANUAL ALERT UI — simplified contacts (only Me + Add New Email + 3 quick buttons)
# ============================================================
st.markdown("---")
st.subheader("📤 Manual Alert (send to any email)")

# initialize contacts in session state (only Me by default)
if "contacts" not in st.session_state:
    st.session_state["contacts"] = [
        {"name": "Me (Manas)", "email": "manas.dfis242604@nfsu.ac.in"},
    ]

# Ensure quick-contact buttons (three additional addresses) exist in a separate list
QUICK_CONTACTS = [
    {"name": "Nandini", "email": "nandini.dfis242606@nfsu.ac.in"},
    {"name": "Jayendra", "email": "jayendra.dfis242605@nfsu.ac.in"},
    {"name": "Ujjaval", "email": "ujjaval.patel@nfsu.ac.in"},
]

# contact search (filters only "Me" plus any added emails)
search_query = st.text_input("Search contacts (type email to filter)")

filtered = []
if search_query:
    q = search_query.lower()
    filtered = [c for c in st.session_state["contacts"] if q in c["email"].lower()]
else:
    filtered = st.session_state["contacts"]

st.write("**Contacts** — click a button to add to recipient field")
# display contacts as buttons (columns)
if filtered:
    cols = st.columns(min(len(filtered), 4))
    for i, c in enumerate(filtered):
        with cols[i % len(cols)]:
            if st.button(f"Add {c['email']}", key=f"add_contact_{i}"):
                # append to recipient field stored in session_state
                if "manual_recipients" not in st.session_state:
                    st.session_state["manual_recipients"] = c["email"]
                else:
                    existing = st.session_state["manual_recipients"]
                    # avoid duplicates
                    emails = [e.strip() for e in existing.split(",") if e.strip()]
                    if c["email"] not in emails:
                        emails.append(c["email"])
                        st.session_state["manual_recipients"] = ", ".join(emails)
                st.success(f"Added {c['email']} to recipients")
else:
    st.info("No contacts match. You can add a new email below.")

# Quick contact buttons (the 3 requested emails) — placed prominently
st.write("**Quick Contacts** — one-click add")
qcols = st.columns(len(QUICK_CONTACTS))
for i, qc in enumerate(QUICK_CONTACTS):
    with qcols[i]:
        if st.button(f"Add {qc['name']}", key=f"quick_add_{i}"):
            if "manual_recipients" not in st.session_state:
                st.session_state["manual_recipients"] = qc["email"]
            else:
                existing = st.session_state["manual_recipients"]
                emails = [e.strip() for e in existing.split(",") if e.strip()]
                if qc["email"] not in emails:
                    emails.append(qc["email"])
                    st.session_state["manual_recipients"] = ", ".join(emails)
            # also ensure contact list contains it for future search
            if not any(c.get("email") == qc["email"] for c in st.session_state["contacts"]):
                st.session_state["contacts"].append({"name": qc["name"], "email": qc["email"]})
            st.success(f"Added {qc['email']} to recipients")

# allow adding a new email only (no name/company)
with st.expander("➕ Add new email"):
    new_email = st.text_input("Email address", key="new_contact_email_only")
    if st.button("Save email"):
        # basic email validation
        def is_valid_email(email):
            pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
            return re.match(pattern, email) is not None

        if not new_email:
            st.error("Email is required.")
        elif not is_valid_email(new_email):
            st.error("Invalid email format.")
        else:
            # store contact with name same as email for simplicity
            st.session_state["contacts"].append({"name": new_email, "email": new_email})
            st.success("Email saved. Use the search box or Add button to include it as recipient.")
            # prefill manual_recipients with the new email
            st.session_state["manual_recipients"] = new_email

# recipient input (allows comma-separated list)
if "manual_recipients" not in st.session_state:
    st.session_state["manual_recipients"] = ""

recipient_input = st.text_input("Recipient email(s) (comma-separated). You can add via contact buttons above.", value=st.session_state["manual_recipients"])

include_graphs_manual = st.checkbox("Include graphs/attachments", value=True)


def is_valid_email_list(s):
    if not s:
        return False
    emails = [e.strip() for e in s.split(",") if e.strip()]
    if not emails:
        return False
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return all(re.match(pattern, e) for e in emails)

# Prepare manual alert message (full detail -- same as automatic)
if st.button("Send Manual Alert"):
    if not is_valid_email_list(recipient_input):
        st.error("Please provide at least one valid recipient email (comma-separated).")
    else:
        # choose the most recent anomaly if exists, else send last sample as info
        if not latest_anomaly_rows.empty:
            latest_anomaly = latest_anomaly_rows.iloc[-1]
            latest = df_raw.iloc[-1]
            subject = "MANUAL ALERT — AI IoT – Anomaly Detected"
            message = f"""
MANUAL ALERT — AI IoT Forensics

Anomaly Detected
Timestamp : {latest_anomaly['ts']}
Anomaly Score : {latest_anomaly['anomaly_score']:.4f}

Sensor Values at Anomaly
Temperature : {latest_anomaly['temperature']} °C
Humidity    : {latest_anomaly['humidity']} %

Current Live Sensor Values
Live Temperature : {latest['temperature']} °C
Live Humidity    : {latest['humidity']} %

Note: This manual alert was triggered from the Streamlit console.
"""
        else:
            latest = df_raw.iloc[-1]
            subject = "MANUAL ALERT — AI IoT – Status Update (no anomaly)"
            message = f"""
MANUAL ALERT — AI IoT Forensics

No anomalies detected in current window.
Current Live Sensor Values
Live Temperature : {latest['temperature']} °C
Live Humidity    : {latest['humidity']} %

Note: This manual alert was triggered from the Streamlit console.
"""

        # Attempt to send using single-email sender (with graphs if requested)
        attachments = []
        if include_graphs_manual:
            try:
                attachments = create_graph_images(scored_df, df_feat, features)
            except Exception as e:
                print("Failed to create attachments for manual send:", e)

        try:
            ok = send_email_alert(subject, message, attachments=attachments if include_graphs_manual else None, receiver_emails=recipient_input)
            if ok:
                st.success(f"Manual alert sent to: {recipient_input}")
            else:
                st.error("Failed to send manual alert. See logs.")
        except Exception as e:
            st.error(f"Failed to send manual alert: {e}")


# ============================================================
# LIVE SENSOR VALUES
# ============================================================
latest = df_raw.iloc[-1]

st.subheader("Live Sensor Status")
c1, c2, c3 = st.columns(3)

c1.metric("Temperature", f"{latest['temperature']:.2f} °C")
c2.metric("Humidity", f"{latest['humidity']:.2f} %")
c3.metric("Last Update", latest["ts"].strftime("%Y-%m-%d %H:%M:%S"))

st.markdown("---")


# ============================================================
# CHARTS — Temperature + Humidity with Anomalies
# ============================================================
st.subheader("Temperature (with anomalies)")
base = alt.Chart(scored_df).encode(x="ts:T")

st.altair_chart(
    base.mark_line().encode(y="temperature:Q") +
    base.transform_filter("datum.is_anomaly == 1")
    .mark_circle(color="red", size=70).encode(y="temperature:Q"),
    use_container_width=True
)

st.subheader("Humidity (with anomalies)")
st.altair_chart(
    base.mark_line(color="green").encode(y="humidity:Q") +
    base.transform_filter("datum.is_anomaly == 1")
    .mark_circle(color="red", size=70).encode(y="humidity:Q"),
    use_container_width=True
)


# ============================================================
# PCA ANOMALY MAP
# ============================================================
st.subheader("PCA Anomaly Map")
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
st.subheader("Anomaly Table")
st.dataframe(scored_df.sort_values("anomaly_score", ascending=False))

# ============================================================
# ========== NEW: Additional OT-style Visuals and Graphs ==========
# ============================================================
st.markdown("---")
st.subheader("OT Ladder & Additional Diagnostics (added)")

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
    st.subheader("Anomalies Over Time (Altair)")
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
    st.subheader("OT-style Quick Diagnostics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly Rate (window)", f"{(scored_df['is_anomaly'].mean()*100):.2f}%")
    col2.metric("Latest Anomaly Score", f"{scored_df['anomaly_score'].max():.4f}")
    col3.metric("Events in window", len(scored_df))
except Exception as e:
    st.warning("Could not compute diagnostics: " + str(e))

st.markdown("---")
st.caption("Added visuals include rolling stats, anomaly histograms by hour, cumulative curve, and a simplified ladder-logic diagram to give an OT flavor. Graphs are attached automatically to alerts when requested.")
# End of file
