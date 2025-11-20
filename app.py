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

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="ESP8266 IoT Forensics — AI Anomaly Detection", layout="wide")
st.title("🔎 IoT Forensics — AI Anomaly Detection")
st.caption("ESP8266 + DHT11 → Firebase → Streamlit (IsolationForest + LOF)")

# -------------------------
# FIREBASE URL — your node
# -------------------------
FIREBASE_URL = (
    "https://iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app/"
    "forensics_logs.json"
)

# -------------------------
# UTILS
# -------------------------
@st.cache_data(ttl=60)
def fetch_raw_data(firebase_url: str) -> pd.DataFrame:
    """Fetch JSON from Firebase and return cleaned DataFrame."""
    try:
        res = requests.get(firebase_url, timeout=10)
        data = res.json()
    except Exception as e:
        st.error(f"Error reading Firebase: {e}")
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
        try:
            ts_dt = datetime.utcfromtimestamp(int(ts))
        except Exception:
            continue
        rows.append({
            "id": key,
            "ts": ts_dt,
            "temperature": float(val.get("temperature")) if val.get("temperature") is not None else np.nan,
            "humidity": float(val.get("humidity")) if val.get("humidity") is not None else np.nan,
            "anomaly_tag": val.get("anomaly", "unknown")
        })

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def feature_engineer(df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Add time-series features: diffs, rolling mean/std, z-scores."""
    tmp = df.copy().set_index("ts")
    # forward/backfill small gaps
    tmp["temperature"] = tmp["temperature"].interpolate(limit=3).ffill().bfill()
    tmp["humidity"] = tmp["humidity"].interpolate(limit=3).ffill().bfill()

    # diffs
    tmp["temp_diff"] = tmp["temperature"].diff().fillna(0)
    tmp["hum_diff"] = tmp["humidity"].diff().fillna(0)

    # rolling stats
    tmp["temp_ma"] = tmp["temperature"].rolling(window=rolling_window, min_periods=1).mean()
    tmp["hum_ma"] = tmp["humidity"].rolling(window=rolling_window, min_periods=1).mean()
    tmp["temp_std"] = tmp["temperature"].rolling(window=rolling_window, min_periods=1).std().fillna(0)
    tmp["hum_std"] = tmp["humidity"].rolling(window=rolling_window, min_periods=1).std().fillna(0)

    # z-scores relative to rolling window (avoid divide-by-zero)
    tmp["temp_z"] = (tmp["temperature"] - tmp["temp_ma"]) / (tmp["temp_std"].replace(0, np.nan))
    tmp["hum_z"] = (tmp["humidity"] - tmp["hum_ma"]) / (tmp["hum_std"].replace(0, np.nan))
    tmp["temp_z"] = tmp["temp_z"].fillna(0)
    tmp["hum_z"] = tmp["hum_z"].fillna(0)

    # hour of day — useful for patterns
    tmp["hour"] = tmp.index.hour

    # reset index
    out = tmp.reset_index()
    return out

def prepare_features(df_feat: pd.DataFrame, features: list):
    X = df_feat[features].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

# -------------------------
# MODEL TRAIN / PREDICT
# -------------------------
@st.cache_resource
def build_isolation_forest(random_state=42):
    return IsolationForest(n_estimators=200, max_samples="auto", contamination=0.01, random_state=random_state, n_jobs=-1)

@st.cache_resource
def build_lof(n_neighbors=20):
    return LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.01)

def train_and_score(df_feat: pd.DataFrame, model_type: str = "iforest", contamination: float = 0.01,
                    features: list = None, random_state: int = 42):
    """Train selected model and return df with anomaly score and label."""
    if features is None:
        raise ValueError("features must be provided")

    X, scaler = prepare_features(df_feat, features)

    # choose model
    if model_type == "iforest":
        model = IsolationForest(n_estimators=200, contamination=contamination,
                                random_state=random_state, n_jobs=-1)
        model.fit(X)
        raw_scores = model.decision_function(X)  # higher = more normal
        # Convert to anomaly score where higher = more anomalous
        anomaly_score = -raw_scores
        label = model.predict(X)  # 1 normal, -1 outlier
        is_anomaly = (label == -1).astype(int)
    elif model_type == "lof":
        model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination)
        model.fit(X)
        raw_scores = model.decision_function(X)
        anomaly_score = -raw_scores
        preds = model.predict(X)
        is_anomaly = (preds == -1).astype(int)
    else:
        raise ValueError("unknown model_type")

    df_out = df_feat.copy().reset_index(drop=True)
    df_out["anomaly_score"] = anomaly_score
    df_out["is_anomaly"] = is_anomaly
    return df_out, model, scaler

# -------------------------
# UI: Side controls
# -------------------------
with st.sidebar:
    st.header("Model & UI Controls")
    st.markdown("**Data / Training window**")
    df_raw = fetch_raw_data(FIREBASE_URL)
    if df_raw.empty:
        st.warning("No data found in Firebase. Wait for ESP8266 to send logs.")
        st.stop()

    # Time window selection
    max_time = df_raw["ts"].max()
    min_time = df_raw["ts"].min()
    default_start = max_time - timedelta(hours=12) if (max_time - min_time) > timedelta(hours=12) else min_time
    start = st.date_input("Training start date", value=default_start.date(), key="start_date")
    end = st.date_input("Training end date", value=max_time.date(), key="end_date")
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    st.markdown("---")
    st.markdown("**Feature engineering**")
    rolling_window = st.slider("Rolling window (samples)", min_value=1, max_value=30, value=5)
    st.markdown("---")
    st.markdown("**Model**")
    model_choice = st.selectbox("Select model", options=["iforest", "lof"], index=0)
    contamination = st.slider("Contamination (expected fraction of anomalies)", 0.001, 0.2, 0.02, step=0.001)
    st.markdown("---")
    st.markdown("**Misc**")
    retrain_btn = st.button("🔁 Train model")
    download_model_btn = st.button("💾 Download trained model (joblib)")

# -------------------------
# PREPARE DATA
# -------------------------
# filter by chosen time window
mask = (df_raw["ts"] >= start_dt) & (df_raw["ts"] <= end_dt)
df_window = df_raw.loc[mask].copy()
if df_window.empty:
    st.warning("No data in the selected time window. Adjust the dates.")
    st.stop()

df_feat = feature_engineer(df_window, rolling_window=rolling_window)

# choose features for the model (tunable)
candidate_features = ["temperature", "humidity", "temp_diff", "hum_diff", "temp_ma", "hum_ma", "temp_z", "hum_z", "hour"]
selected_features = st.multiselect("Select features to use (recommended defaults checked)", 
                                   options=candidate_features,
                                   default=["temperature", "humidity", "temp_diff", "temp_z", "hum_z"])

st.markdown("---")
st.write(f"Data loaded: {len(df_raw)} total rows. Using {len(df_feat)} rows in the selected window.")

# -------------------------
# TRAIN / SCORE (trigger)
# -------------------------
train_needed = retrain_btn or ("model_state" not in st.session_state)

if train_needed:
    with st.spinner("Training model..."):
        try:
            scored_df, trained_model, scaler = train_and_score(df_feat, model_type=model_choice, 
                                                               contamination=contamination, features=selected_features)
            st.session_state["model"] = trained_model
            st.session_state["scaler"] = scaler
            st.session_state["scored_df"] = scored_df
            st.success("Model trained.")
        except Exception as e:
            st.error(f"Training error: {e}")
            st.stop()
else:
    # if not retraining, try to use cached session model
    scored_df = st.session_state.get("scored_df")
    trained_model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    if scored_df is None:
        with st.spinner("Training initial model..."):
            scored_df, trained_model, scaler = train_and_score(df_feat, model_type=model_choice,
                                                               contamination=contamination, features=selected_features)
            st.session_state["model"] = trained_model
            st.session_state["scaler"] = scaler
            st.session_state["scored_df"] = scored_df

# -------------------------
# SHOW METRICS
# -------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows in window", len(df_feat))
with col2:
    n_anom = int(scored_df["is_anomaly"].sum())
    st.metric("Detected anomalies", n_anom)
with col3:
    median_score = float(np.median(scored_df["anomaly_score"]))
    st.metric("Median anomaly score", f"{median_score:.4f}")

# -------------------------
# TIME SERIES PLOT WITH ANOMALIES
# -------------------------
st.subheader("Time series — anomalies highlighted")
base = alt.Chart(scored_df).encode(x="ts:T")
temp_line = base.mark_line().encode(y=alt.Y("temperature:Q", title="Temperature (°C)"))
hum_line = base.mark_line(color="green").encode(y=alt.Y("humidity:Q", title="Humidity (%)"))

# anomaly points (temperature)
anom_points = alt.Chart(scored_df[scored_df["is_anomaly"]==1]).mark_circle(size=70, color="red").encode(
    x="ts:T",
    y="temperature:Q",
    tooltip=["ts:T", "temperature", "humidity", "anomaly_score"]
)

st.altair_chart((temp_line + anom_points).interactive().resolve_scale(y='independent'), use_container_width=True)

# humidity with anomalies
anom_points_h = alt.Chart(scored_df[scored_df["is_anomaly"]==1]).mark_circle(size=70, color="red").encode(
    x="ts:T",
    y="humidity:Q",
    tooltip=["ts:T", "temperature", "humidity", "anomaly_score"]
)
st.altair_chart((hum_line + anom_points_h).interactive().resolve_scale(y='independent'), use_container_width=True)

# -------------------------
# PCA projection for 2D visualization
# -------------------------
st.subheader("PCA projection of features (2D) — anomalies marked")
X_vis, _ = prepare_features(scored_df, selected_features)
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_vis)
vis_df = pd.DataFrame(proj, columns=["pc1", "pc2"])
vis_df["is_anomaly"] = scored_df["is_anomaly"].values
vis_df["ts"] = scored_df["ts"].values
vis_df["score"] = scored_df["anomaly_score"].values

scatter = alt.Chart(vis_df).mark_circle(size=60).encode(
    x="pc1:Q", y="pc2:Q",
    color=alt.condition(alt.datum.is_anomaly==1, alt.value("red"), alt.value("blue")),
    tooltip=["ts:T", "score"]
).interactive()
st.altair_chart(scatter, use_container_width=True)

# -------------------------
# ANOMALY TABLE & EXPLANATIONS
# -------------------------
st.subheader("Anomaly table (sorted by anomaly score desc)")
sorted_anom = scored_df.sort_values("anomaly_score", ascending=False)
st.dataframe(sorted_anom[["ts", "temperature", "humidity", "anomaly_score", "is_anomaly"]].head(200), height=300)

st.markdown("**Explanation tips:**")
st.markdown("""
- `anomaly_score`: higher = more anomalous (we invert model's decision function so larger means more suspicious).  
- `is_anomaly`: binary label (1 = anomaly).  
- Try changing `contamination` — this adjusts sensitivity.  
- Use rolling window / features to capture short bursts or gradual drifts.
""")

# -------------------------
# EXPORTS
# -------------------------
def get_table_download_link(df_to_download: pd.DataFrame, filename="anomaly_report.csv"):
    csv = df_to_download.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇ Download anomaly CSV</a>'
    return href

st.markdown(get_table_download_link(sorted_anom, filename="anomaly_report.csv"), unsafe_allow_html=True)

# download model (joblib)
if download_model_btn:
    if "model" in st.session_state and st.session_state["model"] is not None:
        mem_file = io.BytesIO()
        joblib.dump(st.session_state["model"], mem_file)
        mem_file.seek(0)
        b64 = base64.b64encode(mem_file.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="iforest_model.joblib">⬇ Download model (joblib)</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No model in session. Train a model first.")

# -------------------------
# OPTIONAL: simple alert rule
# -------------------------
st.markdown("---")
st.subheader("Simple alerting (local)")

threshold = st.slider("Anomaly score threshold for alerting (higher = more sensitive)", 
                      float(sorted_anom["anomaly_score"].min()), float(sorted_anom["anomaly_score"].max()), 
                      float(sorted_anom["anomaly_score"].quantile(0.95)))
recent_alerts = sorted_anom[sorted_anom["anomaly_score"] >= threshold][["ts", "temperature", "humidity", "anomaly_score"]]
st.write(f"Events above threshold: {len(recent_alerts)}")
st.dataframe(recent_alerts.head(50))

st.caption("You can wire these events to email/Slack/webhook externally (not included).")
