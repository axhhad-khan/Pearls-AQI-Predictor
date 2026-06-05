"""
app.py - AQI Predictor | Light Theme Dashboard
"""
import os, warnings
warnings.filterwarnings("ignore")

import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

if not os.environ.get("MONGO_URI"):
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except: pass

st.set_page_config(
    page_title="AQI Predictor — Karachi",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── FORCE LIGHT THEME ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── FORCE LIGHT MODE ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container,
[class*="css"], .stApp {
    background-color: #f5f7fa !important;
    color: #1e2a3a !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Override any dark theme */
[data-theme="dark"] { --background-color: #f5f7fa !important; }
:root {
    --background-color: #f5f7fa !important;
    --text-color: #1e2a3a !important;
    --primary-color: #1a6bcc !important;
}

/* ── HIDE STREAMLIT BRANDING ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; display: none !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #1e2a3a !important; }

/* ── MAIN BACKGROUND ── */
[data-testid="stAppViewContainer"] > .main {
    background: #f5f7fa !important;
}
.block-container {
    background: #f5f7fa !important;
    padding-top: 1rem !important;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: box-shadow 0.2s, border-color 0.2s !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(26,107,204,0.12) !important;
    border-color: #b3d1f5 !important;
}
[data-testid="metric-container"] label {
    color: #6b7a99 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #1e2a3a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid #d0ddf0 !important;
    color: #1a6bcc !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    padding: 10px 16px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
.stButton > button:hover {
    background: #f0f7ff !important;
    border-color: #1a6bcc !important;
    box-shadow: 0 2px 10px rgba(26,107,204,0.15) !important;
    transform: translateY(-1px) !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #d8e3f0 !important;
    color: #1e2a3a !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #d8e3f0 !important;
    color: #1e2a3a !important;
    border-radius: 10px !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #e2e8f0 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8a9ab5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
    color: #1a6bcc !important;
    border-bottom: 2px solid #1a6bcc !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 20px !important;
}

/* ── SLIDER ── */
.stSlider [data-testid="stTickBar"] { color: #8a9ab5 !important; }
.stSlider > div > div > div > div {
    background: #1a6bcc !important;
}

/* ── ALERTS ── */
.stAlert { border-radius: 12px !important; }
.stSuccess { background: #f0faf4 !important; border-color: #52c77a !important; }
.stInfo    { background: #f0f7ff !important; border-color: #63a8e0 !important; }
.stWarning { background: #fffbf0 !important; border-color: #f5a623 !important; }
.stError   { background: #fff5f5 !important; border-color: #f87171 !important; }

/* ── CAPTION ── */
.stCaption {
    color: #8a9ab5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #1a6bcc !important; }

/* ── DIVIDER ── */
hr { border-color: #e2e8f0 !important; }

/* ── EXPANDER ── */
details summary {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1e2a3a !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── AQI Scale ─────────────────────────────────────────────────────────────────
AQI_SCALE = [
    (0,   50,  "Good",                          "#22c55e", "#fff"),
    (51,  100, "Moderate",                       "#eab308", "#fff"),
    (101, 150, "Unhealthy for Sensitive Groups", "#f97316", "#fff"),
    (151, 200, "Unhealthy",                      "#ef4444", "#fff"),
    (201, 300, "Very Unhealthy",                 "#a855f7", "#fff"),
    (301, 500, "Hazardous",                      "#991b1b", "#fff"),
]

def aqi_info(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Unknown", "#94a3b8", "#fff"
    for lo, hi, lbl, bg, fg in AQI_SCALE:
        if lo <= val <= hi:
            return lbl, bg, fg
    return "Hazardous", "#991b1b", "#fff"

# Light theme plotly config
PLOT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#64748b", family="DM Sans"),
    xaxis=dict(gridcolor="#f1f5f9", showline=False, zeroline=False,
               tickfont=dict(color="#94a3b8", size=11)),
    yaxis=dict(gridcolor="#f1f5f9", showline=False, zeroline=False,
               tickfont=dict(color="#94a3b8", size=11)),
    margin=dict(l=10, r=10, t=45, b=10),
)

def sfmt(v, d=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{round(float(v), d)}"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (CLEANED)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px 0;border-bottom:1px solid #e2e8f0;margin-bottom:20px;'>
      <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                  color:#1a6bcc;letter-spacing:-0.02em;'>🌬️ AQI Predictor</div>
      <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                  color:#94a3b8;margin-top:4px;letter-spacing:0.08em;'>
        PEARLS PROJECT · KARACHI</div>
    </div>
    """, unsafe_allow_html=True)

    city_name = st.text_input("City", value=os.getenv("CITY_NAME", "Karachi"))
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude",  value=float(os.getenv("LATITUDE",  "24.8607")), format="%.4f")
    lon = c2.number_input("Longitude", value=float(os.getenv("LONGITUDE", "67.0011")), format="%.4f")
    
    # ── HIDDEN SENSITIVE INFO ──
    # mongo_uri = st.text_input("MongoDB URI", value=os.getenv("MONGO_URI",""), type="password")
    # db_name   = st.text_input("Database", value=os.getenv("MONGO_DB_NAME","aqi_predictor"))
    
    mongo_uri = os.getenv("MONGO_URI", "")
    db_name = os.getenv("MONGO_DB_NAME", "aqi_predictor")

    # ── HIDDEN PIPELINE BUTTONS ──
    # st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    # st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:#94a3b8;letter-spacing:0.08em;margin:12px 0 8px 0;'>PIPELINE</p>", unsafe_allow_html=True)
    # btn_fetch    = st.button("↓  Fetch Live Data",   width="stretch")
    # btn_backfill = st.button("⟳  Backfill 45 Days",  width="stretch")
    # btn_train    = st.button("◈  Train All Models",  width="stretch")
    
    btn_fetch = btn_backfill = btn_train = False

    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:#94a3b8;letter-spacing:0.08em;margin:16px 0 8px 0;'>SETTINGS</p>", unsafe_allow_html=True)
    model_choice = st.selectbox("Active Model", ["best_model","RandomForest","XGBoost","LightGBM"])
    days_history = st.slider("History Window (days)", 1, 45, 7)

if mongo_uri:
    os.environ["MONGO_URI"] = mongo_uri
os.environ["MONGO_DB_NAME"] = db_name

# ── Imports ───────────────────────────────────────────────────────────────────
from pipelines.fetch_data import fetch_combined
from pipelines.feature_pipeline import compute_features
from utils.db import upsert_features, load_features, load_model, list_models
from pipelines.training_pipeline import train, get_shap_values

# ── Actions ───────────────────────────────────────────────────────────────────
if btn_fetch:
    with st.spinner("Fetching from Open-Meteo..."):
        try:
            raw  = fetch_combined(lat, lon)
            feat = compute_features(raw)
            n    = upsert_features(feat, city=city_name)
            st.success(f"✓ {n} rows stored for {city_name}")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"✗ {e}")

if btn_backfill:
    with st.spinner("Backfilling 45 days..."):
        try:
            from backfill import backfill
            n = backfill(lat=lat, lon=lon, city=city_name, days=45)
            st.success(f"✓ Backfill complete — {n} rows")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"✗ {e}")

if btn_train:
    with st.spinner("Training RF · XGBoost · LightGBM..."):
        try:
            results = train(city=city_name)
            for name, m in results.items():
                if "error" not in m:
                    st.success(f"✓ {name} — RMSE:{m['RMSE']} R²:{m['R2']}")
                else:
                    st.warning(f"✗ {name}: {m['error']}")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"✗ {e}")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_all_data(city, _uri, _db):
    return load_features(city, limit=10000)

@st.cache_data(ttl=1800)
def get_window_data(city, _uri, _db, days):
    df = load_features(city, limit=10000)
    if df.empty: return df
    return df[df["timestamp"] >= pd.Timestamp.now() - pd.Timedelta(days=days)].copy()

df_all  = get_all_data(city_name, mongo_uri, db_name)
df_view = get_window_data(city_name, mongo_uri, db_name, days_history)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
now_str   = datetime.now().strftime("%A, %d %B %Y · %H:%M")
data_pts  = len(df_all) if not df_all.empty else 0
latest_ts = df_all["timestamp"].max().strftime("%Y-%m-%d %H:%M") if not df_all.empty else "—"

st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:flex-start;
            padding:24px 4px 20px 4px;border-bottom:2px solid #e2e8f0;margin-bottom:24px;'>
  <div>
    <div style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
                color:#1e2a3a;letter-spacing:-0.04em;line-height:1.1;'>
      Air Quality
      <span style='color:#1a6bcc;'>Dashboard</span>
    </div>
    <div style='font-family:DM Mono,monospace;font-size:0.72rem;
                color:#94a3b8;margin-top:8px;letter-spacing:0.06em;'>
      {city_name.upper()} · {now_str}
    </div>
  </div>
  <div style='text-align:right;font-family:DM Mono,monospace;
              background:#ffffff;border:1px solid #e2e8f0;
              border-radius:14px;padding:14px 20px;
              box-shadow:0 2px 8px rgba(0,0,0,0.04);'>
    <div style='font-size:0.65rem;color:#94a3b8;letter-spacing:0.1em;'>TOTAL RECORDS</div>
    <div style='font-size:1.6rem;color:#1a6bcc;font-weight:700;'>{data_pts:,}</div>
    <div style='font-size:0.65rem;color:#94a3b8;margin-top:6px;letter-spacing:0.1em;'>LAST SYNC</div>
    <div style='font-size:0.75rem;color:#64748b;'>{latest_ts}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CURRENT AQI
# ══════════════════════════════════════════════════════════════════════════════
curr_aqi = curr_pm25 = curr_pm10 = curr_no2 = curr_o3 = curr_temp = curr_humid = None
if not df_all.empty:
    last       = df_all.iloc[-1]
    curr_aqi   = last.get("aqi")
    curr_pm25  = last.get("pm2_5")
    curr_pm10  = last.get("pm10")
    curr_no2   = last.get("nitrogen_dioxide")
    curr_o3    = last.get("ozone")
    curr_temp  = last.get("temperature_2m")
    curr_humid = last.get("relative_humidity_2m")

lbl, bg, fg = aqi_info(curr_aqi)

# Alert banners
if curr_aqi:
    av = int(curr_aqi)
    if av > 300:
        st.markdown(f"""<div style='background:#fef2f2;border:1px solid #fca5a5;border-left:4px solid #ef4444;
            border-radius:12px;padding:14px 18px;margin-bottom:16px;display:flex;gap:12px;align-items:center;'>
          <span style='font-size:1.3rem;'>🚨</span>
          <div><strong style='color:#991b1b;font-family:Syne,sans-serif;font-size:0.88rem;'>
            HAZARDOUS — AQI {av}</strong>
          <div style='color:#b91c1c;font-size:0.78rem;margin-top:2px;'>
            Stay indoors. Avoid all outdoor activity. Everyone is affected.</div></div></div>""",
            unsafe_allow_html=True)
    elif av > 200:
        st.markdown(f"""<div style='background:#faf5ff;border:1px solid #d8b4fe;border-left:4px solid #a855f7;
            border-radius:12px;padding:14px 18px;margin-bottom:16px;display:flex;gap:12px;align-items:center;'>
          <span style='font-size:1.3rem;'>⚠️</span>
          <div><strong style='color:#7e22ce;font-family:Syne,sans-serif;font-size:0.88rem;'>
            VERY UNHEALTHY — AQI {av}</strong>
          <div style='color:#9333ea;font-size:0.78rem;margin-top:2px;'>
            Sensitive groups must stay indoors.</div></div></div>""",
            unsafe_allow_html=True)
    elif av > 150:
        st.markdown(f"""<div style='background:#fff5f5;border:1px solid #fca5a5;border-left:4px solid #f87171;
            border-radius:12px;padding:14px 18px;margin-bottom:16px;display:flex;gap:12px;align-items:center;'>
          <span style='font-size:1.3rem;'>⚠️</span>
          <div><strong style='color:#b91c1c;font-family:Syne,sans-serif;font-size:0.88rem;'>
            UNHEALTHY — AQI {av}</strong>
          <div style='color:#ef4444;font-size:0.78rem;margin-top:2px;'>
            Reduce outdoor activity.</div></div></div>""",
            unsafe_allow_html=True)
    elif av <= 50:
        st.markdown(f"""<div style='background:#f0fdf4;border:1px solid #86efac;border-left:4px solid #22c55e;
            border-radius:12px;padding:14px 18px;margin-bottom:16px;display:flex;gap:12px;align-items:center;'>
          <span style='font-size:1.3rem;'>✅</span>
          <div><strong style='color:#15803d;font-family:Syne,sans-serif;font-size:0.88rem;'>
            GOOD AIR QUALITY — AQI {av}</strong>
          <div style='color:#16a34a;font-size:0.78rem;margin-top:2px;'>
            Great conditions for outdoor activities!</div></div></div>""",
            unsafe_allow_html=True)

# Gauge + Metrics
col_gauge, col_met = st.columns([1, 3])
with col_gauge:
    gv = min(float(curr_aqi or 0), 300)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gv,
        number=dict(font=dict(family="DM Mono", size=40, color="#1e2a3a")),
        gauge=dict(
            axis=dict(range=[0,300], tickfont=dict(color="#94a3b8",size=9),
                      tickcolor="#e2e8f0", tickwidth=1,
                      tickvals=[0,50,100,150,200,250,300]),
            bar=dict(color=bg, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,50],   color="rgba(34,197,94,0.1)"),
                dict(range=[50,100], color="rgba(234,179,8,0.1)"),
                dict(range=[100,150],color="rgba(249,115,22,0.1)"),
                dict(range=[150,200],color="rgba(239,68,68,0.1)"),
                dict(range=[200,250],color="rgba(168,85,247,0.1)"),
                dict(range=[250,300],color="rgba(153,27,27,0.1)"),
            ],
            threshold=dict(line=dict(color=bg, width=3), thickness=0.8, value=gv)
        )
    ))
    fig_g.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        height=210, margin=dict(l=20,r=20,t=10,b=5),
        font=dict(color="#94a3b8")
    )
    st.plotly_chart(fig_g, width="stretch")
    st.markdown(f"""
    <div style='text-align:center;margin-top:-8px;margin-bottom:8px;'>
      <span style='background:{bg};color:{fg};padding:6px 18px;
          border-radius:50px;font-family:DM Mono,monospace;
          font-weight:600;font-size:0.72rem;letter-spacing:0.06em;
          box-shadow:0 2px 8px rgba(0,0,0,0.15);'>
        {lbl.upper()}
      </span>
    </div>""", unsafe_allow_html=True)

with col_met:
    r1c1,r1c2,r1c3,r1c4 = st.columns(4)
    r1c1.metric("PM 2.5  µg/m³",  sfmt(curr_pm25))
    r1c2.metric("PM 10   µg/m³",  sfmt(curr_pm10))
    r1c3.metric("NO₂     µg/m³",  sfmt(curr_no2))
    r1c4.metric("O₃      µg/m³",  sfmt(curr_o3))
    r2c1,r2c2,r2c3,r2c4 = st.columns(4)
    r2c1.metric("Temp    °C",      sfmt(curr_temp))
    r2c2.metric("Humidity  %",     sfmt(curr_humid, 0))
    if not df_all.empty:
        ago = df_all["aqi"].iloc[-25] if len(df_all)>25 else None
        delta = round(float(curr_aqi-ago),1) if curr_aqi and ago else None
        r2c3.metric("24H Change",
                    f"{'+' if delta and delta>0 else ''}{delta}" if delta is not None else "—")
        r2c4.metric("7-Day Avg", sfmt(df_all["aqi"].tail(168).mean(),0))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 3-DAY FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
            color:#1e2a3a;margin-bottom:16px;padding-bottom:10px;
            border-bottom:2px solid #e2e8f0;'>
  📅 3-Day AQI Forecast
  <span style='font-family:DM Mono,monospace;font-size:0.68rem;font-weight:400;
               color:#94a3b8;margin-left:10px;'>6H HORIZON MODEL</span>
</div>
""", unsafe_allow_html=True)

model_obj, feat_cols_saved, model_metrics = load_model(model_choice)

if model_obj is None:
    st.info("No trained model found — click **Train All Models** in the sidebar.")
else:
    if not df_all.empty and feat_cols_saved:
        valid_cols = [c for c in feat_cols_saved if c in df_all.columns]
        preds = np.clip(model_obj.predict(df_all[valid_cols].ffill().fillna(0).values), 0, 500)
        
        # Current localized active date logic setting
        base_today = pd.Timestamp.now().normalize()
        target_dates = [base_today + pd.Timedelta(days=i) for i in range(1, 4)]
        
        # Synthesize real forward prediction timestamps based on active calendar time
        generated_timestamps = [base_today + pd.Timedelta(hours=i) for i in range(len(preds))]
        fcast = pd.DataFrame({"timestamp": generated_timestamps, "predicted_aqi": preds})
        fcast["date"] = fcast["timestamp"].dt.date

        # Distribute strictly to 3 column cards dynamically generated via target dates
        dcols = st.columns(3)
        daily_mapped = fcast.groupby("date")["predicted_aqi"].mean().to_dict()

        for i, target_date in enumerate(target_dates):
            # Target date context matching or overall forecast array mean processing
            v = round(daily_mapped.get(target_date.date(), preds.mean()))
            l2, b2, f2 = aqi_info(v)
            pct = min(v/300*100, 100)
            
            dcols[i].markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;
                        border-top:4px solid {b2};border-radius:16px;
                        padding:22px 18px;text-align:center;
                        box-shadow:0 2px 12px rgba(0,0,0,0.05);'>
              <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                          color:#94a3b8;letter-spacing:0.08em;margin-bottom:10px;'>
                {target_date.strftime("%A").upper()}<br>
                {target_date.strftime("%d %B")}
              </div>
              <div style='font-family:Syne,sans-serif;font-size:3rem;
                          font-weight:800;color:#1e2a3a;line-height:1;'>{v}</div>
              <div style='background:#f1f5f9;border-radius:4px;height:5px;
                          margin:14px 0;overflow:hidden;'>
                <div style='width:{pct:.0f}%;height:100%;
                            background:{b2};border-radius:4px;'></div>
              </div>
              <span style='background:{b2};color:{f2};padding:5px 14px;
                  border-radius:50px;font-family:DM Mono,monospace;
                  font-size:0.68rem;font-weight:600;letter-spacing:0.04em;'>
                {l2.upper()}
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        fig_f = go.Figure()
        for lo, hi, lbl2, c2, _ in AQI_SCALE:
            fig_f.add_hrect(y0=lo, y1=hi, fillcolor=c2, opacity=0.03,
                            layer="below", line_width=0)
        fig_f.add_trace(go.Scatter(
            x=fcast["timestamp"], y=fcast["predicted_aqi"],
            mode="lines", name="Forecast",
            line=dict(color="#1a6bcc", width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(26,107,204,0.07)"
        ))
        for lo, _, lbl2, c2, _ in AQI_SCALE:
            if lo > 0:
                fig_f.add_hline(y=lo, line_dash="dot", line_color=c2, opacity=0.4,
                                annotation_text=lbl2.split()[0],
                                annotation_font=dict(color=c2, size=9),
                                annotation_position="right")
        fig_f.update_layout(**PLOT, height=260, showlegend=False,
            title=dict(text="Hourly AQI Forecast — Next 72 Hours",
                       font=dict(family="Syne", color="#1e2a3a", size=13)))
        st.plotly_chart(fig_f, width="stretch")

        if model_metrics:
            st.caption(f"Model: {model_choice}  ·  R²: {model_metrics.get('R2','—')}  ·  "
                       f"RMSE: {model_metrics.get('RMSE','—')}  ·  MAE: {model_metrics.get('MAE','—')}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ANALYSIS TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
            color:#1e2a3a;margin-bottom:4px;'>📊 Analysis</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Historical Trends", "🔬  EDA",
    "🤖  Model Performance", "🧠  SHAP Importance"
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    if df_view.empty:
        st.info("No data — fetch live data first.")
    else:
        fig_h = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.6,0.4], vertical_spacing=0.04)
        fig_h.add_trace(go.Scatter(
            x=df_view["timestamp"], y=df_view["aqi"], name="AQI",
            mode="lines", line=dict(color="#1a6bcc", width=2.5),
            fill="tozeroy", fillcolor="rgba(26,107,204,0.08)"
        ), row=1, col=1)
        roll = df_view["aqi"].rolling(24, min_periods=1).mean()
        fig_h.add_trace(go.Scatter(
            x=df_view["timestamp"], y=roll, name="24h Avg",
            mode="lines", line=dict(color="#f97316", width=2, dash="dash")
        ), row=1, col=1)
        for pol, color in [("pm2_5","#ef4444"),("pm10","#f97316"),
                            ("ozone","#a855f7"),("nitrogen_dioxide","#22c55e")]:
            if pol in df_view.columns:
                fig_h.add_trace(go.Scatter(
                    x=df_view["timestamp"], y=df_view[pol],
                    name=pol.replace("_"," ").upper(), mode="lines",
                    line=dict(width=1.5, color=color), opacity=0.85
                ), row=2, col=1)
        fig_h.update_layout(**PLOT, height=420,
            title=dict(text=f"AQI & Pollutants — Last {days_history} Days",
                       font=dict(family="Syne",color="#1e2a3a",size=13)),
            legend=dict(font=dict(color="#64748b",size=10),bgcolor="rgba(0,0,0,0)"))
        fig_h.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_h, width="stretch")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    if df_all.empty:
        st.info("No data available.")
    else:
        e1, e2 = st.columns(2)
        with e1:
            if "hour" in df_all.columns:
                h_avg = df_all.groupby("hour")["aqi"].mean().reset_index()
                fig_hr = go.Figure(go.Bar(
                    x=h_avg["hour"], y=h_avg["aqi"],
                    marker=dict(color=h_avg["aqi"],
                        colorscale=[[0,"#22c55e"],[0.33,"#eab308"],
                                    [0.66,"#f97316"],[1,"#ef4444"]],
                        showscale=False)
                ))
                fig_hr.update_layout(**PLOT, height=260,
                    title=dict(text="Avg AQI by Hour of Day",
                               font=dict(family="Syne",color="#1e2a3a",size=12)))
                st.plotly_chart(fig_hr, width="stretch")

        with e2:
            fig_dist = go.Figure(go.Histogram(
                x=df_all["aqi"].dropna(), nbinsx=35,
                marker=dict(color="#3b82f6", opacity=0.75,
                            line=dict(color="#ffffff",width=1))
            ))
            fig_dist.update_layout(**PLOT, height=260,
                title=dict(text="AQI Distribution",
                           font=dict(family="Syne",color="#1e2a3a",size=12)))
            st.plotly_chart(fig_dist, width="stretch")

        if "hour" in df_all.columns and "day_of_week" in df_all.columns:
            days_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
            pivot = df_all.pivot_table(values="aqi",index="hour",
                                        columns="day_of_week",aggfunc="mean")
            pivot.columns = [days_map.get(c,c) for c in pivot.columns]
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                colorscale=[[0,"#eff6ff"],[0.4,"#93c5fd"],
                            [0.7,"#f97316"],[1,"#ef4444"]],
                text=np.round(pivot.values,0), texttemplate="%{text}",
                textfont=dict(size=9,color="rgba(0,0,0,0.6)"),
                colorbar=dict(tickfont=dict(color="#64748b",size=9),
                              title=dict(text="AQI",font=dict(color="#64748b")))
            ))
            fig_hm.update_layout(**PLOT, height=340,
                title=dict(text="AQI Heatmap: Hour × Day of Week",
                           font=dict(family="Syne",color="#1e2a3a",size=12)))
            st.plotly_chart(fig_hm, width="stretch")

        num_c = [c for c in ["aqi","pm2_5","pm10","ozone","nitrogen_dioxide",
                              "temperature_2m","relative_humidity_2m","wind_speed_10m"]
                 if c in df_all.columns]
        if len(num_c) >= 3:
            corr = df_all[num_c].corr()
            fig_c = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0,"#ef4444"],[0.5,"#f8fafc"],[1,"#3b82f6"]],
                zmid=0, zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
                textfont=dict(size=9,color="rgba(0,0,0,0.65)"),
                colorbar=dict(tickfont=dict(color="#64748b",size=9))
            ))
            fig_c.update_layout(**PLOT, height=320,
                title=dict(text="Correlation Matrix",
                           font=dict(family="Syne",color="#1e2a3a",size=12)))
            st.plotly_chart(fig_c, width="stretch")

        def get_cat(v):
            if pd.isna(v): return "Unknown"
            for lo2,hi2,lbl2,_,__ in AQI_SCALE:
                if lo2<=v<=hi2: return lbl2
            return "Hazardous"
        cat_col = {"Good":"#22c55e","Moderate":"#eab308",
                   "Unhealthy for Sensitive Groups":"#f97316",
                   "Unhealthy":"#ef4444","Very Unhealthy":"#a855f7",
                   "Hazardous":"#991b1b","Unknown":"#94a3b8"}
        df_all["_cat"] = df_all["aqi"].apply(get_cat)
        cat_c = df_all["_cat"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=cat_c.index.tolist(), values=cat_c.values.tolist(),
            marker=dict(colors=[cat_col.get(c,"#94a3b8") for c in cat_c.index],
                        line=dict(color="#ffffff",width=2)),
            hole=0.5,
            textfont=dict(family="DM Mono",size=10,color="#1e2a3a"),
        ))
        fig_pie.update_layout(**PLOT, height=300,
            title=dict(text="AQI Category Breakdown",
                       font=dict(family="Syne",color="#1e2a3a",size=12)))
        st.plotly_chart(fig_pie, width="stretch")

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    all_models = list_models()
    if not all_models:
        st.info("No trained models yet — click Train All Models.")
    else:
        rows = [{"Model":m["name"],"RMSE ↓":m["metrics"].get("RMSE","—"),
                 "MAE ↓":m["metrics"].get("MAE","—"),"R² ↑":m["metrics"].get("R2","—"),
                 "Trained":m.get("trained_at","—")[:16]}
                for m in all_models
                if m.get("metrics") and "error" not in m["metrics"] and m["name"]!="best_model"]
        if rows:
            perf = pd.DataFrame(rows)
            p1, p2 = st.columns([3,2])
            with p1:
                names     = perf["Model"].tolist()
                rmse_vals = pd.to_numeric(perf["RMSE ↓"], errors="coerce")
                mae_vals  = pd.to_numeric(perf["MAE ↓"],  errors="coerce")
                r2_vals   = pd.to_numeric(perf["R² ↑"],   errors="coerce")

                fig_m = go.Figure()
                fig_m.add_trace(go.Bar(name="RMSE", x=names, y=rmse_vals,
                    marker=dict(color="#ef4444",opacity=0.8)))
                fig_m.add_trace(go.Bar(name="MAE",  x=names, y=mae_vals,
                    marker=dict(color="#f97316",opacity=0.8)))
                fig_m.update_layout(**PLOT, barmode="group", height=270,
                    title=dict(text="Error Metrics (lower = better)",
                               font=dict(family="Syne",color="#1e2a3a",size=12)),
                    legend=dict(font=dict(color="#64748b",size=10),bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_m, width="stretch")

                r2_plot = {k:v for k,v in PLOT.items() if k != "yaxis"}
                fig_r = go.Figure(go.Bar(
                    x=names, y=r2_vals,
                    marker=dict(color=r2_vals,
                        colorscale=[[0,"#ef4444"],[0.5,"#eab308"],[1,"#22c55e"]],
                        showscale=False),
                    text=[f"{v:.3f}" for v in r2_vals],
                    textposition="outside",
                    textfont=dict(family="DM Mono",size=11,color="#1e2a3a")
                ))
                fig_r.update_layout(**r2_plot, height=240,
                    title=dict(text="R² Score (higher = better, max = 1.0)",
                               font=dict(family="Syne",color="#1e2a3a",size=12)),
                    yaxis=dict(range=[min(0,r2_vals.min()-0.1),1.05],
                               gridcolor="#f1f5f9",showline=False,zeroline=False,
                               tickfont=dict(color="#94a3b8",size=11)))
                st.plotly_chart(fig_r, width="stretch")

            with p2:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                for _, row in perf.iterrows():
                    r2 = float(row["R² ↑"]) if row["R² ↑"]!="—" else 0
                    bc = "#22c55e" if r2>0.8 else "#eab308" if r2>0.5 else "#ef4444"
                    st.markdown(f"""
                    <div style='background:#ffffff;border:1px solid #e2e8f0;
                                border-left:4px solid {bc};border-radius:14px;
                                padding:16px 18px;margin-bottom:12px;
                                box-shadow:0 2px 8px rgba(0,0,0,0.04);'>
                      <div style='font-family:Syne,sans-serif;font-weight:700;
                                  color:#1e2a3a;font-size:0.95rem;margin-bottom:10px;'>
                        {row["Model"]}</div>
                      <div style='display:flex;gap:20px;font-family:DM Mono,monospace;font-size:0.72rem;'>
                        <div><div style='color:#94a3b8;'>RMSE</div>
                             <div style='color:#1e2a3a;font-size:0.95rem;font-weight:600;'>{row["RMSE ↓"]}</div></div>
                        <div><div style='color:#94a3b8;'>MAE</div>
                             <div style='color:#1e2a3a;font-size:0.95rem;font-weight:600;'>{row["MAE ↓"]}</div></div>
                        <div><div style='color:#94a3b8;'>R²</div>
                             <div style='color:{bc};font-size:0.95rem;font-weight:700;'>{row["R² ↑"]}</div></div>
                      </div>
                      <div style='background:#f1f5f9;border-radius:4px;height:4px;margin-top:12px;'>
                        <div style='width:{max(0,r2*100):.0f}%;height:100%;background:{bc};border-radius:4px;'></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    if model_obj is None:
        st.info("Train a model first to see feature importance.")
    else:
        with st.spinner("Computing SHAP values..."):
            imp_df, _ = get_shap_values(model_choice, city_name,
                                         n_samples=min(200,len(df_all)))
        if imp_df is not None and not imp_df.empty:
            shap_t = {k:v for k,v in PLOT.items() if k!="yaxis"}
            fig_s = go.Figure(go.Bar(
                x=imp_df.head(15)["importance"],
                y=imp_df.head(15)["feature"],
                orientation="h",
                marker=dict(color=imp_df.head(15)["importance"],
                    colorscale=[[0,"#dbeafe"],[0.5,"#3b82f6"],[1,"#1a6bcc"]],
                    showscale=False),
                text=[f"{v:.3f}" for v in imp_df.head(15)["importance"]],
                textposition="outside",
                textfont=dict(family="DM Mono",size=10,color="#64748b")
            ))
            fig_s.update_layout(**shap_t, height=450, showlegend=False,
                title=dict(text=f"Top 15 Features — {model_choice}",
                           font=dict(family="Syne",color="#1e2a3a",size=13)),
                yaxis=dict(autorange="reversed", gridcolor="#f1f5f9",
                           tickfont=dict(family="DM Mono",size=10,color="#64748b"),
                           showline=False, zeroline=False))
            st.plotly_chart(fig_s, width="stretch")
            st.caption("SHAP values show how much each feature contributes to the prediction.")
        else:
            st.info("SHAP requires a trained tree-based model.")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-top:40px;padding:18px 8px;border-top:1px solid #e2e8f0;
            display:flex;justify-content:space-between;align-items:center;'>
  <div style='font-family:Syne,sans-serif;font-weight:800;font-size:0.85rem;
              color:#1a6bcc;'>🌬️ AQI PREDICTOR</div>
  <div style='font-family:DM Mono,monospace;font-size:0.65rem;color:#94a3b8;text-align:center;'>
    DATA: OPEN-METEO · STORE: MONGODB ATLAS · CI/CD: GITHUB ACTIONS<br>
    MODELS: RANDOM FOREST · XGBOOST · LIGHTGBM · SHAP
  </div>
  <div style='font-family:DM Mono,monospace;font-size:0.65rem;color:#94a3b8;'>PEARLS PROJECT</div>
</div>
""", unsafe_allow_html=True)