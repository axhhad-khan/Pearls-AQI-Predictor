"""
app.py - AQI Predictor Streamlit Dashboard
PDF Requirements:
1. Feature Store load
2. Model predictions + dashboard
3. Streamlit + Plotly
4. EDA trends
5. SHAP feature importance
6. Hazardous AQI alerts
7. 3-day forecast (Day1, Day2, Day3)
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
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="AQI Predictor — Karachi",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: linear-gradient(135deg,#0a0e1a,#0d1b2a,#0a1628); }
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 20px; text-align: center;
}
.section-title {
    font-size: 1.2rem; font-weight: 600; color: #e2e8f0;
    border-bottom: 2px solid rgba(99,179,237,0.3);
    padding-bottom: 8px; margin-bottom: 12px;
}
.alert-box {
    border-radius: 12px; padding: 14px; margin: 8px 0; font-weight: 600;
}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px;
}
div[data-testid="metric-container"] label { color:#94a3b8!important; font-size:0.8rem!important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#e2e8f0!important; font-family:'JetBrains Mono',monospace; font-size:1.5rem!important;
}
.stSidebar { background: rgba(10,14,26,0.95)!important; }
.stButton>button {
    background: linear-gradient(135deg,#3b82f6,#2563eb);
    color:white; border:none; border-radius:10px;
    font-weight:600; padding:10px 24px;
}
</style>
""", unsafe_allow_html=True)

# ── AQI Scale ─────────────────────────────────────────────────────────────────
AQI_SCALE = [
    (0,   50,  "Good",                          "#00e400", "#000"),
    (51,  100, "Moderate",                       "#ffff00", "#000"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "#000"),
    (151, 200, "Unhealthy",                      "#ff0000", "#fff"),
    (201, 300, "Very Unhealthy",                 "#8f3f97", "#fff"),
    (301, 500, "Hazardous",                      "#7e0023", "#fff"),
]

def aqi_info(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Unknown", "#555", "#fff"
    for lo, hi, lbl, bg, fg in AQI_SCALE:
        if lo <= val <= hi:
            return lbl, bg, fg
    return "Hazardous", "#7e0023", "#fff"

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Space Grotesk"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    city_name = st.text_input("City", value=os.getenv("CITY_NAME", "Karachi"))
    c1, c2    = st.columns(2)
    lat = c1.number_input("Lat",  value=float(os.getenv("LATITUDE",  "24.8607")), format="%.4f")
    lon = c2.number_input("Lon",  value=float(os.getenv("LONGITUDE", "67.0011")), format="%.4f")
    mongo_uri = st.text_input("MongoDB URI", value=os.getenv("MONGO_URI","mongodb://localhost:27017/"), type="password")
    db_name   = st.text_input("DB Name",     value=os.getenv("MONGO_DB_NAME","aqi_predictor"))

    st.markdown("---")
    st.markdown("### 🔄 Pipeline Controls")
    btn_fetch    = st.button("📡 Fetch Live Data",   use_container_width=True)
    btn_backfill = st.button("🗃️ Backfill 45 Days",  use_container_width=True)
    btn_train    = st.button("🤖 Train All Models",  use_container_width=True)

    st.markdown("---")
    model_choice = st.selectbox("Forecast Model",
        ["best_model","RandomForest","XGBoost","LightGBM"])
    days_history = st.slider("History (days)", 1, 45, 7)

os.environ["MONGO_URI"]     = mongo_uri
os.environ["MONGO_DB_NAME"] = db_name

# ── Imports ───────────────────────────────────────────────────────────────────
from pipelines.fetch_data       import fetch_combined
from pipelines.feature_pipeline import compute_features, aqi_category
from utils.db                   import (upsert_features, load_features,
                                         load_model, list_models)
from pipelines.training_pipeline import train, get_shap_values

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:20px 0 5px 0;'>
  <h1 style='font-size:2.5rem;font-weight:700;color:#e2e8f0;margin:0;'>
    🌬️ AQI Predictor
  </h1>
  <p style='color:#64748b;font-size:0.95rem;'>
    Air Quality Index · Real-time · 3-Day Forecast · {city_name}
  </p>
</div>
""", unsafe_allow_html=True)

# ── Action Handlers ───────────────────────────────────────────────────────────
if btn_fetch:
    with st.spinner("Fetching live data from Open-Meteo..."):
        try:
            raw  = fetch_combined(lat, lon)
            feat = compute_features(raw)
            n    = upsert_features(feat, city=city_name)
            st.success(f"✅ Fetched {n} rows for {city_name}")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"❌ {e}")

if btn_backfill:
    with st.spinner("Backfilling 45 days..."):
        try:
            from backfill import backfill
            n = backfill(lat=lat, lon=lon, city=city_name, days=45)
            st.success(f"✅ Backfill done — {n} rows")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"❌ {e}")

if btn_train:
    with st.spinner("Training RF, XGBoost, LightGBM..."):
        try:
            results = train(city=city_name)
            st.success("✅ Training complete!")
            for name, m in results.items():
                if "error" not in m:
                    st.info(f"**{name}** → RMSE:{m['RMSE']} | MAE:{m['MAE']} | R²:{m['R2']}")
                else:
                    st.warning(f"**{name}** failed: {m['error']}")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"❌ {e}")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_data(city, _uri, _db, days):
    df = load_features(city, limit=10000)
    if df.empty:
        return df
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    return df[df["timestamp"] >= cutoff].copy()

@st.cache_data(ttl=1800)
def get_all_data(city, _uri, _db):
    return load_features(city, limit=10000)

df_view = get_data(city_name, mongo_uri, db_name, days_history)
df_all  = get_all_data(city_name, mongo_uri, db_name)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Current AQI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">📍 Current Air Quality</p>', unsafe_allow_html=True)

curr_aqi = curr_pm25 = curr_pm10 = curr_no2 = curr_o3 = curr_temp = curr_humid = None
if not df_all.empty:
    last = df_all.iloc[-1]
    curr_aqi   = last.get("aqi")
    curr_pm25  = last.get("pm2_5")
    curr_pm10  = last.get("pm10")
    curr_no2   = last.get("nitrogen_dioxide")
    curr_o3    = last.get("ozone")
    curr_temp  = last.get("temperature_2m")
    curr_humid = last.get("relative_humidity_2m")

lbl, bg, fg = aqi_info(curr_aqi)

# ── Hazardous Alert ──────────────────────────────────────────────────────────
if curr_aqi:
    if curr_aqi > 300:
        st.markdown(f"""<div class="alert-box" style="background:rgba(126,0,35,0.3);
            border:2px solid #7e0023;color:#fca5a5;">
            🚨 HAZARDOUS AQI {int(curr_aqi)} — Stay indoors! Avoid all outdoor activity.
            Health emergency conditions. Everyone is affected.</div>""",
            unsafe_allow_html=True)
    elif curr_aqi > 200:
        st.markdown(f"""<div class="alert-box" style="background:rgba(143,63,151,0.3);
            border:2px solid #8f3f97;color:#e9d5ff;">
            ⚠️ VERY UNHEALTHY AQI {int(curr_aqi)} — Avoid prolonged outdoor exertion.
            Sensitive groups should stay indoors.</div>""",
            unsafe_allow_html=True)
    elif curr_aqi > 150:
        st.markdown(f"""<div class="alert-box" style="background:rgba(255,0,0,0.2);
            border:2px solid #ff0000;color:#fca5a5;">
            ⚠️ UNHEALTHY AQI {int(curr_aqi)} — Everyone may experience health effects.
            Reduce outdoor activity.</div>""",
            unsafe_allow_html=True)
    elif curr_aqi <= 50:
        st.markdown(f"""<div class="alert-box" style="background:rgba(0,228,0,0.1);
            border:2px solid #00e400;color:#86efac;">
            ✅ GOOD AQI {int(curr_aqi)} — Air quality is satisfactory. Enjoy outdoor activities!</div>""",
            unsafe_allow_html=True)

def sfmt(v, d=1):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{round(float(v),d)}"

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("🌫️ AQI",         sfmt(curr_aqi,  0))
c2.metric("🔴 PM2.5 µg/m³", sfmt(curr_pm25))
c3.metric("🟠 PM10 µg/m³",  sfmt(curr_pm10))
c4.metric("🟣 NO₂ µg/m³",   sfmt(curr_no2))
c5.metric("🔵 O₃ µg/m³",    sfmt(curr_o3))
c6.metric("🌡️ Temp °C",     sfmt(curr_temp))
c7.metric("💧 Humidity %",   sfmt(curr_humid, 0))

if curr_aqi:
    st.markdown(f"""
    <div style='text-align:center;margin:12px 0;'>
      <span style='background:{bg};color:{fg};padding:10px 30px;
          border-radius:50px;font-weight:700;font-size:1rem;
          letter-spacing:1px;text-transform:uppercase;'>
        {lbl}
      </span>
      <span style='color:#64748b;font-size:0.85rem;margin-left:12px;'>
        Last updated: {df_all.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M") if not df_all.empty else "N/A"}
      </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 3-Day Forecast
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">📅 3-Day AQI Forecast</p>', unsafe_allow_html=True)

model_obj, feat_cols_saved, model_metrics = load_model(model_choice)

if model_obj is None:
    st.warning("⚠️ No trained model found. Click **Train All Models** in the sidebar.")
else:
    if not df_all.empty and feat_cols_saved:
        valid_cols = [c for c in feat_cols_saved if c in df_all.columns]
        df_feat    = df_all[valid_cols].copy()
        df_feat    = df_feat.ffill()

        if not df_feat.dropna().empty:
            X_all   = df_feat.ffill().fillna(0).values
            preds   = model_obj.predict(X_all)

            # Build forecast timestamps (6h ahead per prediction)
            pred_times = df_all["timestamp"] + timedelta(hours=6)

            fcast_df = pd.DataFrame({
                "timestamp":     pred_times.values,
                "predicted_aqi": np.clip(preds, 0, 500),
            })
            fcast_df["date"]     = pd.to_datetime(fcast_df["timestamp"]).dt.date
            fcast_df["category"] = fcast_df["predicted_aqi"].apply(lambda x: aqi_info(x)[0])
            fcast_df["color"]    = fcast_df["predicted_aqi"].apply(lambda x: aqi_info(x)[1])

            # Future only
            now      = pd.Timestamp.now()
            future   = fcast_df[pd.to_datetime(fcast_df["timestamp"]) >= now]
            daily    = future.groupby("date")["predicted_aqi"].mean().reset_index().head(3)

            if len(daily) > 0:
                day_cols = st.columns(max(len(daily), 1))
                for i, (_, row) in enumerate(daily.iterrows()):
                    v = round(row["predicted_aqi"])
                    l, b, f2 = aqi_info(v)
                    day_cols[i].markdown(f"""
                    <div class="metric-card">
                      <div style='color:#94a3b8;font-size:0.85rem;margin-bottom:8px;'>
                        {pd.Timestamp(row["date"]).strftime("%A, %b %d")}
                      </div>
                      <div style='font-size:2.8rem;font-weight:700;color:#e2e8f0;
                                  font-family:"JetBrains Mono",monospace;'>{v}</div>
                      <span style='background:{b};color:{f2};padding:6px 18px;
                          border-radius:50px;font-weight:700;font-size:0.75rem;
                          margin-top:8px;display:inline-block;'>{l}</span>
                    </div>""", unsafe_allow_html=True)

            # Forecast chart
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=fcast_df["timestamp"], y=fcast_df["predicted_aqi"],
                mode="lines", line=dict(color="#60a5fa", width=2.5, shape="spline"),
                fill="tozeroy", fillcolor="rgba(96,165,250,0.08)", name="Forecast AQI"
            ))
            for lo, hi, lbl2, col, _ in AQI_SCALE:
                if lo > 0:
                    fig_f.add_hline(y=lo, line_dash="dot", line_color=col,
                                    opacity=0.4, annotation_text=lbl2,
                                    annotation_font_color=col, annotation_font_size=9)
            fig_f.update_layout(**PLOT, height=300,
                title=dict(text="AQI Forecast — Next 3 Days (6h horizon model)",
                           font=dict(color="#e2e8f0", size=14)))
            st.plotly_chart(fig_f, use_container_width=True)

    if model_metrics:
        st.caption(f"Model: **{model_choice}** · R²:{model_metrics.get('R2','?')} · "
                   f"RMSE:{model_metrics.get('RMSE','?')} · MAE:{model_metrics.get('MAE','?')} · "
                   f"Target: 6h ahead")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Historical Trends
# ══════════════════════════════════════════════════════════════════════════════
if not df_view.empty:
    st.markdown("---")
    st.markdown('<p class="section-title">📈 Historical Trends</p>', unsafe_allow_html=True)

    fig_h = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.4], vertical_spacing=0.04)

    fig_h.add_trace(go.Scatter(
        x=df_view["timestamp"], y=df_view["aqi"],
        name="AQI", mode="lines",
        line=dict(color="#60a5fa", width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.07)"
    ), row=1, col=1)

    # Rolling 24h
    roll = df_view["aqi"].rolling(24, min_periods=1).mean()
    fig_h.add_trace(go.Scatter(
        x=df_view["timestamp"], y=roll,
        name="24h Mean", mode="lines",
        line=dict(color="#f97316", width=2, dash="dash")
    ), row=1, col=1)

    for pol, color in [("pm2_5","#f87171"),("pm10","#fb923c"),
                        ("ozone","#a78bfa"),("nitrogen_dioxide","#34d399")]:
        if pol in df_view.columns:
            fig_h.add_trace(go.Scatter(
                x=df_view["timestamp"], y=df_view[pol],
                name=pol.upper().replace("_"," "), mode="lines",
                line=dict(width=1.5), opacity=0.8
            ), row=2, col=1)

    fig_h.update_layout(**PLOT, height=450,
        title=dict(text=f"AQI & Pollutants — Last {days_history} Days",
                   font=dict(color="#e2e8f0", size=14)))
    fig_h.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig_h, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EDA (PDF requirement: "Perform EDA to identify trends")
# ══════════════════════════════════════════════════════════════════════════════
if not df_all.empty:
    st.markdown("---")
    st.markdown('<p class="section-title">🔬 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⏰ Time Patterns", "📊 Distributions", "🌡️ Weather Impact"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            if "hour" in df_all.columns:
                h_avg = df_all.groupby("hour")["aqi"].mean().reset_index()
                fig_h2 = px.bar(h_avg, x="hour", y="aqi",
                    color="aqi", color_continuous_scale=["#00e400","#ffff00","#ff7e00","#ff0000","#8f3f97"],
                    title="Avg AQI by Hour of Day",
                    labels={"aqi":"Avg AQI","hour":"Hour"})
                fig_h2.update_layout(**PLOT, height=280, coloraxis_showscale=False)
                st.plotly_chart(fig_h2, use_container_width=True)

        with col_b:
            if "day_of_week" in df_all.columns:
                days = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
                d_avg = df_all.groupby("day_of_week")["aqi"].mean().reset_index()
                d_avg["day"] = d_avg["day_of_week"].map(days)
                fig_d = px.bar(d_avg, x="day", y="aqi",
                    color="aqi", color_continuous_scale=["#00e400","#ffff00","#ff0000"],
                    title="Avg AQI by Day of Week")
                fig_d.update_layout(**PLOT, height=280, coloraxis_showscale=False)
                st.plotly_chart(fig_d, use_container_width=True)

        # Heatmap hour vs day
        if "hour" in df_all.columns and "day_of_week" in df_all.columns:
            pivot = df_all.pivot_table(values="aqi", index="hour",
                                        columns="day_of_week", aggfunc="mean")
            pivot.columns = [days.get(c,c) for c in pivot.columns]
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                colorscale="RdYlGn_r", colorbar=dict(title="AQI"),
                text=np.round(pivot.values,1), texttemplate="%{text}"
            ))
            fig_hm.update_layout(**PLOT, height=380,
                title=dict(text="AQI Heatmap: Hour × Day of Week",
                           font=dict(color="#e2e8f0")))
            st.plotly_chart(fig_hm, use_container_width=True)

    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            fig_dist = px.histogram(df_all.dropna(subset=["aqi"]), x="aqi", nbins=40,
                color_discrete_sequence=["#60a5fa"], title="AQI Distribution")
            fig_dist.update_layout(**PLOT, height=280)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_d:
            # AQI category pie
            def get_cat(v):
                if pd.isna(v): return "Unknown"
                for lo,hi,lbl,_,__ in AQI_SCALE:
                    if lo<=v<=hi: return lbl
                return "Hazardous"
            df_all["cat"] = df_all["aqi"].apply(get_cat)
            cat_c = df_all["cat"].value_counts()
            cat_colors = {"Good":"#00e400","Moderate":"#ffff00",
                          "Unhealthy for Sensitive Groups":"#ff7e00",
                          "Unhealthy":"#ff0000","Very Unhealthy":"#8f3f97",
                          "Hazardous":"#7e0023","Unknown":"#555"}
            fig_pie = go.Figure(go.Pie(
                labels=cat_c.index.tolist(),
                values=cat_c.values.tolist(),
                marker_colors=[cat_colors.get(c,"#555") for c in cat_c.index],
                hole=0.4
            ))
            fig_pie.update_layout(**PLOT, height=280,
                title=dict(text="AQI Category Breakdown", font=dict(color="#e2e8f0")))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Correlation matrix
        num_cols = ["aqi","pm2_5","pm10","ozone","nitrogen_dioxide",
                    "temperature_2m","relative_humidity_2m","wind_speed_10m"]
        num_cols = [c for c in num_cols if c in df_all.columns]
        if len(num_cols) >= 3:
            corr = df_all[num_cols].corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale="RdBu_r", zmid=0,
                text=corr.round(2).values, texttemplate="%{text}",
            ))
            fig_corr.update_layout(**PLOT, height=350,
                title=dict(text="Correlation Matrix", font=dict(color="#e2e8f0")))
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        col_e, col_f = st.columns(2)
        with col_e:
            if "temperature_2m" in df_all.columns:
                fig_t = px.scatter(df_all.dropna(subset=["temperature_2m","aqi"]),
                    x="temperature_2m", y="aqi", color="aqi",
                    color_continuous_scale="RdYlGn_r",
                    title="Temperature vs AQI", opacity=0.5,
                    labels={"temperature_2m":"Temp (°C)","aqi":"AQI"})
                fig_t.update_layout(**PLOT, height=280, coloraxis_showscale=False)
                st.plotly_chart(fig_t, use_container_width=True)

        with col_f:
            if "wind_speed_10m" in df_all.columns:
                fig_w = px.scatter(df_all.dropna(subset=["wind_speed_10m","aqi"]),
                    x="wind_speed_10m", y="aqi", color="aqi",
                    color_continuous_scale="RdYlGn_r",
                    title="Wind Speed vs AQI", opacity=0.5,
                    labels={"wind_speed_10m":"Wind (km/h)","aqi":"AQI"})
                fig_w.update_layout(**PLOT, height=280, coloraxis_showscale=False)
                st.plotly_chart(fig_w, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Model Performance Comparison
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">🤖 Model Performance Comparison</p>', unsafe_allow_html=True)

all_models = list_models()
if all_models:
    rows = []
    for m in all_models:
        if m.get("metrics") and "error" not in m["metrics"] and m["name"] != "best_model":
            rows.append({
                "Model":      m["name"],
                "RMSE":       m["metrics"].get("RMSE","—"),
                "MAE":        m["metrics"].get("MAE","—"),
                "R²":         m["metrics"].get("R2","—"),
                "Trained At": m.get("trained_at","—")[:19],
            })
    if rows:
        perf_df  = pd.DataFrame(rows)
        col_g, col_h = st.columns([2, 1])
        with col_g:
            fig_perf = go.Figure()
            for metric, color in [("RMSE","#f87171"),("MAE","#fb923c")]:
                fig_perf.add_trace(go.Bar(
                    name=metric, x=perf_df["Model"],
                    y=pd.to_numeric(perf_df[metric], errors="coerce"),
                    marker_color=color, opacity=0.85
                ))
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                name="R²", x=perf_df["Model"],
                y=pd.to_numeric(perf_df["R²"], errors="coerce"),
                marker_color="#34d399", opacity=0.85
            ))
            tab_rmse, tab_r2 = st.tabs(["RMSE & MAE", "R² Score"])
            with tab_rmse:
                fig_perf.update_layout(**PLOT, barmode="group", height=280,
                    title=dict(text="Error Metrics (lower=better)",
                               font=dict(color="#e2e8f0")))
                st.plotly_chart(fig_perf, use_container_width=True)
            with tab_r2:
                fig_r2.update_layout(**PLOT, height=280,
                    title=dict(text="R² Score (higher=better, max=1.0)",
                               font=dict(color="#e2e8f0")))
                st.plotly_chart(fig_r2, use_container_width=True)
        with col_h:
            st.dataframe(perf_df[["Model","RMSE","MAE","R²"]],
                         use_container_width=True, hide_index=True)
else:
    st.info("No trained models yet. Click **Train All Models** in the sidebar.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SHAP Feature Importance (PDF requirement)
# ══════════════════════════════════════════════════════════════════════════════
if model_obj is not None and not df_all.empty:
    st.markdown("---")
    st.markdown('<p class="section-title">🧠 SHAP Feature Importance</p>', unsafe_allow_html=True)
    with st.expander("View Feature Importance (SHAP)", expanded=True):
        with st.spinner("Computing SHAP values..."):
            imp_df, _ = get_shap_values(model_choice, city_name, n_samples=min(200, len(df_all)))
        if imp_df is not None:
            shap_theme = {k:v for k,v in PLOT.items() if k != "yaxis"}
            fig_shap = px.bar(imp_df.head(15), x="importance", y="feature",
                orientation="h", color="importance",
                color_continuous_scale=["#1e3a5f","#60a5fa"],
                title=f"Top 15 Features — {model_choice}")
            fig_shap.update_layout(**shap_theme, height=420,
                coloraxis_showscale=False, showlegend=False,
                yaxis=dict(autorange="reversed",
                           gridcolor="rgba(255,255,255,0.06)",
                           showline=False, zeroline=False))
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("SHAP values show how much each feature contributes to the prediction.")
        else:
            st.info("SHAP requires a trained tree model.")

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#475569;font-size:0.8rem;padding:12px 0;'>
  Data: <a href='https://open-meteo.com' target='_blank' style='color:#60a5fa;'>Open-Meteo</a>
  (free, no API key) &nbsp;·&nbsp;
  Store: MongoDB Atlas &nbsp;·&nbsp;
  Models: Random Forest · XGBoost · LightGBM &nbsp;·&nbsp;
  CI/CD: GitHub Actions
</div>
""", unsafe_allow_html=True)
