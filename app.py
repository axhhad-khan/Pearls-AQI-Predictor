"""
app.py  –  Streamlit AQI Prediction Dashboard
Run:  streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
}

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.aqi-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,179,237,0.3);
}

.alert-hazardous {
    background: rgba(126,0,35,0.3);
    border: 2px solid #7e0023;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    color: #fca5a5;
    font-weight: 600;
}

.alert-good {
    background: rgba(0,228,0,0.1);
    border: 2px solid #00e400;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    color: #86efac;
    font-weight: 600;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
}

div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.5px;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem !important;
}

.stSidebar {
    background: rgba(10,14,26,0.9) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}

.plot-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 8px;
}

h1 { color: #e2e8f0 !important; }
h2 { color: #cbd5e1 !important; }
h3 { color: #94a3b8 !important; }
p, span, div { color: #94a3b8; }

.stButton>button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 24px;
    font-family: 'Space Grotesk', sans-serif;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(59,130,246,0.4);
}
</style>
""", unsafe_allow_html=True)

# ── AQI helpers ───────────────────────────────────────────────────────────────
AQI_SCALE = [
    (0,   50,  "Good",                           "#00e400", "#000"),
    (51,  100, "Moderate",                        "#ffff00", "#000"),
    (101, 150, "Unhealthy for Sensitive Groups",  "#ff7e00", "#000"),
    (151, 200, "Unhealthy",                       "#ff0000", "#fff"),
    (201, 300, "Very Unhealthy",                  "#8f3f97", "#fff"),
    (301, 500, "Hazardous",                       "#7e0023", "#fff"),
]

def aqi_info(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Unknown", "#555", "#fff"
    for lo, hi, label, bg, fg in AQI_SCALE:
        if lo <= val <= hi:
            return label, bg, fg
    return "Hazardous", "#7e0023", "#fff"


PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Space Grotesk"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    city_name = st.text_input("City Name", value=os.getenv("CITY_NAME", "Karachi"))
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude",  value=float(os.getenv("LATITUDE",  "24.8607")), format="%.4f")
    with col2:
        lon = st.number_input("Longitude", value=float(os.getenv("LONGITUDE", "67.0011")), format="%.4f")

    mongo_uri = st.text_input(
        "MongoDB URI",
        value=os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
        type="password",
        help="Paste your MongoDB Atlas connection string"
    )
    db_name = st.text_input("Database Name", value=os.getenv("MONGO_DB_NAME", "aqi_predictor"))

    st.markdown("---")
    st.markdown("### 🔄 Data Management")

    run_fetch = st.button("📡 Fetch Live Data", use_container_width=True)
    run_backfill = st.button("🗃️ Backfill 30 Days", use_container_width=True)
    run_train = st.button("🤖 Train Models", use_container_width=True)

    st.markdown("---")
    model_choice = st.selectbox(
        "Prediction Model",
        ["best_model", "RandomForest", "XGBoost", "LightGBM"],
        index=0
    )


# ── Inject runtime env ────────────────────────────────────────────────────────
os.environ["MONGO_URI"]     = mongo_uri
os.environ["MONGO_DB_NAME"] = db_name

# ── Imports (after env is set) ────────────────────────────────────────────────
from pipelines.fetch_data       import fetch_combined
from pipelines.feature_pipeline import compute_features, aqi_category
from utils.db                   import upsert_features, load_features, load_model, list_models, load_forecast_features
from pipelines.training_pipeline import train, get_shap_values


# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; padding: 20px 0 10px 0;'>
  <h1 style='font-size:2.8rem; font-weight:700; color:#e2e8f0; margin:0;'>
    🌬️ AQI Predictor
  </h1>
  <p style='color:#64748b; font-size:1rem; margin-top:6px;'>
    Air Quality Index · 3-Day Forecast · {city_name}
  </p>
</div>
""", unsafe_allow_html=True)


# ── Action handlers ───────────────────────────────────────────────────────────
if run_fetch:
    with st.spinner("Fetching live data from Open-Meteo…"):
        try:
            raw  = fetch_combined(lat, lon)
            feat = compute_features(raw)
            n    = upsert_features(feat, city=city_name)
            st.success(f"✅ Fetched and stored {n} hourly rows for {city_name}")
        except Exception as e:
            st.error(f"❌ Fetch failed: {e}")

if run_backfill:
    with st.spinner("Backfilling 30 days of historical data…"):
        try:
            from backfill import backfill
            n = backfill(lat=lat, lon=lon, city=city_name, days=30)
            st.success(f"✅ Backfill complete – {n} rows stored")
        except Exception as e:
            st.error(f"❌ Backfill failed: {e}")

if run_train:
    with st.spinner("Training Random Forest, XGBoost, and LightGBM…"):
        try:
            results = train(city=city_name)
            st.success("✅ Training complete!")
            for name, metrics in results.items():
                if "error" not in metrics:
                    st.info(f"**{name}** → RMSE: {metrics['RMSE']} | MAE: {metrics['MAE']} | R²: {metrics['R2']}")
                else:
                    st.warning(f"**{name}** failed: {metrics['error']}")
        except Exception as e:
            st.error(f"❌ Training failed: {e}")


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_feature_data(city, _mongo_uri, _db_name):
    return load_features(city, limit=5000)

df_hist = get_feature_data(city_name, mongo_uri, db_name)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Current AQI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-title">📍 Current Air Quality</p>', unsafe_allow_html=True)

current_aqi    = None
current_pm25   = None
current_pm10   = None
current_no2    = None
current_o3     = None
current_temp   = None

if not df_hist.empty:
    last = df_hist.iloc[-1]
    current_aqi  = last.get("aqi")
    current_pm25 = last.get("pm2_5")
    current_pm10 = last.get("pm10")
    current_no2  = last.get("nitrogen_dioxide")
    current_o3   = last.get("ozone")
    current_temp = last.get("temperature_2m")

label, bg_color, fg_color = aqi_info(current_aqi)

# Alert for hazardous conditions
if current_aqi and current_aqi > 200:
    st.markdown(f"""
    <div class="alert-hazardous">
        ⚠️ HAZARDOUS AIR QUALITY ALERT — AQI {int(current_aqi)} ({label})
        <br>Limit outdoor activities. Sensitive groups should stay indoors.
    </div>
    """, unsafe_allow_html=True)
elif current_aqi and current_aqi <= 50:
    st.markdown(f"""
    <div class="alert-good">
        ✅ Air quality is GOOD (AQI {int(current_aqi)}) — Great day to be outdoors!
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

def safe_metric(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{round(float(val), decimals)}"

col1.metric("🌫️ AQI",           safe_metric(current_aqi, 0), help="US Air Quality Index")
col2.metric("🔴 PM2.5 (µg/m³)", safe_metric(current_pm25))
col3.metric("🟠 PM10 (µg/m³)",  safe_metric(current_pm10))
col4.metric("🟣 NO₂ (µg/m³)",   safe_metric(current_no2))
col5.metric("🔵 O₃ (µg/m³)",    safe_metric(current_o3))
col6.metric("🌡️ Temp (°C)",     safe_metric(current_temp))

if current_aqi:
    st.markdown(f"""
    <div style='text-align:center; margin: 16px 0;'>
      <span class="aqi-badge" style="background:{bg_color}; color:{fg_color}; font-size:1rem; padding:10px 30px;">
        {label}
      </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – 3-Day Forecast
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-title">📅 3-Day AQI Forecast</p>', unsafe_allow_html=True)

model_obj, feat_cols_saved, model_metrics = load_model(model_choice)

if model_obj is None:
    st.warning("⚠️ No trained model found. Click **Train Models** in the sidebar first.")
else:
    # Build forecast from the most recent features
    if not df_hist.empty and feat_cols_saved:
        valid_cols = [c for c in feat_cols_saved if c in df_hist.columns]
        X_latest   = df_hist[valid_cols].dropna().tail(72)   # use last 72 rows

        if not X_latest.empty:
            preds = model_obj.predict(X_latest.values)
            pred_times = df_hist.loc[X_latest.index, "timestamp"] + timedelta(hours=24)

            forecast_df = pd.DataFrame({
                "timestamp": pred_times.values,
                "predicted_aqi": preds
            })
            forecast_df["category"] = forecast_df["predicted_aqi"].apply(
                lambda x: aqi_info(x)[0]
            )
            forecast_df["color"] = forecast_df["predicted_aqi"].apply(
                lambda x: aqi_info(x)[1]
            )

            # Daily summary
            forecast_df["date"] = pd.to_datetime(forecast_df["timestamp"]).dt.date
            daily = forecast_df.groupby("date")["predicted_aqi"].mean().reset_index()
            daily = daily.head(3)

            day_cols = st.columns(len(daily))
            for i, (_, row) in enumerate(daily.iterrows()):
                aqi_val = round(row["predicted_aqi"])
                lbl, bg, fg = aqi_info(aqi_val)
                day_cols[i].markdown(f"""
                <div class="metric-card">
                  <div style='color:#94a3b8; font-size:0.85rem; margin-bottom:8px;'>
                    {row['date'].strftime('%A, %b %d')}
                  </div>
                  <div style='font-size:2.5rem; font-weight:700; color:#e2e8f0;
                              font-family: JetBrains Mono, monospace;'>
                    {aqi_val}
                  </div>
                  <span class="aqi-badge" style="background:{bg}; color:{fg}; font-size:0.75rem; margin-top:8px; display:inline-block;">
                    {lbl}
                  </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Hourly forecast chart
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["predicted_aqi"],
                mode="lines",
                line=dict(color="#60a5fa", width=2.5, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(96,165,250,0.08)",
                name="Predicted AQI",
            ))
            # Add AQI threshold lines
            for lo, hi, lbl, col, _ in AQI_SCALE:
                fig_forecast.add_hline(
                    y=lo, line_dash="dot", line_color=col,
                    opacity=0.4, annotation_text=lbl if lo > 0 else "",
                    annotation_font_color=col, annotation_font_size=10
                )

            fig_forecast.update_layout(
                **PLOT_THEME,
                title=dict(text="Hourly AQI Forecast (Next 3 Days)", font=dict(color="#e2e8f0", size=15)),
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        if model_metrics:
            r2   = model_metrics.get("R2", "N/A")
            rmse = model_metrics.get("RMSE", "N/A")
            mae  = model_metrics.get("MAE", "N/A")
            st.caption(f"Model: **{model_choice}** · R²: {r2} · RMSE: {rmse} · MAE: {mae}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Historical Trends
# ─────────────────────────────────────────────────────────────────────────────
if not df_hist.empty:
    st.markdown("---")
    st.markdown('<p class="section-title">📈 Historical AQI Trends</p>', unsafe_allow_html=True)

    days_back = st.slider("Show last N days", 1, 30, 7, key="hist_slider")
    cutoff    = pd.Timestamp.now() - pd.Timedelta(days=days_back)
    df_view   = df_hist[df_hist["timestamp"] >= cutoff]

    if not df_view.empty:
        fig_hist = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.6, 0.4], vertical_spacing=0.05)

        # AQI line
        fig_hist.add_trace(go.Scatter(
            x=df_view["timestamp"], y=df_view["aqi"],
            name="AQI", line=dict(color="#60a5fa", width=2),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.06)"
        ), row=1, col=1)

        # Pollutants
        for pol, color in [("pm2_5","#f87171"), ("pm10","#fb923c"),
                            ("ozone","#a78bfa"), ("nitrogen_dioxide","#34d399")]:
            if pol in df_view.columns:
                fig_hist.add_trace(go.Scatter(
                    x=df_view["timestamp"], y=df_view[pol],
                    name=pol.replace("_"," ").upper(),
                    line=dict(width=1.5), opacity=0.8
                ), row=2, col=1)

        fig_hist.update_layout(
            **PLOT_THEME,
            height=450,
            title=dict(text=f"AQI & Pollutants — Last {days_back} Days",
                       font=dict(color="#e2e8f0", size=15)),
        )
        fig_hist.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_hist.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – EDA
# ─────────────────────────────────────────────────────────────────────────────
if not df_hist.empty:
    st.markdown("---")
    st.markdown('<p class="section-title">🔬 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Hourly AQI pattern
    with col_a:
        if "hour" in df_hist.columns:
            hourly_avg = df_hist.groupby("hour")["aqi"].mean().reset_index()
            fig_hour = px.bar(hourly_avg, x="hour", y="aqi",
                              color="aqi",
                              color_continuous_scale=["#00e400","#ffff00","#ff7e00","#ff0000","#8f3f97"],
                              labels={"aqi":"Avg AQI","hour":"Hour of Day"},
                              title="Average AQI by Hour of Day")
            fig_hour.update_layout(**PLOT_THEME, height=280, showlegend=False,
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_hour, use_container_width=True)

    # AQI distribution
    with col_b:
        fig_dist = px.histogram(df_hist.dropna(subset=["aqi"]), x="aqi", nbins=40,
                                color_discrete_sequence=["#60a5fa"],
                                title="AQI Distribution")
        fig_dist.update_layout(**PLOT_THEME, height=280, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Correlation heatmap
    num_cols = ["aqi","pm2_5","pm10","ozone","nitrogen_dioxide",
                "temperature_2m","relative_humidity_2m","wind_speed_10m"]
    num_cols = [c for c in num_cols if c in df_hist.columns]
    if len(num_cols) >= 3:
        corr = df_hist[num_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu_r", zmid=0,
            text=corr.round(2).values, texttemplate="%{text}",
            showscale=True,
        ))
        fig_corr.update_layout(**PLOT_THEME, height=350,
                               title=dict(text="Correlation Matrix",
                                          font=dict(color="#e2e8f0")))
        st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Model Performance & Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-title">🤖 Model Performance</p>', unsafe_allow_html=True)

all_models = list_models()
if all_models:
    metrics_rows = []
    for m in all_models:
        if m.get("metrics") and "error" not in m["metrics"]:
            metrics_rows.append({
                "Model":      m["name"],
                "RMSE":       m["metrics"].get("RMSE","—"),
                "MAE":        m["metrics"].get("MAE","—"),
                "R²":         m["metrics"].get("R2","—"),
                "Trained At": m.get("trained_at","—"),
            })
    if metrics_rows:
        perf_df = pd.DataFrame(metrics_rows)
        # Bar chart comparison
        fig_perf = go.Figure()
        model_names = [r for r in perf_df["Model"] if r != "best_model"]
        perf_sub    = perf_df[perf_df["Model"].isin(model_names)]
        for metric, color in [("RMSE","#f87171"),("MAE","#fb923c"),("R²","#34d399")]:
            if metric in perf_sub.columns:
                fig_perf.add_trace(go.Bar(
                    name=metric, x=perf_sub["Model"],
                    y=pd.to_numeric(perf_sub[metric], errors="coerce"),
                    marker_color=color, opacity=0.85
                ))
        fig_perf.update_layout(
            **PLOT_THEME, barmode="group", height=300,
            title=dict(text="Model Comparison (lower RMSE/MAE = better · higher R² = better)",
                       font=dict(color="#e2e8f0", size=13)),
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        st.dataframe(perf_df.style.set_properties(**{"background-color":"rgba(0,0,0,0)"}),
                     use_container_width=True)
else:
    st.info("No trained models yet. Click **Train Models** in the sidebar.")

# SHAP Feature Importance
if model_obj is not None and not df_hist.empty:
    with st.expander("🧠 Feature Importance (SHAP)", expanded=False):
        with st.spinner("Computing SHAP values…"):
            importance_df, _ = get_shap_values(model_choice, city_name, n_samples=100)
        if importance_df is not None:
            fig_shap = px.bar(importance_df.head(15), x="importance", y="feature",
                              orientation="h",
                              color="importance",
                              color_continuous_scale=["#1e3a5f","#60a5fa"],
                              title=f"Top 15 Features – {model_choice}")
            shap_theme = {k: v for k, v in PLOT_THEME.items() if k != "yaxis"}
            fig_shap.update_layout(**shap_theme, height=400, showlegend=False,
                                   coloraxis_showscale=False,
                                   yaxis=dict(autorange="reversed",
                                              gridcolor="rgba(255,255,255,0.06)",
                                              showline=False, zeroline=False))
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP not available for this model (requires tree-based model).")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.8rem; padding:16px 0;'>
  Data: <a href='https://open-meteo.com' target='_blank' style='color:#60a5fa;'>Open-Meteo</a> (free, no API key)
  · Store: MongoDB Atlas · Models: Random Forest · XGBoost · LightGBM
  · Automation: GitHub Actions
</div>
""", unsafe_allow_html=True)