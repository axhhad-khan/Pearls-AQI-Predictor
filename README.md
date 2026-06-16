# Pearls AQI Predictor — End-to-End Air Quality Forecasting

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=flat&logo=mongodb&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pearls-aqi-predictor.streamlit.app)

> Predict the **Air Quality Index (AQI)** for any city for the next **3 days** using a robust, 100% serverless Machine Learning stack.

🚀 **Live Demo:** [Explore the Live Dashboard](https://pearls-aqi-predictor-11.streamlit.app)

---

## 🚀 Overview

The **Pearls AQI Predictor** is an automated, end-to-end machine learning pipeline that fetches real-time meteorological and pollutant data, engineers predictive features, trains state-of-the-art tree-based models, and serves forecasts via an interactive web dashboard. 

Built entirely on a **serverless architecture**, it requires zero dedicated infrastructure, leveraging GitHub Actions for orchestration and MongoDB Atlas as a centralized Feature Store and Model Registry.

### ✨ Key Features
* **100% Serverless:** Completely automated using GitHub Actions (cron jobs).
* **Zero Cost Data:** Integrates with the Open-Meteo API (No API keys required).
* **Advanced ML:** Utilizes Random Forest, XGBoost, and LightGBM for robust time-series forecasting.
* **Explainable AI:** Integrates SHAP values to explain model predictions.
* **Interactive UI:** A sleek Streamlit dashboard powered by Plotly visualizations.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Open-Meteo API] -->|Raw Data| B(Feature Pipeline)
    B -->|Engineered Features| C[(MongoDB Atlas<br/>Feature Store)]
    C -->|Historical Data| D{Training Pipeline<br/>RF / XGBoost / LightGBM}
    D -->|Best Model| E[(MongoDB Atlas<br/>Model Registry)]
    E -->|Inference| F[Streamlit Dashboard]
    C -->|Current Context| F
