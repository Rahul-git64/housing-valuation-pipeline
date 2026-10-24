"""Streamlit interactive regional housing valuation explorer and drift monitor."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Housing Valuation Pipeline",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Regional Housing Valuation & Quantile Interval Explorer")
st.markdown(
    "Estimate property valuations with **90% Quantile Regression bounds (`p10`, `p50`, `p90`)**, "
    "Haversine metro distance features, and real-time **Income-to-Price Affordability Drift** supervision."
)

col1, col2, col3 = st.columns(3)

with col1:
    median_income = st.slider("Median Household Income ($10k USD)", 1.0, 15.0, 6.2, 0.1)
    house_age = st.slider("Property Age (Years)", 1.0, 100.0, 22.0, 1.0)
    overall_quality = st.slider("Overall Structural Quality (1-10)", 1, 10, 7)
    ocean_proximity = st.selectbox(
        "Coastal Proximity",
        ["<1H OCEAN", "NEAR BAY", "NEAR OCEAN", "INLAND", "ISLAND"],
    )

with col2:
    avg_rooms = st.number_input("Average Total Rooms", min_value=1.5, max_value=20.0, value=6.0, step=0.2)
    avg_bedrooms = st.number_input("Average Bedrooms", min_value=0.5, max_value=float(avg_rooms), value=1.2, step=0.1)
    lot_area_sqft = st.number_input("Lot Area (sq ft)", min_value=500.0, max_value=50000.0, value=5500.0, step=250.0)

with col3:
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.5, value=37.7749, format="%.4f")
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.4194, format="%.4f")
    population = st.number_input("Block Population", min_value=100.0, max_value=25000.0, value=1200.0, step=100.0)
    avg_occupancy = st.number_input("Average Household Occupancy", min_value=1.0, max_value=10.0, value=2.7, step=0.1)

if st.button("Calculate Property Valuation", type="primary"):
    payload = {
        "median_income": median_income,
        "house_age": house_age,
        "avg_rooms": avg_rooms,
        "avg_bedrooms": avg_bedrooms,
        "population": population,
        "avg_occupancy": avg_occupancy,
        "latitude": latitude,
        "longitude": longitude,
        "lot_area_sqft": lot_area_sqft,
        "overall_quality": overall_quality,
        "ocean_proximity": ocean_proximity,
    }
    try:
        resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Point Valuation", f"${data['predicted_price_usd']:,.0f}")
        m2.metric("90% Interval (p10 - p90)", f"${data['price_p10_usd']:,.0f} – ${data['price_p90_usd']:,.0f}")
        m3.metric("Market Tier", data["valuation_tier"])
        m4.metric("Price-to-Income Ratio", f"{data['income_to_price_ratio']:.2f}x")
    except Exception as exc:
        st.error(f"Valuation request failed: {exc}")

st.divider()
if st.button("Inspect Covariate & Affordability Drift (/monitor/drift)"):
    try:
        drift_resp = httpx.get(f"{API_URL}/monitor/drift", timeout=10.0)
        st.json(drift_resp.json())
    except Exception as exc:
        st.error(f"Drift telemetry request failed: {exc}")
