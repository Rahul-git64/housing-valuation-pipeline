"""Integration tests for FastAPI housing valuation endpoints and affordability drift supervision."""

from __future__ import annotations

from fastapi.testclient import TestClient

from housing_valuation.api.main import create_app
from housing_valuation.core.config import load_config


def test_health_and_predict_endpoints() -> None:
    cfg = load_config()
    app = create_app(cfg)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    h_body = health.json()
    assert h_body["status"] == "healthy"
    assert "r2" in h_body["model_metrics"]

    payload = {
        "median_income": 7.2,
        "house_age": 16.0,
        "avg_rooms": 6.8,
        "avg_bedrooms": 1.3,
        "population": 1350.0,
        "avg_occupancy": 2.6,
        "latitude": 37.78,
        "longitude": -122.41,
        "lot_area_sqft": 6200.0,
        "overall_quality": 8,
        "ocean_proximity": "NEAR BAY",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["price_p10_usd"] <= pred["price_p50_usd"] <= pred["price_p90_usd"]
    assert pred["price_p10_usd"] <= pred["predicted_price_usd"] <= pred["price_p90_usd"]
    assert pred["valuation_tier"] in {"Economy", "Mid-Market", "Prime", "Luxury"}
    assert pred["income_to_price_ratio"] > 0


def test_legacy_endpoint_and_drift_supervision() -> None:
    cfg = load_config()
    app = create_app(cfg)
    client = TestClient(app)

    # Insufficient data initially
    drift_init = client.get("/monitor/drift")
    assert drift_init.status_code == 200
    assert drift_init.json()["status"] == "insufficient_data"

    # Send 6 legacy Boston payloads with unethical columns that must be stripped
    for rm in [5.5, 6.0, 6.4, 6.8, 7.1, 7.5]:
        legacy_payload = {
            "CRIM": 0.03,
            "ZN": 15.0,
            "CHAS": 1,
            "RM": rm,
            "AGE": 30.0,
            "DIS": 3.5,
            "B": 390.0,
            "LSTAT": 5.2,
        }
        r = client.post("/predict/legacy", json=legacy_payload)
        assert r.status_code == 200

    drift_after = client.get("/monitor/drift")
    assert drift_after.status_code == 200
    body = drift_after.json()
    assert body["sample_size"] == 6
    assert body["status"] in {"healthy", "drift_detected"}
    assert "income_to_price_ratio_drift" in body
    assert "median_income" in body["numerical_features"]
