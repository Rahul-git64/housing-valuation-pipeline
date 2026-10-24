"""Unit tests for quantile regression trainer, interval monotonicity, and model registry."""

from __future__ import annotations

from housing_valuation.core.config import load_config
from housing_valuation.models.registry import ModelRegistry, classify_valuation_tier
from housing_valuation.models.trainer import train_pipeline


def test_trainer_and_quantile_intervals() -> None:
    cfg = load_config()
    metadata = train_pipeline(cfg)
    assert "metrics" in metadata
    assert metadata["metrics"]["r2"] >= cfg.model.r2_threshold
    assert 0.50 <= metadata["metrics"]["empirical_coverage_p10_p90"] <= 0.99
    assert cfg.model.model_path.exists()
    assert cfg.model.preprocessor_path.exists()
    assert cfg.model.reference_stats_path.exists()


def test_model_registry_monotonic_bounds_and_tiers() -> None:
    cfg = load_config()
    registry = ModelRegistry(cfg)
    registry.ensure_loaded(auto_train=True)

    payload = {
        "median_income": 6.8,
        "house_age": 18.0,
        "avg_rooms": 6.4,
        "avg_bedrooms": 1.2,
        "population": 1150.0,
        "avg_occupancy": 2.6,
        "latitude": 37.76,
        "longitude": -122.42,
        "lot_area_sqft": 5800.0,
        "overall_quality": 8,
        "ocean_proximity": "NEAR BAY",
    }
    result = registry.predict_single(payload)
    assert result["price_p10_usd"] <= result["price_p50_usd"] <= result["price_p90_usd"]
    assert result["price_p10_usd"] <= result["predicted_price_usd"] <= result["price_p90_usd"]
    assert result["valuation_tier"] in {"Economy", "Mid-Market", "Prime", "Luxury"}
    assert result["income_to_price_ratio"] > 0

    assert classify_valuation_tier(150000.0) == "Economy"
    assert classify_valuation_tier(275000.0) == "Mid-Market"
    assert classify_valuation_tier(420000.0) == "Prime"
    assert classify_valuation_tier(650000.0) == "Luxury"
