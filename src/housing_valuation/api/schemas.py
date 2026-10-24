"""Pydantic schemas for FastAPI housing valuation inference and drift supervision."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HousingPredictionRequest(BaseModel):
    """Input payload for modern California & Ames regional housing valuation."""

    median_income: float = Field(..., gt=0.0, le=50.0, description="Median household income in $10,000s USD")
    house_age: float = Field(25.0, ge=0.0, le=150.0, description="Property age in years")
    avg_rooms: float = Field(..., gt=0.5, le=30.0, description="Average total rooms per dwelling")
    avg_bedrooms: float = Field(1.1, gt=0.0, le=15.0, description="Average bedrooms per dwelling")
    population: float = Field(1200.0, gt=0.0, le=100000.0, description="Block group population")
    avg_occupancy: float = Field(2.8, gt=0.2, le=25.0, description="Average household occupancy")
    latitude: float = Field(37.7749, ge=32.0, le=42.5, description="Property latitude")
    longitude: float = Field(-122.4194, ge=-125.0, le=-114.0, description="Property longitude")
    lot_area_sqft: float = Field(5500.0, ge=300.0, le=250000.0, description="Lot area in square feet")
    overall_quality: int = Field(6, ge=1, le=10, description="Structural condition and finish quality (1-10)")
    ocean_proximity: str = Field("<1H OCEAN", description="Coastal proximity category")


class HousingPredictionResponse(BaseModel):
    """Response payload with point valuation, p10/p50/p90 intervals, and affordability ratio."""

    predicted_price_usd: float
    price_p10_usd: float
    price_p50_usd: float
    price_p90_usd: float
    valuation_tier: str
    income_to_price_ratio: float
    confidence_score: float
    model_name: str
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Service health and active model telemetry."""

    status: str
    service: str
    version: str
    uptime_seconds: float
    active_model: str
    model_metrics: dict[str, float]
    buffered_requests: int


class NumericalDriftMetric(BaseModel):
    """KS-test and PSI drift telemetry for a single numerical feature."""

    ks_statistic: float
    ks_p_value: float
    psi: float
    mean_reference: float
    mean_current: float
    drift_detected: bool


class CategoricalDriftMetric(BaseModel):
    """L1 distribution shift telemetry for a categorical feature."""

    l1_divergence: float
    drift_detected: bool


class DriftReportResponse(BaseModel):
    """Complete covariate and income-to-price affordability drift report."""

    status: str
    overall_drift_detected: bool
    sample_size: int
    reference_size: int
    income_to_price_ratio_drift: dict[str, Any]
    numerical_features: dict[str, NumericalDriftMetric]
    categorical_features: dict[str, CategoricalDriftMetric]
