"""Unit tests for California/Ames housing ingestion, legacy adapter, and feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from housing_valuation.core.config import load_config
from housing_valuation.core.exceptions import DataValidationError
from housing_valuation.data.ingestion import generate_synthetic_housing_dataset, ingest_data
from housing_valuation.data.preprocessing import (
    HousingFeatureEngineer,
    build_preprocessing_pipeline,
    haversine_distance_km,
)
from housing_valuation.data.validation import (
    adapt_legacy_boston_payload,
    validate_housing_dataframe,
    validate_single_payload,
)


def test_synthetic_dataset_and_ingestion(tmp_path: Path) -> None:
    cfg = load_config()
    df = generate_synthetic_housing_dataset(n_rows=120, random_state=7)
    assert len(df) == 120
    assert "B" not in df.columns
    assert "LSTAT" not in df.columns
    assert "median_house_value" in df.columns
    assert (df["median_house_value"] > 0).all()

    validated = validate_housing_dataframe(df, cfg, require_target=True)
    assert len(validated) == 120

    csv_source = tmp_path / "sample_housing.csv"
    df.to_csv(csv_source, index=False)
    train_df, test_df = ingest_data(cfg, source_path=csv_source)
    assert len(train_df) + len(test_df) == 120


def test_legacy_boston_adapter_strips_unethical_columns() -> None:
    cfg = load_config()
    legacy_payload = {
        "CRIM": 0.045,
        "ZN": 18.0,
        "INDUS": 2.31,
        "CHAS": 1,
        "NOX": 0.538,
        "RM": 6.575,
        "AGE": 65.2,
        "DIS": 4.09,
        "RAD": 1,
        "TAX": 296.0,
        "PTRATIO": 15.3,
        "B": 396.90,
        "LSTAT": 4.98,
    }
    adapted = adapt_legacy_boston_payload(legacy_payload)
    assert "B" not in adapted
    assert "LSTAT" not in adapted
    assert adapted["avg_rooms"] == pytest.approx(6.575)
    assert adapted["house_age"] == pytest.approx(65.2)
    assert adapted["ocean_proximity"] == "NEAR BAY"

    validated = validate_single_payload(legacy_payload, cfg)
    assert validated["median_income"] > 0


def test_validation_rejects_invalid_bounds() -> None:
    cfg = load_config()
    bad_payload = {
        "median_income": 5.2,
        "house_age": 20.0,
        "avg_rooms": 4.0,
        "avg_bedrooms": 6.5,  # exceeds avg_rooms
        "population": 900.0,
        "avg_occupancy": 2.5,
        "latitude": 37.77,
        "longitude": -122.41,
        "lot_area_sqft": 5000.0,
        "overall_quality": 7,
        "ocean_proximity": "NEAR BAY",
    }
    with pytest.raises(DataValidationError, match="avg_bedrooms"):
        validate_single_payload(bad_payload, cfg)


def test_feature_engineering_and_pipeline_shapes() -> None:
    cfg = load_config()
    df = generate_synthetic_housing_dataset(n_rows=60, random_state=19)
    df.loc[0, "avg_bedrooms"] = df.loc[0, "avg_rooms"] * 1.5  # trigger ratio cap

    engineer = HousingFeatureEngineer()
    transformed_df = engineer.fit_transform(df.drop(columns=["median_house_value"]))
    assert transformed_df["bedroom_to_room_ratio"].max() <= 1.0
    assert (transformed_df["min_metro_distance_km"] >= 0.0).all()

    dist_zero = haversine_distance_km(
        np.array([37.7749]),
        np.array([-122.4194]),
        37.7749,
        -122.4194,
    )
    assert dist_zero[0] == pytest.approx(0.0, abs=1e-5)

    pipeline = build_preprocessing_pipeline(cfg)
    X_mat = pipeline.fit_transform(df.drop(columns=["median_house_value"]))
    assert X_mat.shape[0] == 60
    assert np.isfinite(X_mat).all()
