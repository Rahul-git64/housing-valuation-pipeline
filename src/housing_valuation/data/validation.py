"""Schema validation and backward-compatible legacy Boston-to-California/Ames adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from housing_valuation.core.config import AppConfig
from housing_valuation.core.exceptions import DataValidationError

DEPRECATED_DEMOGRAPHIC_COLUMNS: frozenset[str] = frozenset({"B", "b", "LSTAT", "lstat"})


def adapt_legacy_boston_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Adapt legacy Boston housing keys to the modern California/Ames schema while stripping unethical fields."""
    cleaned = {k: v for k, v in payload.items() if k not in DEPRECATED_DEMOGRAPHIC_COLUMNS}

    # Check if modern schema is already present
    if "median_income" in cleaned and "avg_rooms" in cleaned:
        return {
            "median_income": float(cleaned["median_income"]),
            "house_age": float(cleaned.get("house_age", 25.0)),
            "avg_rooms": float(cleaned["avg_rooms"]),
            "avg_bedrooms": float(cleaned.get("avg_bedrooms", max(1.0, float(cleaned["avg_rooms"]) * 0.2))),
            "population": float(cleaned.get("population", 1200.0)),
            "avg_occupancy": float(cleaned.get("avg_occupancy", 2.8)),
            "latitude": float(cleaned.get("latitude", 37.7749)),
            "longitude": float(cleaned.get("longitude", -122.4194)),
            "lot_area_sqft": float(cleaned.get("lot_area_sqft", 5500.0)),
            "overall_quality": int(cleaned.get("overall_quality", 6)),
            "ocean_proximity": str(cleaned.get("ocean_proximity", "<1H OCEAN")),
        }

    # Map legacy Boston structural keys (RM, AGE, TAX, DIS, CHAS, ZN)
    if "RM" in cleaned or "rm" in cleaned:
        rm = float(cleaned.get("RM", cleaned.get("rm", 5.8)))
        age = float(cleaned.get("AGE", cleaned.get("Age", cleaned.get("age", 35.0))))
        zn = float(cleaned.get("ZN", cleaned.get("zn", 12.5)))
        chas = int(float(cleaned.get("CHAS", cleaned.get("chas", 0))))
        dis = float(cleaned.get("DIS", cleaned.get("dis", 3.8)))

        inferred_income = round(max(1.5, min(14.0, (rm - 2.5) * 1.45 + zn * 0.02)), 3)
        inferred_bedrooms = round(max(1.0, rm * 0.21), 3)
        inferred_lot = round(max(1800.0, 4200.0 + zn * 65.0), 1)
        inferred_prox = "NEAR BAY" if chas == 1 else ("<1H OCEAN" if dis < 4.5 else "INLAND")

        return {
            "median_income": inferred_income,
            "house_age": min(100.0, max(1.0, age)),
            "avg_rooms": rm,
            "avg_bedrooms": inferred_bedrooms,
            "population": 1150.0,
            "avg_occupancy": 2.75,
            "latitude": 37.75 if chas == 1 else 36.85,
            "longitude": -122.35 if chas == 1 else -119.85,
            "lot_area_sqft": inferred_lot,
            "overall_quality": min(10, max(1, int(round(rm)))),
            "ocean_proximity": inferred_prox,
        }

    raise DataValidationError(
        "Payload must provide modern California/Ames fields ('median_income', 'avg_rooms', ...) "
        "or legacy structural keys ('RM', 'AGE', ...)."
    )


def validate_single_payload(payload: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Validate and normalize a single property record for inference."""
    adapted = adapt_legacy_boston_payload(payload)

    if adapted["median_income"] <= 0 or adapted["median_income"] > 50.0:
        raise DataValidationError(f"median_income out of bounds (0, 50]: {adapted['median_income']}")

    if adapted["house_age"] < 0 or adapted["house_age"] > 150.0:
        raise DataValidationError(f"house_age out of bounds [0, 150]: {adapted['house_age']}")

    if adapted["avg_rooms"] <= 0.5 or adapted["avg_rooms"] > 30.0:
        raise DataValidationError(f"avg_rooms out of bounds (0.5, 30]: {adapted['avg_rooms']}")

    if adapted["avg_bedrooms"] <= 0.0 or adapted["avg_bedrooms"] > adapted["avg_rooms"]:
        raise DataValidationError(
            f"avg_bedrooms ({adapted['avg_bedrooms']}) must be positive and <= avg_rooms ({adapted['avg_rooms']})"
        )

    if adapted["population"] <= 0 or adapted["population"] > 100000:
        raise DataValidationError(f"population out of bounds: {adapted['population']}")

    if adapted["avg_occupancy"] <= 0.2 or adapted["avg_occupancy"] > 25.0:
        raise DataValidationError(f"avg_occupancy out of bounds (0.2, 25]: {adapted['avg_occupancy']}")

    if not (32.0 <= adapted["latitude"] <= 42.5):
        raise DataValidationError(f"latitude out of regional bounds [32.0, 42.5]: {adapted['latitude']}")

    if not (-125.0 <= adapted["longitude"] <= -114.0):
        raise DataValidationError(f"longitude out of regional bounds [-125.0, -114.0]: {adapted['longitude']}")

    if adapted["lot_area_sqft"] < 300.0 or adapted["lot_area_sqft"] > 250000.0:
        raise DataValidationError(f"lot_area_sqft out of bounds [300, 250000]: {adapted['lot_area_sqft']}")

    if not (1 <= int(adapted["overall_quality"]) <= 10):
        raise DataValidationError(f"overall_quality must be in [1, 10]: {adapted['overall_quality']}")

    allowed_prox = set(config.features.categories.get("ocean_proximity", []))
    if adapted["ocean_proximity"] not in allowed_prox:
        raise DataValidationError(
            f"Unsupported ocean_proximity '{adapted['ocean_proximity']}'. Allowed: {sorted(allowed_prox)}"
        )

    return adapted


def validate_housing_dataframe(
    df: pd.DataFrame,
    config: AppConfig,
    require_target: bool = True,
) -> pd.DataFrame:
    """Validate a housing DataFrame against schema, range constraints, and ethical filters."""
    if df.empty:
        raise DataValidationError("Input housing DataFrame cannot be empty.")

    # Drop any deprecated columns if present
    cleaned_df = df.drop(columns=[c for c in df.columns if c in DEPRECATED_DEMOGRAPHIC_COLUMNS], errors="ignore").copy()

    required_cols = list(config.features.numerical_columns) + list(config.features.categorical_columns)
    if require_target:
        required_cols.append(config.features.target_column)

    missing = [col for col in required_cols if col not in cleaned_df.columns]
    if missing:
        raise DataValidationError(f"Missing required housing columns: {missing}")

    if cleaned_df[required_cols].isnull().any().any():
        cleaned_df = cleaned_df.dropna(subset=required_cols).reset_index(drop=True)
        if cleaned_df.empty:
            raise DataValidationError("All rows dropped due to null values in required columns.")

    if (cleaned_df["median_income"] <= 0).any():
        raise DataValidationError("median_income must be strictly positive.")

    if (cleaned_df["avg_rooms"] <= 0).any() or (cleaned_df["avg_bedrooms"] <= 0).any():
        raise DataValidationError("avg_rooms and avg_bedrooms must be strictly positive.")

    if (cleaned_df["avg_bedrooms"] > cleaned_df["avg_rooms"]).any():
        raise DataValidationError("avg_bedrooms cannot exceed avg_rooms.")

    allowed_prox = set(config.features.categories.get("ocean_proximity", []))
    invalid_prox = set(cleaned_df["ocean_proximity"].astype(str).unique()) - allowed_prox
    if invalid_prox:
        raise DataValidationError(f"Invalid ocean_proximity values found: {sorted(invalid_prox)}")

    if require_target and (cleaned_df[config.features.target_column] <= 0).any():
        raise DataValidationError(f"{config.features.target_column} must be strictly positive.")

    return cleaned_df
