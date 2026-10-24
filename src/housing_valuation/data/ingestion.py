"""Deterministic offline California & Ames housing dataset generator and ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from housing_valuation.core.config import AppConfig
from housing_valuation.core.exceptions import DataIngestionError
from housing_valuation.core.logger import get_logger, log_event
from housing_valuation.data.validation import validate_housing_dataframe

logger = get_logger(__name__)

# Regional economic hubs: (latitude, longitude, base_premium_usd)
REGIONAL_HUBS: dict[str, tuple[float, float, float]] = {
    "San Francisco": (37.7749, -122.4194, 85000.0),
    "San Jose": (37.3382, -121.8863, 78000.0),
    "Los Angeles": (34.0522, -118.2437, 55000.0),
    "San Diego": (32.7157, -117.1611, 48000.0),
    "Sacramento": (38.5816, -121.4944, 18000.0),
    "Fresno": (36.7378, -119.7871, 5000.0),
}

OCEAN_PROXIMITY_PREMIUM: dict[str, float] = {
    "ISLAND": 115000.0,
    "NEAR BAY": 68000.0,
    "NEAR OCEAN": 58000.0,
    "<1H OCEAN": 32000.0,
    "INLAND": -24000.0,
}


def generate_synthetic_housing_dataset(n_rows: int = 1600, random_state: int = 42) -> pd.DataFrame:
    """Generate a deterministic, realistic California & Ames hybrid housing dataset offline."""
    rng = np.random.default_rng(random_state)

    hub_names = list(REGIONAL_HUBS.keys())
    hub_probs = [0.22, 0.16, 0.28, 0.14, 0.12, 0.08]
    chosen_hubs = rng.choice(hub_names, size=n_rows, p=hub_probs)

    latitudes = np.empty(n_rows, dtype=float)
    longitudes = np.empty(n_rows, dtype=float)
    ocean_proximities = []

    for i, hub in enumerate(chosen_hubs):
        base_lat, base_lon, _ = REGIONAL_HUBS[hub]
        lat = np.clip(base_lat + rng.normal(0.0, 0.45), 32.5, 41.9)
        lon = np.clip(base_lon + rng.normal(0.0, 0.45), -124.3, -114.2)
        latitudes[i] = round(float(lat), 4)
        longitudes[i] = round(float(lon), 4)

        if hub in ("Sacramento", "Fresno"):
            ocean_proximities.append(
                str(rng.choice(["INLAND", "<1H OCEAN"], p=[0.85, 0.15]))
            )
        elif hub == "San Francisco":
            ocean_proximities.append(
                str(rng.choice(["NEAR BAY", "NEAR OCEAN", "<1H OCEAN"], p=[0.60, 0.25, 0.15]))
            )
        else:
            ocean_proximities.append(
                str(
                    rng.choice(
                        ["<1H OCEAN", "NEAR OCEAN", "INLAND", "NEAR BAY", "ISLAND"],
                        p=[0.50, 0.25, 0.15, 0.08, 0.02],
                    )
                )
            )

    median_income = np.round(np.clip(rng.lognormal(mean=1.32, sigma=0.42, size=n_rows), 1.0, 15.0), 4)
    house_age = np.round(np.clip(rng.uniform(2.0, 52.0, size=n_rows), 1.0, 100.0), 1)
    overall_quality = np.clip(
        np.round(3.2 + 0.55 * median_income + rng.normal(0.0, 1.1, size=n_rows)),
        1,
        10,
    ).astype(int)

    avg_rooms = np.round(
        np.clip(3.1 + 0.38 * median_income + 0.18 * overall_quality + rng.normal(0.0, 0.65, size=n_rows), 2.0, 14.0),
        3,
    )
    bedroom_ratio = np.clip(0.21 - 0.006 * median_income + rng.normal(0.0, 0.025, size=n_rows), 0.12, 0.45)
    avg_bedrooms = np.round(np.maximum(0.8, avg_rooms * bedroom_ratio), 3)

    population = np.round(np.clip(rng.lognormal(mean=7.05, sigma=0.45, size=n_rows), 150.0, 12000.0), 0)
    avg_occupancy = np.round(np.clip(rng.normal(2.85, 0.55, size=n_rows), 1.2, 6.5), 3)
    lot_area_sqft = np.round(
        np.clip(2800.0 + 620.0 * avg_rooms + rng.normal(0.0, 1200.0, size=n_rows), 1200.0, 25000.0),
        1,
    )

    # Calculate true underlying price with realistic heteroscedastic variance
    prices = np.empty(n_rows, dtype=float)
    for i in range(n_rows):
        hub_premium = REGIONAL_HUBS[str(chosen_hubs[i])][2]
        prox_premium = OCEAN_PROXIMITY_PREMIUM[ocean_proximities[i]]
        income_effect = 43500.0 * median_income[i] + 1850.0 * (median_income[i] ** 1.4)
        quality_effect = 14500.0 * overall_quality[i]
        room_effect = 9200.0 * avg_rooms[i] - 11000.0 * (avg_bedrooms[i] / avg_rooms[i])
        lot_effect = 3.8 * lot_area_sqft[i]
        age_effect = -620.0 * house_age[i] + (14000.0 if house_age[i] > 42 and prox_premium > 0 else 0.0)
        occupancy_effect = -8500.0 * max(0.0, avg_occupancy[i] - 2.5)

        noise_scale = 12000.0 + 2800.0 * median_income[i]
        noise = rng.normal(0.0, noise_scale)

        raw_price = (
            62000.0
            + income_effect
            + quality_effect
            + room_effect
            + lot_effect
            + age_effect
            + occupancy_effect
            + hub_premium
            + prox_premium
            + noise
        )
        prices[i] = round(float(np.clip(raw_price, 48000.0, 850000.0)), 2)

    return pd.DataFrame(
        {
            "median_income": median_income,
            "house_age": house_age,
            "avg_rooms": avg_rooms,
            "avg_bedrooms": avg_bedrooms,
            "population": population,
            "avg_occupancy": avg_occupancy,
            "latitude": latitudes,
            "longitude": longitudes,
            "lot_area_sqft": lot_area_sqft,
            "overall_quality": overall_quality,
            "ocean_proximity": ocean_proximities,
            "median_house_value": prices,
        }
    )


def ingest_data(config: AppConfig, source_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest housing data from CSV or generate deterministic offline dataset and split train/test."""
    try:
        if source_path and source_path.exists():
            df = pd.read_csv(source_path)
            log_event(logger, "loaded_external_housing_csv", path=str(source_path), rows=len(df))
        elif config.data.raw_data_path.exists():
            df = pd.read_csv(config.data.raw_data_path)
            log_event(logger, "loaded_cached_raw_housing_csv", path=str(config.data.raw_data_path), rows=len(df))
        else:
            df = generate_synthetic_housing_dataset(
                n_rows=config.data.synthetic_rows,
                random_state=config.data.random_state,
            )
            config.data.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(config.data.raw_data_path, index=False)
            log_event(
                logger,
                "generated_synthetic_housing_dataset",
                path=str(config.data.raw_data_path),
                rows=len(df),
            )
    except Exception as exc:
        raise DataIngestionError(f"Failed to ingest housing dataset: {exc}") from exc

    validated_df = validate_housing_dataframe(df, config, require_target=True)

    train_df, test_df = train_test_split(
        validated_df,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=validated_df["ocean_proximity"],
    )

    config.data.train_data_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.data.train_data_path, index=False)
    test_df.to_csv(config.data.test_data_path, index=False)

    log_event(
        logger,
        "housing_data_split_completed",
        train_rows=len(train_df),
        test_rows=len(test_df),
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
