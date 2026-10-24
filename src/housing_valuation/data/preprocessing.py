"""Haversine geospatial feature engineering and scikit-learn preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from housing_valuation.core.config import AppConfig

EARTH_RADIUS_KM = 6371.0088

METRO_COORDS: dict[str, tuple[float, float]] = {
    "sf": (37.7749, -122.4194),
    "la": (34.0522, -118.2437),
    "sd": (32.7157, -117.1611),
    "sj": (37.3382, -121.8863),
}


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Compute vectorized great-circle Haversine distance in kilometers."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_KM * c


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain feature transformer adding structural ratios and metro proximity distances."""

    ENGINEERED_COLUMNS: tuple[str, ...] = (
        "bedroom_to_room_ratio",
        "rooms_per_person",
        "income_per_room",
        "log_lot_area",
        "dist_to_sf_km",
        "dist_to_la_km",
        "min_metro_distance_km",
    )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HousingFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        safe_rooms = np.maximum(df["avg_rooms"].astype(float), 0.5)
        safe_occ = np.maximum(df["avg_occupancy"].astype(float), 0.25)
        safe_lot = np.maximum(df["lot_area_sqft"].astype(float), 100.0)

        df["bedroom_to_room_ratio"] = df["avg_bedrooms"].astype(float) / safe_rooms
        df["rooms_per_person"] = safe_rooms / safe_occ
        df["income_per_room"] = df["median_income"].astype(float) / safe_rooms
        df["log_lot_area"] = np.log1p(safe_lot)

        lat = df["latitude"].astype(float).to_numpy()
        lon = df["longitude"].astype(float).to_numpy()

        dist_sf = haversine_distance_km(lat, lon, *METRO_COORDS["sf"])
        dist_la = haversine_distance_km(lat, lon, *METRO_COORDS["la"])
        dist_sd = haversine_distance_km(lat, lon, *METRO_COORDS["sd"])
        dist_sj = haversine_distance_km(lat, lon, *METRO_COORDS["sj"])

        df["dist_to_sf_km"] = dist_sf
        df["dist_to_la_km"] = dist_la
        df["min_metro_distance_km"] = np.minimum.reduce([dist_sf, dist_la, dist_sd, dist_sj])
        return df


def build_preprocessing_pipeline(config: AppConfig) -> Pipeline:
    """Create a fitted-ready sklearn Pipeline with feature engineering, imputation, and scaling."""
    engineered_num_cols = list(config.features.numerical_columns) + list(
        HousingFeatureEngineer.ENGINEERED_COLUMNS
    )
    cat_cols = list(config.features.categorical_columns)

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_categories = [config.features.categories[col] for col in cat_cols]
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=cat_categories,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, engineered_num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_engineer", HousingFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
