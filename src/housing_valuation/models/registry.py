"""Model registry for loading point + quantile estimators and executing single/batch valuations."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from housing_valuation.core.config import AppConfig
from housing_valuation.core.exceptions import InferenceError
from housing_valuation.core.logger import get_logger, log_event
from housing_valuation.data.validation import validate_single_payload
from housing_valuation.models.trainer import train_pipeline

logger = get_logger(__name__)


def classify_valuation_tier(price_usd: float) -> str:
    """Categorize predicted property value into regional market tiers."""
    if price_usd < 200000.0:
        return "Economy"
    if price_usd < 350000.0:
        return "Mid-Market"
    if price_usd < 500000.0:
        return "Prime"
    return "Luxury"


class ModelRegistry:
    """Manages lifecycle of preprocessor, point/quantile models, metadata, and reference stats."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.preprocessor: Any = None
        self.model_bundle: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.reference_stats: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return bool(self.preprocessor is not None and self.model_bundle)

    def ensure_loaded(self, auto_train: bool = True) -> None:
        """Load artifacts from disk or trigger training if missing."""
        if self.is_loaded:
            return

        m_cfg = self.config.model
        artifacts_exist = (
            m_cfg.preprocessor_path.exists()
            and m_cfg.model_path.exists()
            and m_cfg.metadata_path.exists()
            and m_cfg.reference_stats_path.exists()
        )

        if not artifacts_exist:
            if not auto_train:
                raise InferenceError("Model artifacts not found and auto_train is disabled.")
            log_event(logger, "auto_training_triggered", reason="missing_artifacts")
            train_pipeline(self.config)

        try:
            self.preprocessor = joblib.load(m_cfg.preprocessor_path)
            self.model_bundle = joblib.load(m_cfg.model_path)
            with m_cfg.metadata_path.open("r", encoding="utf-8") as f_meta:
                self.metadata = json.load(f_meta)
            with m_cfg.reference_stats_path.open("r", encoding="utf-8") as f_ref:
                self.reference_stats = json.load(f_ref)
        except Exception as exc:
            raise InferenceError(f"Failed to load model artifacts: {exc}") from exc

    def predict_single(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate property payload and predict point estimate + monotonic p10/p50/p90 bounds."""
        self.ensure_loaded(auto_train=True)
        validated = validate_single_payload(payload, self.config)

        input_df = pd.DataFrame([validated])
        try:
            X_trans = self.preprocessor.transform(input_df)
            point_pred = float(self.model_bundle["point_model"].predict(X_trans)[0])
            raw_p10 = float(self.model_bundle["p10_model"].predict(X_trans)[0])
            raw_p50 = float(self.model_bundle["p50_model"].predict(X_trans)[0])
            raw_p90 = float(self.model_bundle["p90_model"].predict(X_trans)[0])
        except Exception as exc:
            raise InferenceError(f"Model inference execution failed: {exc}") from exc

        # Guarantee non-crossing monotonic quantile bounds
        ordered_q = sorted([max(25000.0, raw_p10), max(25000.0, raw_p50), max(25000.0, raw_p90)])
        p10_usd = round(ordered_q[0], 2)
        p50_usd = round(ordered_q[1], 2)
        p90_usd = round(ordered_q[2], 2)
        point_usd = round(float(np.clip(point_pred, p10_usd, p90_usd)), 2)

        annual_income_usd = max(validated["median_income"] * 10000.0, 1000.0)
        income_to_price_ratio = round(point_usd / annual_income_usd, 3)

        rel_width = (p90_usd - p10_usd) / max(point_usd, 1.0)
        confidence_score = round(float(np.clip(1.0 - 0.55 * rel_width, 0.50, 0.98)), 3)

        return {
            "predicted_price_usd": point_usd,
            "price_p10_usd": p10_usd,
            "price_p50_usd": p50_usd,
            "price_p90_usd": p90_usd,
            "valuation_tier": classify_valuation_tier(point_usd),
            "income_to_price_ratio": income_to_price_ratio,
            "confidence_score": confidence_score,
            "adapted_features": validated,
            "model_name": str(self.metadata.get("model_name", "GradientBoostingRegressor+QuantileGBR")),
            "model_version": str(self.metadata.get("model_version", self.config.version)),
        }
