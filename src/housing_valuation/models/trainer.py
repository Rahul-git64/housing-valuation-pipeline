"""Point and Quantile Regression model trainer, evaluator, and reference stats exporter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)

from housing_valuation.core.config import AppConfig
from housing_valuation.core.exceptions import ModelTrainingError
from housing_valuation.core.logger import get_logger, log_event
from housing_valuation.data.ingestion import ingest_data
from housing_valuation.data.preprocessing import build_preprocessing_pipeline

logger = get_logger(__name__)


def evaluate_point_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics: RMSE, MAE, MAPE, and R2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape_pct": round(mape, 2),
        "r2": round(r2, 4),
    }


def evaluate_quantile_intervals(
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> dict[str, float]:
    """Compute quantile pinball losses, empirical interval coverage, and median width."""
    coverage = float(np.mean((y_true >= p10) & (y_true <= p90)))
    median_width = float(np.median(p90 - p10))
    return {
        "pinball_p10": round(float(mean_pinball_loss(y_true, p10, alpha=0.10)), 2),
        "pinball_p50": round(float(mean_pinball_loss(y_true, p50, alpha=0.50)), 2),
        "pinball_p90": round(float(mean_pinball_loss(y_true, p90, alpha=0.90)), 2),
        "empirical_coverage_p10_p90": round(coverage, 4),
        "median_interval_width_usd": round(median_width, 2),
    }


def build_reference_stats(train_df: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Compute reference distributions for numerical features, categories, and income-to-price ratio."""
    numerical_stats: dict[str, Any] = {}
    rng = np.random.default_rng(config.data.random_state)

    for col in config.features.numerical_columns:
        series = train_df[col].astype(float).to_numpy()
        sample_size = min(len(series), 400)
        sampled = rng.choice(series, size=sample_size, replace=False)
        quantiles = np.quantile(series, np.linspace(0.0, 1.0, 11)).tolist()
        numerical_stats[col] = {
            "mean": round(float(np.mean(series)), 4),
            "std": round(float(np.std(series)), 4),
            "quantiles": [round(float(q), 4) for q in quantiles],
            "sample": [round(float(x), 4) for x in sampled],
        }

    categorical_stats: dict[str, dict[str, float]] = {}
    for col in config.features.categorical_columns:
        counts = train_df[col].astype(str).value_counts(normalize=True)
        categorical_stats[col] = {str(k): round(float(v), 4) for k, v in counts.items()}

    # Compute baseline price-to-income ratio (house value / annual household income)
    annual_income_usd = np.maximum(train_df["median_income"].astype(float).to_numpy() * 10000.0, 1000.0)
    price_to_income = train_df[config.features.target_column].astype(float).to_numpy() / annual_income_usd
    pti_sample = rng.choice(price_to_income, size=min(len(price_to_income), 400), replace=False)
    pti_quantiles = np.quantile(price_to_income, np.linspace(0.0, 1.0, 11)).tolist()

    return {
        "n_reference": len(train_df),
        "numerical": numerical_stats,
        "categorical": categorical_stats,
        "income_to_price_ratio": {
            "mean": round(float(np.mean(price_to_income)), 4),
            "std": round(float(np.std(price_to_income)), 4),
            "quantiles": [round(float(q), 4) for q in pti_quantiles],
            "sample": [round(float(x), 4) for x in pti_sample],
        },
    }


def train_pipeline(config: AppConfig) -> dict[str, Any]:
    """Execute end-to-end training for point estimator and p10/p50/p90 quantile regressors."""
    train_df, test_df = ingest_data(config)
    target_col = config.features.target_column

    X_train_raw = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(float).to_numpy()
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(float).to_numpy()

    preprocessor = build_preprocessing_pipeline(config)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    seed = config.data.random_state
    candidates: dict[str, Any] = {
        "Ridge": Ridge(alpha=1.5),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=140,
            max_depth=14,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=180,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            random_state=seed,
        ),
    }

    leaderboard: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model: Any = None
    best_r2 = -np.inf

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_point_predictions(y_test, preds)
        leaderboard[name] = metrics
        log_event(logger, "candidate_model_evaluated", model=name, **metrics)
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_name = name
            best_model = model

    if best_r2 < config.model.r2_threshold:
        raise ModelTrainingError(
            f"Best point model {best_name} R2 ({best_r2:.4f}) below threshold ({config.model.r2_threshold:.4f})"
        )

    # Train quantile regression estimators for p10, p50, and p90
    q_cfg = config.model.quantiles
    quantile_models: dict[str, GradientBoostingRegressor] = {}
    for q_label, alpha in [("p10", q_cfg["p10"]), ("p50", q_cfg["p50"]), ("p90", q_cfg["p90"])]:
        q_reg = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=160,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            random_state=seed,
        )
        q_reg.fit(X_train, y_train)
        quantile_models[q_label] = q_reg

    raw_p10 = quantile_models["p10"].predict(X_test)
    raw_p50 = quantile_models["p50"].predict(X_test)
    raw_p90 = quantile_models["p90"].predict(X_test)

    # Enforce monotonic quantile ordering on test evaluation
    sorted_quantiles = np.sort(np.vstack([raw_p10, raw_p50, raw_p90]), axis=0)
    interval_metrics = evaluate_quantile_intervals(
        y_test,
        sorted_quantiles[0],
        sorted_quantiles[1],
        sorted_quantiles[2],
    )

    config.model.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.model.preprocessor_path)

    model_bundle = {
        "point_model": best_model,
        "p10_model": quantile_models["p10"],
        "p50_model": quantile_models["p50"],
        "p90_model": quantile_models["p90"],
    }
    joblib.dump(model_bundle, config.model.model_path)

    combined_metrics = {**leaderboard[best_name], **interval_metrics}
    metadata: dict[str, Any] = {
        "model_name": f"{best_name}+QuantileGBR",
        "model_version": config.version,
        "trained_at": datetime.now(UTC).isoformat(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": combined_metrics,
        "leaderboard": leaderboard,
    }

    with config.model.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    ref_stats = build_reference_stats(train_df, config)
    with config.model.reference_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(ref_stats, handle, indent=2)

    log_event(
        logger,
        "model_training_completed",
        winner=metadata["model_name"],
        r2=combined_metrics["r2"],
        rmse=combined_metrics["rmse"],
        coverage_p10_p90=combined_metrics["empirical_coverage_p10_p90"],
    )
    return metadata
