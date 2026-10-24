"""FastAPI application with quantile valuation inference and affordability drift supervision."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from housing_valuation.api.monitoring import DriftSupervisor
from housing_valuation.api.schemas import (
    DriftReportResponse,
    HealthResponse,
    HousingPredictionRequest,
    HousingPredictionResponse,
)
from housing_valuation.core.config import AppConfig, load_config
from housing_valuation.core.exceptions import HousingValuationError
from housing_valuation.core.logger import get_logger, log_event
from housing_valuation.models.registry import ModelRegistry

logger = get_logger(__name__)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI housing valuation and drift supervision application."""
    cfg = config or load_config()
    registry = ModelRegistry(cfg)
    supervisor = DriftSupervisor(cfg)
    start_time = time.monotonic()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        log_event(logger, "service_startup_initiated", version=cfg.version)
        registry.ensure_loaded(auto_train=True)
        log_event(
            logger,
            "service_startup_complete",
            active_model=registry.metadata.get("model_name", "unknown"),
        )
        yield

    app = FastAPI(
        title="Housing Valuation Pipeline Service",
        description=(
            "Regional California & Ames housing valuation API with p10/p50/p90 quantile intervals "
            "and covariate + affordability drift supervision."
        ),
        version=cfg.version,
        lifespan=lifespan,
    )

    app.state.config = cfg
    app.state.registry = registry
    app.state.supervisor = supervisor
    app.state.start_time = start_time

    @app.get("/health", response_model=HealthResponse, tags=["Supervision"])
    def health_check() -> HealthResponse:
        """Return service health, uptime, and active model metadata."""
        app.state.registry.ensure_loaded(auto_train=True)
        uptime = round(time.monotonic() - app.state.start_time, 2)
        meta = app.state.registry.metadata
        metrics = {
            k: float(v)
            for k, v in meta.get("metrics", {}).items()
            if isinstance(v, (int, float))
        }
        return HealthResponse(
            status="healthy",
            service=cfg.project_name,
            version=cfg.version,
            uptime_seconds=uptime,
            active_model=str(meta.get("model_name", "unknown")),
            model_metrics=metrics,
            buffered_requests=len(app.state.supervisor.buffer),
        )

    @app.post("/predict", response_model=HousingPredictionResponse, tags=["Inference"])
    def predict_valuation(request: HousingPredictionRequest) -> HousingPredictionResponse:
        """Validate property features, predict point + p10/p50/p90 bounds, and record drift telemetry."""
        t0 = time.perf_counter()
        payload = request.model_dump()
        try:
            prediction = app.state.registry.predict_single(payload)
        except HousingValuationError as exc:
            log_event(logger, "inference_failure", error=exc.message)
            raise HTTPException(status_code=422, detail=exc.message) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        app.state.supervisor.record_request(
            prediction["adapted_features"],
            prediction["income_to_price_ratio"],
        )

        log_event(
            logger,
            "valuation_completed",
            ocean_proximity=request.ocean_proximity,
            predicted_price_usd=prediction["predicted_price_usd"],
            price_p10_usd=prediction["price_p10_usd"],
            price_p90_usd=prediction["price_p90_usd"],
            valuation_tier=prediction["valuation_tier"],
            income_to_price_ratio=prediction["income_to_price_ratio"],
            latency_ms=latency_ms,
        )

        return HousingPredictionResponse(
            predicted_price_usd=prediction["predicted_price_usd"],
            price_p10_usd=prediction["price_p10_usd"],
            price_p50_usd=prediction["price_p50_usd"],
            price_p90_usd=prediction["price_p90_usd"],
            valuation_tier=prediction["valuation_tier"],
            income_to_price_ratio=prediction["income_to_price_ratio"],
            confidence_score=prediction["confidence_score"],
            model_name=prediction["model_name"],
            model_version=prediction["model_version"],
            latency_ms=latency_ms,
        )

    @app.post("/predict/legacy", response_model=HousingPredictionResponse, tags=["Inference"])
    def predict_legacy_payload(payload: dict[str, Any]) -> HousingPredictionResponse:
        """Backward-compatible endpoint adapting legacy structural keys while stripping unethical columns."""
        t0 = time.perf_counter()
        try:
            prediction = app.state.registry.predict_single(payload)
        except HousingValuationError as exc:
            log_event(logger, "legacy_inference_failure", error=exc.message)
            raise HTTPException(status_code=422, detail=exc.message) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        app.state.supervisor.record_request(
            prediction["adapted_features"],
            prediction["income_to_price_ratio"],
        )

        return HousingPredictionResponse(
            predicted_price_usd=prediction["predicted_price_usd"],
            price_p10_usd=prediction["price_p10_usd"],
            price_p50_usd=prediction["price_p50_usd"],
            price_p90_usd=prediction["price_p90_usd"],
            valuation_tier=prediction["valuation_tier"],
            income_to_price_ratio=prediction["income_to_price_ratio"],
            confidence_score=prediction["confidence_score"],
            model_name=prediction["model_name"],
            model_version=prediction["model_version"],
            latency_ms=latency_ms,
        )

    @app.get("/monitor/drift", response_model=DriftReportResponse, tags=["Supervision"])
    def monitor_drift() -> DriftReportResponse:
        """Evaluate covariate and income-to-price affordability drift on buffered requests."""
        app.state.registry.ensure_loaded(auto_train=True)
        report = app.state.supervisor.evaluate(app.state.registry.reference_stats)
        log_event(
            logger,
            "drift_evaluation_executed",
            status=report["status"],
            sample_size=report["sample_size"],
            overall_drift_detected=report["overall_drift_detected"],
        )
        return DriftReportResponse(**report)

    return app


app = create_app()
