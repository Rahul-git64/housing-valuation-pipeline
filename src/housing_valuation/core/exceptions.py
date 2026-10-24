"""Custom exception hierarchy for housing valuation pipeline."""

from __future__ import annotations


class HousingValuationError(Exception):
    """Base exception for all housing valuation pipeline errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(HousingValuationError):
    """Raised when YAML configuration cannot be loaded or validated."""


class DataIngestionError(HousingValuationError):
    """Raised when raw dataset acquisition or parsing fails."""


class DataValidationError(HousingValuationError):
    """Raised when housing schema validation or legacy payload adaptation fails."""


class FeatureEngineeringError(HousingValuationError):
    """Raised when geospatial or structural ratio feature transformation fails."""


class ModelTrainingError(HousingValuationError):
    """Raised when quantile or point regression training fails quality gates."""


class InferenceError(HousingValuationError):
    """Raised when model artifacts are missing or valuation prediction fails."""
