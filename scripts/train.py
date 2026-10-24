"""CLI script to execute housing valuation point + quantile training pipeline."""

from __future__ import annotations

from housing_valuation.core.config import load_config
from housing_valuation.core.logger import get_logger, log_event
from housing_valuation.models.trainer import train_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Load config and train housing valuation models."""
    config = load_config()
    metadata = train_pipeline(config)
    log_event(
        logger,
        "cli_training_finished",
        model_name=metadata["model_name"],
        metrics=metadata["metrics"],
    )


if __name__ == "__main__":
    main()
