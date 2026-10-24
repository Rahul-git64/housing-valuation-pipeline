"""YAML configuration loader with cross-platform Path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from housing_valuation.core.exceptions import ConfigurationError


def get_project_root() -> Path:
    """Resolve the repository root directory across editable and site-packages installs."""
    module_parent = Path(__file__).resolve().parents[3]
    if (module_parent / "configs" / "config.yaml").exists():
        return module_parent
    return Path.cwd()


@dataclass(frozen=True)
class DataConfig:
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float
    random_state: int
    synthetic_rows: int


@dataclass(frozen=True)
class FeaturesConfig:
    target_column: str
    numerical_columns: list[str]
    categorical_columns: list[str]
    categories: dict[str, list[str]]


@dataclass(frozen=True)
class ModelConfig:
    artifacts_dir: Path
    preprocessor_filename: str
    model_filename: str
    metadata_filename: str
    reference_stats_filename: str
    r2_threshold: float
    quantiles: dict[str, float]

    @property
    def preprocessor_path(self) -> Path:
        return self.artifacts_dir / self.preprocessor_filename

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / self.metadata_filename

    @property
    def reference_stats_path(self) -> Path:
        return self.artifacts_dir / self.reference_stats_filename


@dataclass(frozen=True)
class MonitoringConfig:
    window_size: int
    ks_alpha: float
    psi_threshold: float
    categorical_l1_threshold: float
    income_price_ratio_psi_threshold: float


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    version: str
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    monitoring: MonitoringConfig


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load and validate application configuration from YAML."""
    root = get_project_root()
    resolved_path = Path(config_path) if config_path else root / "configs" / "config.yaml"
    if not resolved_path.exists():
        raise ConfigurationError(f"Configuration file not found: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse YAML config: {exc}") from exc

    try:
        data_cfg = raw["data"]
        feat_cfg = raw["features"]
        model_cfg = raw["model"]
        mon_cfg = raw["monitoring"]

        return AppConfig(
            project_name=raw["project"]["name"],
            version=raw["project"]["version"],
            data=DataConfig(
                raw_data_path=root / data_cfg["raw_data_path"],
                train_data_path=root / data_cfg["train_data_path"],
                test_data_path=root / data_cfg["test_data_path"],
                test_size=float(data_cfg["test_size"]),
                random_state=int(data_cfg["random_state"]),
                synthetic_rows=int(data_cfg.get("synthetic_rows", 1600)),
            ),
            features=FeaturesConfig(
                target_column=feat_cfg["target_column"],
                numerical_columns=list(feat_cfg["numerical_columns"]),
                categorical_columns=list(feat_cfg["categorical_columns"]),
                categories={k: list(v) for k, v in feat_cfg["categories"].items()},
            ),
            model=ModelConfig(
                artifacts_dir=root / model_cfg["artifacts_dir"],
                preprocessor_filename=model_cfg["preprocessor_filename"],
                model_filename=model_cfg["model_filename"],
                metadata_filename=model_cfg["metadata_filename"],
                reference_stats_filename=model_cfg["reference_stats_filename"],
                r2_threshold=float(model_cfg["r2_threshold"]),
                quantiles={k: float(v) for k, v in model_cfg.get("quantiles", {"p10": 0.1, "p50": 0.5, "p90": 0.9}).items()},
            ),
            monitoring=MonitoringConfig(
                window_size=int(mon_cfg["window_size"]),
                ks_alpha=float(mon_cfg["ks_alpha"]),
                psi_threshold=float(mon_cfg["psi_threshold"]),
                categorical_l1_threshold=float(mon_cfg["categorical_l1_threshold"]),
                income_price_ratio_psi_threshold=float(mon_cfg.get("income_price_ratio_psi_threshold", 0.20)),
            ),
        )
    except KeyError as exc:
        raise ConfigurationError(f"Missing required configuration key: {exc}") from exc
