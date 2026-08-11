"""Centralized configuration management.

Loads settings from config.yaml and provides typed access to all
paths, hyperparameters, and thresholds used throughout the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    external_dir: Path
    raw_files: list[str]


@dataclass(frozen=True)
class ModelConfig:
    output_dir: Path
    model_filename: str
    lgbm_params: dict[str, Any]

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.model_filename


@dataclass(frozen=True)
class FeaturesConfig:
    rolling_windows: list[int]
    lag_days: list[int]
    output_filename: str


@dataclass(frozen=True)
class MatchingConfig:
    model_name: str
    similarity_threshold: float
    mutual_neighbors_k: int
    faiss_nprobe: int
    output_filename: str


@dataclass(frozen=True)
class ShapConfig:
    sample_size: int
    output_dir: Path
    random_seed: int


@dataclass(frozen=True)
class MarketDynamicsConfig:
    output_dir: Path
    sample_size: int
    max_lag_days: int
    min_correlation: float
    min_stores_for_common: int


@dataclass(frozen=True)
class AnomalyConfig:
    contamination: float
    random_state: int


@dataclass(frozen=True)
class BenchmarkingConfig:
    n_iterations: int
    warmup_iterations: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    file: str | None


@dataclass
class Settings:
    """Immutable application settings loaded from config.yaml."""

    data: DataConfig
    model: ModelConfig
    features: FeaturesConfig
    matching: MatchingConfig
    shap: ShapConfig
    market_dynamics: MarketDynamicsConfig
    anomaly: AnomalyConfig
    benchmarking: BenchmarkingConfig
    logging: LoggingConfig


def _resolve_path(raw: str) -> Path:
    """Resolve a config path relative to PROJECT_ROOT."""
    return PROJECT_ROOT / raw


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings from a YAML config file.

    Parameters
    ----------
    config_path : Path, optional
        Path to the config file.  Defaults to ``<project_root>/config.yaml``.

    Returns
    -------
    Settings
        Fully resolved settings object.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    logger.info("Loading configuration from %s", path)

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    data_cfg = raw["data"]
    model_cfg = raw["model"]

    return Settings(
        data=DataConfig(
            raw_dir=_resolve_path(data_cfg["raw_dir"]),
            interim_dir=_resolve_path(data_cfg["interim_dir"]),
            processed_dir=_resolve_path(data_cfg["processed_dir"]),
            external_dir=_resolve_path(data_cfg["external_dir"]),
            raw_files=data_cfg["raw_files"],
        ),
        model=ModelConfig(
            output_dir=_resolve_path(model_cfg["output_dir"]),
            model_filename=model_cfg["model_filename"],
            lgbm_params=model_cfg["lgbm_params"],
        ),
        features=FeaturesConfig(**raw["features"]),
        matching=MatchingConfig(**raw["matching"]),
        shap=ShapConfig(
            sample_size=raw["shap"]["sample_size"],
            output_dir=_resolve_path(raw["shap"]["output_dir"]),
            random_seed=raw["shap"]["random_seed"],
        ),
        market_dynamics=MarketDynamicsConfig(
            output_dir=_resolve_path(raw["market_dynamics"]["output_dir"]),
            sample_size=raw["market_dynamics"]["sample_size"],
            max_lag_days=raw["market_dynamics"]["max_lag_days"],
            min_correlation=raw["market_dynamics"]["min_correlation"],
            min_stores_for_common=raw["market_dynamics"]["min_stores_for_common"],
        ),
        anomaly=AnomalyConfig(**raw["anomaly"]),
        benchmarking=BenchmarkingConfig(**raw["benchmarking"]),
        logging=LoggingConfig(**raw["logging"]),
    )
