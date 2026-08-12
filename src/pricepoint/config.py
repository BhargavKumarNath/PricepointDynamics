"""Centralized configuration management.

Loads settings from config.yaml and provides typed, validated access to all
paths, hyperparameters, and thresholds used throughout the pipeline.

Built on ``pydantic-settings`` rather than hand-rolled dataclass parsing:
values are validated (wrong type/missing key fails fast with a clear error
instead of a downstream ``KeyError`` or silent ``AttributeError``), and any
setting can be overridden via an environment variable
(``PRICEPOINT_<SECTION>__<FIELD>``, e.g. ``PRICEPOINT_MODEL__OUTPUT_DIR``)
without editing ``config.yaml`` -- useful once the API runs in a container
where the image's config file isn't meant to be edited per-deployment.
Environment overrides take priority over the YAML file but not over an
explicit ``config_path`` passed to :func:`load_settings`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

logger = logging.getLogger(__name__)

# src/pricepoint/config.py -> src/pricepoint -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _resolve_path(v: Path) -> Path:
    """Resolve a config path relative to PROJECT_ROOT, unless already absolute."""
    return v if v.is_absolute() else PROJECT_ROOT / v


class DataConfig(BaseModel):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    external_dir: Path
    raw_files: list[str]

    _resolve_dirs = field_validator("raw_dir", "interim_dir", "processed_dir", "external_dir", mode="after")(
        _resolve_path
    )


class ModelConfig(BaseModel):
    output_dir: Path
    model_filename: str
    lgbm_params: dict[str, Any]

    _resolve_dir = field_validator("output_dir", mode="after")(_resolve_path)

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.model_filename


class FeaturesConfig(BaseModel):
    rolling_windows: list[int]
    lag_days: list[int]
    output_filename: str


class MatchingConfig(BaseModel):
    model_name: str
    similarity_threshold: float
    mutual_neighbors_k: int
    faiss_nprobe: int
    output_filename: str


class ShapConfig(BaseModel):
    sample_size: int
    output_dir: Path
    random_seed: int

    _resolve_dir = field_validator("output_dir", mode="after")(_resolve_path)


class MarketDynamicsConfig(BaseModel):
    output_dir: Path
    sample_size: int
    max_lag_days: int
    min_correlation: float
    min_stores_for_common: int

    _resolve_dir = field_validator("output_dir", mode="after")(_resolve_path)


class AnomalyConfig(BaseModel):
    contamination: float
    random_state: int


class BenchmarkingConfig(BaseModel):
    n_iterations: int
    warmup_iterations: int


class LoggingConfig(BaseModel):
    level: str
    format: str
    file: str | None = None


class Settings(BaseSettings):
    """Application settings loaded from config.yaml, overridable via env vars."""

    model_config = SettingsConfigDict(
        yaml_file=DEFAULT_CONFIG_PATH,
        env_prefix="PRICEPOINT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data: DataConfig
    model: ModelConfig
    features: FeaturesConfig
    matching: MatchingConfig
    shap: ShapConfig
    market_dynamics: MarketDynamicsConfig
    anomaly: AnomalyConfig
    benchmarking: BenchmarkingConfig
    logging: LoggingConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence (highest first): explicit init kwargs > env vars > YAML
        # file > secret files. Env vars deliberately outrank the YAML file so
        # a container deployment can override a setting without editing the
        # image's config.yaml.
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings from a YAML config file.

    Parameters
    ----------
    config_path : Path, optional
        Path to the config file. Defaults to ``<project_root>/config.yaml``.
        When given explicitly, its values take priority over environment
        variables (the caller asked for *this* file specifically).

    Returns
    -------
    Settings
        Fully resolved, validated settings object.

    Raises
    ------
    pydantic.ValidationError
        If a required field is missing or fails type validation.
    """
    if config_path is None:
        logger.info("Loading configuration from %s", DEFAULT_CONFIG_PATH)
        # mypy sees a BaseModel with required fields and expects them as
        # constructor kwargs; it doesn't know pydantic-settings fills them
        # in at runtime from the YAML/env sources registered in
        # `settings_customise_sources` above -- a known, common false
        # positive for BaseSettings subclasses (verified: this call
        # succeeds and raises pydantic.ValidationError, not TypeError, if
        # the YAML is genuinely incomplete).
        return Settings()  # type: ignore[call-arg]

    logger.info("Loading configuration from %s", config_path)
    yaml_source = YamlConfigSettingsSource(Settings, yaml_file=config_path)
    return Settings(**yaml_source())
