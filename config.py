"""Application configuration via Pydantic settings.

Reads from environment variables and ``.env`` files.  All settings are
prefixed with ``CAUSALMERGE_`` to avoid collisions with other tools.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("causalmerge")

# ── Settings ──────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Central configuration for CausalMerge.

    Values are loaded in this priority order:
    1. Explicit constructor arguments
    2. Environment variables (prefixed ``CAUSALMERGE_``)
    3. ``.env`` file in the working directory
    4. Built-in defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="CAUSALMERGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Merge settings ────────────────────────────────────────────────────
    confidence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum merged confidence for including an edge in the output graph.",
    )
    cycle_resolution: str = Field(
        default="min_weight",
        description=(
            "Strategy for breaking cycles in the merged graph. "
            "Options: 'min_weight' (remove lowest-confidence edge), "
            "'oldest_source' (remove edge from oldest contributing source)."
        ),
    )

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ── API server ───────────────────────────────────────────────────────
    api_host: str = Field(default="127.0.0.1", description="API listen address.")
    api_port: int = Field(default=8000, description="API listen port.")

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log level: {v!r}")
        return v

    @field_validator("cycle_resolution")
    @classmethod
    def _validate_cycle_resolution(cls, v: str) -> str:
        allowed = ("min_weight", "oldest_source")
        if v not in allowed:
            raise ValueError(f"cycle_resolution must be one of {allowed}, got {v!r}")
        return v


# ── Singleton accessor ────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    """Set up Python logging based on application settings."""
    settings = settings or get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-16s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.debug("CausalMerge logging configured at %s level", settings.log_level)
