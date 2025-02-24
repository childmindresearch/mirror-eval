"""Configuration module for the ctk_functions package."""

import functools
import logging
from pathlib import Path

import pydantic
import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    """App settings."""

    OPIK_API_KEY: str | None = pydantic.Field(None)
    OPIK_WORKSPACE: str | None = pydantic.Field(None)

    LOGGER_VERBOSITY: int = logging.INFO

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@functools.lru_cache
def get_settings() -> Settings:
    """Gets the app settings."""
    return Settings()


def get_logger() -> logging.Logger:
    """Gets the ctk-functions logger."""
    logger = logging.getLogger("ctk-functions")
    if logger.hasHandlers():
        return logger

    logger.setLevel(get_settings().LOGGER_VERBOSITY)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s - %(message)s",  # noqa: E501
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
