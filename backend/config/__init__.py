"""Environment-based configuration for the backend."""

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
	environment: str
	debug: bool
	secret_key: str
	database_url: str | None


config = Config(
	environment=os.getenv("APP_ENV", "development"),
	debug=_as_bool(os.getenv("DEBUG"), default=False),
	secret_key=os.getenv("SECRET_KEY", "change-me-in-production"),
	database_url=os.getenv("DATABASE_URL"),
)
