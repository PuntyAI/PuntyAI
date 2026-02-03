"""Application configuration using Pydantic settings."""

import secrets
from datetime import date, datetime
from pathlib import Path
from functools import lru_cache
from zoneinfo import ZoneInfo

MELB_TZ = ZoneInfo("Australia/Melbourne")

# Generate a random secret key for development if not configured
_DEFAULT_SECRET_KEY = secrets.token_hex(32)


def melb_now() -> datetime:
    """Current time in Melbourne (AEDT/AEST automatically)."""
    return datetime.now(MELB_TZ)


def melb_now_naive() -> datetime:
    """Current time in Melbourne as naive datetime (for SQLAlchemy defaults).

    SQLite doesn't handle timezone-aware datetimes well, so we store
    Melbourne local time as naive datetime. This function is used as
    the default factory for model created_at/updated_at fields.
    """
    return melb_now().replace(tzinfo=None)


def melb_today() -> date:
    """Today's date in Melbourne timezone."""
    return melb_now().date()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PUNTY_",
        extra="ignore",
    )

    # Database
    db_path: Path = Path("./data/punty.db")

    # App
    secret_key: str = ""  # Will use random key if not set
    debug: bool = False
    log_level: str = "INFO"

    # OpenAI (no prefix - standard env var)
    openai_api_key: str = ""

    # Twitter/X API
    twitter_api_key: str = ""
    twitter_api_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_secret: str = ""

    # WhatsApp Business API
    whatsapp_api_token: str = ""
    whatsapp_phone_number_id: str = ""

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    allowed_emails: str = ""  # Required: comma-separated list of allowed emails

    def model_post_init(self, __context) -> None:
        """Ensure secret_key is set (generate random if not configured)."""
        if not self.secret_key:
            object.__setattr__(self, 'secret_key', _DEFAULT_SECRET_KEY)

    @property
    def database_url(self) -> str:
        """SQLite database URL for SQLAlchemy."""
        return f"sqlite+aiosqlite:///{self.db_path}"


class FullSettings(Settings):
    """Settings with OpenAI key loaded from standard env var (no PUNTY_ prefix)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PUNTY_",
        extra="ignore",
    )

    # These use standard env var names (no prefix)
    openai_api_key: str = ""

    def model_post_init(self, __context) -> None:
        """Load OpenAI key from standard env var or .env file if not set."""
        import os
        from dotenv import dotenv_values

        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            env_vals = dotenv_values(".env")
            self.openai_api_key = env_vals.get("OPENAI_API_KEY", "")


@lru_cache
def get_settings() -> FullSettings:
    """Get cached settings instance."""
    return FullSettings()


# Export for convenience
settings = get_settings()
