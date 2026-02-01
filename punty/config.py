"""Application configuration using Pydantic settings."""

from pathlib import Path
from functools import lru_cache

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
    secret_key: str = "change-me-in-production"
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
    allowed_emails: str = "gerardr@gmail.com,punty@punty.ai"

    @property
    def database_url(self) -> str:
        """SQLite database URL for SQLAlchemy."""
        return f"sqlite+aiosqlite:///{self.db_path}"


# Override specific fields to not use PUNTY_ prefix
class FullSettings(Settings):
    """Full settings with OpenAI key from standard env var."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
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
