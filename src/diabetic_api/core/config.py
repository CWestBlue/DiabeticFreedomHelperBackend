"""Application configuration using Pydantic Settings."""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "diabetic_db"

    # LLM Provider Selection
    llm_provider: LLMProvider = LLMProvider.OPENAI

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Google Gemini Configuration
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"  # Options: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro

    # LLM Settings
    llm_temperature: float = 0.0

    # Usage Limits (set to 0 to disable)
    daily_llm_call_limit: int = 100  # Max LLM calls per day (0 = unlimited)
    monthly_llm_call_limit: int = 3000  # Max LLM calls per month (0 = unlimited)
    daily_token_limit: int = 0  # Max tokens per day (0 = unlimited) - input + output
    monthly_token_limit: int = 10000000  # Max tokens per month (0 = unlimited)
    usage_tracking_enabled: bool = True  # Track usage even if limits are disabled

    # App
    debug: bool = False
    app_name: str = "Diabetic AI API"
    api_version: str = "1.0.0"

    @property
    def is_llm_configured(self) -> bool:
        """Check if the selected LLM provider is configured."""
        if self.llm_provider == LLMProvider.OPENAI:
            return bool(self.openai_api_key)
        elif self.llm_provider == LLMProvider.GEMINI:
            return bool(self.google_api_key)
        return False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

