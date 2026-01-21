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

    # LangSmith Tracing (https://smith.langchain.com/)
    langchain_tracing_v2: bool = False  # Enable LangSmith tracing
    langchain_api_key: str = ""  # LangSmith API key
    langchain_project: str = "DiabeticAIChat"  # Project name in LangSmith
    langchain_endpoint: str = "https://api.smith.langchain.com"  # LangSmith API endpoint

    # CareLink Sync Configuration (Token-based API)
    # Get token by logging into CareLink in browser, then extract 'auth_tmp_token' cookie
    carelink_token: str = ""  # Initial auth token from browser cookie
    carelink_country_code: str = "us"  # 'us' or country code for EU
    carelink_patient_username: str = ""  # Optional: specific patient username (for care partners)
    sync_schedule_enabled: bool = False  # Enable weekly auto-sync
    sync_schedule_day: int = 0  # 0=Monday, 6=Sunday
    sync_schedule_hour: int = 6  # Hour in 24h format (0-23)
    
    # Legacy Selenium-based sync (deprecated)
    carelink_username: str = ""
    carelink_password: str = ""
    carelink_headless: bool = True

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

    @property
    def is_carelink_configured(self) -> bool:
        """Check if CareLink is configured (token-based or legacy)."""
        # Prefer token-based auth
        if self.carelink_token:
            return True
        # Fall back to legacy username/password
        return bool(self.carelink_username and self.carelink_password)
    
    @property
    def carelink_use_token_auth(self) -> bool:
        """Check if token-based auth should be used."""
        return bool(self.carelink_token)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

