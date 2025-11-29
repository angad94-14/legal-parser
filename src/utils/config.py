"""Configuration management using pydantic-settings"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Keys
    openai_api_key: str
    anthropic_api_key: str | None = None  # ← Python 3.12 native syntax

    # Environment
    environment: str = "development"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Model Settings
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Singleton instance
settings = Settings()