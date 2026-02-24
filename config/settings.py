"""
Application configuration settings.
Uses pydantic-settings for environment variable management.
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Predictive Medication Adherence Engine"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/medication_adherence",
        description="PostgreSQL connection string",
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Claude API (Anthropic)
    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-3-5-sonnet-20241022"

    # Twilio (SMS/Voice)
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None

    # ML Model paths
    model_path: str = "models/risk_prediction_model.joblib"
    feature_config_path: str = "config/features.yaml"

    # Risk thresholds
    risk_threshold_low: float = 30.0
    risk_threshold_high: float = 70.0

    # Intervention settings
    max_interventions_per_day: int = 3
    intervention_cooldown_hours: int = 24

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090


class FeatureFlags(BaseSettings):
    """Feature flags for gradual rollout."""

    model_config = SettingsConfigDict(env_prefix="FF_")

    enable_sms_interventions: bool = True
    enable_voice_interventions: bool = False
    enable_chatbot: bool = True
    enable_email_campaigns: bool = True
    enable_care_manager_alerts: bool = True
    enable_ab_testing: bool = True
    enable_reinforcement_learning: bool = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache
def get_feature_flags() -> FeatureFlags:
    """Get cached feature flags instance."""
    return FeatureFlags()
