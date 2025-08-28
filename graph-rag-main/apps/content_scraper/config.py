"""Configuration management for content scraper."""

import os
from typing import Optional, List
from pydantic import Field, validator

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
from functools import lru_cache


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="content_scraper", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")

    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    class Config:
        env_prefix = "DATABASE_"


class RedisConfig(BaseSettings):
    """Redis configuration."""

    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: str = Field(default="", env="REDIS_PASSWORD_LLM")

    broker_db: int = Field(default=0, env="CELERY_BROKER_CONTENT_SCRAPER_DB")
    backend_db: int = Field(default=1, env="CELERY_BACKEND_CONTENT_SCRAPER_DB")
    cache_db: int = Field(default=2, env="REDIS_CACHE_DB")

    cache_ttl: int = Field(default=900, env="CACHE_TTL")  # 15 minutes

    class Config:
        env_file = ".env"


class ScraperConfig(BaseSettings):
    """Scraper configuration."""

    playwright_timeout_ms: int = Field(default=180000, env="PLAYWRIGHT_TIMEOUT_MS")
    max_retries: int = Field(default=3, env="SCRAPER_MAX_RETRIES")
    retry_delay: int = Field(default=60, env="SCRAPER_RETRY_DELAY")

    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        env="SCRAPER_USER_AGENT",
    )

    ignore_links: bool = Field(default=True, env="SCRAPER_IGNORE_LINKS")
    ignore_images: bool = Field(default=True, env="SCRAPER_IGNORE_IMAGES")

    class Config:
        env_prefix = "SCRAPER_"


class APIConfig(BaseSettings):
    """API configuration."""

    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8099, env="API_PORT")

    title: str = Field(default="Content Scraper API", env="API_TITLE")
    description: str = Field(
        default="API for scraping and parsing web content", env="API_DESCRIPTION"
    )
    version: str = Field(default="1.0.0", env="API_VERSION")

    # CORS settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")

    class Config:
        env_prefix = "API_"


class SerperConfig(BaseSettings):
    """Serper API configuration."""

    api_key: str = Field(default="", env="SERPER_API_KEY")
    default_num_results: int = Field(default=10, env="SERPER_DEFAULT_NUM_RESULTS")
    default_country: str = Field(default="us", env="SERPER_DEFAULT_COUNTRY")
    default_language: str = Field(default="en", env="SERPER_DEFAULT_LANGUAGE")

    timeout: int = Field(default=30, env="SERPER_TIMEOUT")

    @field_validator("api_key")
    def validate_api_key(cls, v):
        """Validate API key is provided."""
        if not v:
            raise ValueError("SERPER_API_KEY must be set")
        return v

    class Config:
        env_prefix = "SERPER_"


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT"
    )

    # File logging
    file_enabled: bool = Field(default=False, env="LOG_FILE_ENABLED")
    file_path: str = Field(default="logs/content_scraper.log", env="LOG_FILE_PATH")
    file_rotation: str = Field(default="midnight", env="LOG_FILE_ROTATION")
    file_retention: int = Field(default=7, env="LOG_FILE_RETENTION")

    # JSON logging
    json_enabled: bool = Field(default=False, env="LOG_JSON_ENABLED")

    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main settings aggregator."""

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    scraper: ScraperConfig = ScraperConfig()
    api: APIConfig = APIConfig()
    serper: SerperConfig = SerperConfig()
    logging: LoggingConfig = LoggingConfig()

    # Feature flags
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_verification: bool = Field(default=False, env="ENABLE_VERIFICATION")
    enable_metrics: bool = Field(default=False, env="ENABLE_METRICS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()


# Convenience functions
def get_database_url() -> str:
    """Get database URL from settings."""
    return get_settings().database.url


def get_redis_url(db: int = 0) -> str:
    """Get Redis URL from settings."""
    settings = get_settings().redis
    return f"redis://:{settings.password}@{settings.host}:{settings.port}/{db}"


def get_celery_broker_url() -> str:
    """Get Celery broker URL."""
    settings = get_settings().redis
    return get_redis_url(settings.broker_db)


def get_celery_backend_url() -> str:
    """Get Celery backend URL."""
    settings = get_settings().redis
    return get_redis_url(settings.backend_db)
