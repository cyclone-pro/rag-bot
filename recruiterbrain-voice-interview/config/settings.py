"""
Application settings and configuration.
Loads environment variables and provides typed access.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_DEBUG: bool = True
    BASE_URL: str = "http://localhost:8000"
    
    # Telnyx
    TELNYX_API_KEY: str
    TELNYX_PHONE_NUMBER: str
    TELNYX_CONNECTION_ID: str
    TELNYX_WEBHOOK_SECRET: Optional[str] = None
    
    # Deepgram
    DEEPGRAM_API_KEY: str
    
    # Google Cloud
    GOOGLE_APPLICATION_CREDENTIALS: str
    GOOGLE_CLOUD_PROJECT: str
    GCS_BUCKET_NAME: str
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.7
    
    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "recruiterbrain"
    POSTGRES_USER: str = "recruiterbrain_user"
    POSTGRES_PASSWORD: str
    
    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Webhooks
    WEBHOOK_BASE_URL: str
    
    # Interview Configuration
    MAX_INTERVIEW_DURATION_MINUTES: int = 12
    DEFAULT_QUESTIONS_COUNT: int = 6
    SILENCE_THRESHOLD_THINKING: int = 3
    SILENCE_THRESHOLD_PROMPT: int = 7
    SILENCE_THRESHOLD_SKIP: int = 15
    
    # Embedding
    EMBEDDING_MODEL: str = "intfloat/e5-base-v2"
    EMBEDDING_DIMENSION: int = 768
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/interview.log"
    
    # Optional monitoring
    SENTRY_DSN: Optional[str] = None
    
    @property
    def DATABASE_URL(self) -> str:
        """PostgreSQL connection URL."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """Async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def REDIS_URL(self) -> str:
        """Redis connection URL."""
        password_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{password_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
