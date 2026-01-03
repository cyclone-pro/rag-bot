"""
Application Settings
Loads configuration from environment variables
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application configuration"""
    
    # ===== APPLICATION =====
    app_name: str = "RecruiterBrain Voice Interview"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # ===== DATABASE - POSTGRESQL =====
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "recruiterbrain"
    postgres_user: str
    postgres_password: str
    
    # Connection Pooling
    postgres_pool_size: int = 50
    postgres_max_overflow: int = 100
    postgres_pool_timeout: int = 30
    postgres_pool_recycle: int = 3600
    
    @property
    def postgres_url(self) -> str:
        """Build PostgreSQL async URL"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    # ===== DATABASE - MILVUS =====
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "interview_transcripts_v2"
    milvus_pool_size: int = 10
    
    # ===== EMBEDDINGS =====
    embedding_model: str = "intfloat/e5-base-v2"
    embedding_dim: int = 768
    embedding_device: str = "cpu"
    
    # ===== LIVEKIT =====
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    #livekit_sip_domain: str = "wysdsmxq.sip.livekit.cloud"
    livekit_sip_domain:str ="54jym8vfe6a.sip.livekit.cloud"
    livekit_outbound_trunk_id: str = Field(..., env="LIVEKIT_OUTBOUND_TRUNK_ID")
    livekit_agent_name: str 

    # ===== DEEPGRAM =====
    deepgram_api_key: str
    deepgram_model: str = "nova-2"
    deepgram_language: str = "en-US"
    
    # ===== GOOGLE CLOUD TTS =====
    google_application_credentials: str
    google_tts_voice_name: str = "en-US-Neural2-J"
    google_tts_language_code: str = "en-US"
    google_tts_speaking_rate: float = 1.0
    google_tts_pitch: float = 0.0
    
    # ===== OPENAI =====
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000
    
    # ===== TELNYX =====
    telnyx_api_key: str
    telnyx_public_key: Optional[str] = None
    telnyx_phone_number: str
    telnyx_connection_id: Optional[str] = None
    
    #=====Twillio=====

    # Twilio Configuration (for SIP calling via LiveKit)
    twilio_account_sid: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    twilio_phone_number: Optional[str] = Field(default=None, env="TWILIO_PHONE_NUMBER")
    # ===== API =====
    api_bind_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = True
    api_public_base_url: str = "http://localhost:8000"
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True
    
    # ===== SECURITY =====
    secret_key: str
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # ===== INTERVIEW SETTINGS =====
    interview_max_duration_seconds: int = 720  # 12 minutes
    interview_questions_count: int = 6
    interview_timeout_seconds: int = 30
    
    # ===== RATE LIMITING =====
    rate_limit_enabled: bool = True
    rate_limit_calls: int = 100
    rate_limit_period: int = 3600
    
    # ===== STORAGE (Optional) =====
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # ===== MONITORING =====
    sentry_dsn: Optional[str] = None
    prometheus_port: int = 9090
    
    # ===== FEATURE FLAGS =====
    enable_recording: bool = True
    enable_consent_check: bool = True
    enable_skills_extraction: bool = True
    enable_sentiment_analysis: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
