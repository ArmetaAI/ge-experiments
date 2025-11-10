"""
Application configuration using Pydantic Settings.

Centralized environment variable management with automatic validation.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings from environment variables.
    
    Local dev: Load from .env file
    Production: Environment variables from Cloud Run
    """
    
    # Database Configuration
    DB_USER: str
    DB_PASS: str = ""  # Can be DB_PASS or DB_PASSWORD_GOSEXPERT
    DB_PASSWORD_GOSEXPERT: str = ""  # Alternative name for password
    DB_NAME: str
    INSTANCE_CONNECTION_NAME: str  # Format: project:region:instance (production) OR localhost:port (local)
    DATABASE_URL: str = ""  # Optional: For local development via Cloud SQL Proxy
    
    @property
    def database_password(self) -> str:
        """Get database password from either DB_PASS or DB_PASSWORD_GOSEXPERT."""
        return self.DB_PASS or self.DB_PASSWORD_GOSEXPERT
    
    # Google Cloud Storage
    GCS_BUCKET_NAME: str
    GCS_TEST_BUCKET_NAME: str
    GCS_PROJECT_ID: str = ""  # Optional: Defaults to auto-detection if empty
    
    LMNR_PROJECT_API_KEY: str  # Laminar API Key
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures settings are loaded only once
    and reused throughout the application lifecycle.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Global settings instance for convenience
settings = get_settings()
