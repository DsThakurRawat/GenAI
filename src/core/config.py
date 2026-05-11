from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings and environment variables.
    Pydantic automatically loads these from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # App Config
    APP_NAME: str = "GenAI-Production"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # LangSmith (Highly recommended for production LangChain)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "default"

# Global settings instance
settings = Settings()
