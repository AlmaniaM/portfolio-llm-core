"""Shared LLM configuration settings."""

from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """Shared LLM settings for portfolio services.

    Load from environment variables. Each service can subclass this
    and add service-specific settings.
    """

    # Active provider
    llm_provider: str = "groq"

    # Groq settings
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.3
    groq_max_tokens: int = 2048
    groq_timeout: int = 30

    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2048
    openai_timeout: int = 30

    # Anthropic settings
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_temperature: float = 0.7
    anthropic_max_tokens: int = 2048
    anthropic_timeout: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
