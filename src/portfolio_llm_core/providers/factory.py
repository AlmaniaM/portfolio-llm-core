"""Factory function for creating LLM providers."""

from .base import LLMProvider


def create_provider(provider: str, api_key: str, **kwargs) -> LLMProvider:
    """
    Create an LLM provider by name.

    Args:
        provider: Provider name ('groq', 'openai')
        api_key: API key for the provider
        **kwargs: Provider-specific configuration (model, temperature, max_tokens, timeout)

    Returns:
        An LLMProvider instance

    Raises:
        ValueError: If the provider name is unknown
        ImportError: If the required SDK is not installed
    """
    if provider == "groq":
        from .groq_provider import GroqProvider
        return GroqProvider(api_key=api_key, **kwargs)
    elif provider == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. Supported providers: groq, openai"
        )
