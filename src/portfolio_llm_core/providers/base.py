"""Protocol interface for LLM providers."""

from typing import AsyncIterator, Protocol, runtime_checkable

from ..models import ChatRequest, ChatResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement."""

    @property
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'groq', 'openai')."""
        ...

    @property
    def default_model(self) -> str:
        """Default model name for this provider."""
        ...

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request and return the complete response."""
        ...

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Send a chat request and stream response tokens."""
        ...
