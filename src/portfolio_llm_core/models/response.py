"""Unified response models for LLM providers."""

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Token usage for a chat completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Unified chat response from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: TokenUsage = TokenUsage()
    finish_reason: str | None = None
    raw: dict = {}
