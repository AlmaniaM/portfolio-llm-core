"""Unified request models for LLM providers."""

from typing import Any
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    """Unified chat request for any LLM provider."""

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    extra: dict[str, Any] = {}
