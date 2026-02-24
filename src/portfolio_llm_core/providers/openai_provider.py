"""OpenAI LLM provider."""

from typing import AsyncIterator

from ..models import ChatMessage, ChatRequest, ChatResponse, TokenUsage


class OpenAIProvider:
    """OpenAI provider using the official OpenAI SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install portfolio-llm-core[openai]"
            )

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._model

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to OpenAI and return the complete response."""
        response = await self._client.chat.completions.create(
            model=request.model or self._model,
            messages=self._to_openai_messages(request.messages),
            temperature=request.temperature if request.temperature is not None else self._temperature,
            max_tokens=request.max_tokens or self._max_tokens,
            stream=False,
        )

        choice = response.choices[0]
        usage = response.usage

        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider_name,
            finish_reason=choice.finish_reason,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Send a chat request to OpenAI and stream response tokens."""
        stream = await self._client.chat.completions.create(
            model=request.model or self._model,
            messages=self._to_openai_messages(request.messages),
            temperature=request.temperature if request.temperature is not None else self._temperature,
            max_tokens=request.max_tokens or self._max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
