# portfolio-llm-core

Shared Python LLM library for portfolio services. Provides a unified provider interface, JSON extraction, output sanitization, and cost calculation — without requiring LangChain as a dependency.

## Installation

```bash
# Groq only
pip install -e ".[groq]"

# OpenAI only
pip install -e ".[openai]"

# Both
pip install -e ".[groq,openai]"
```

## Usage

```python
from portfolio_llm_core import create_provider, ChatRequest, ChatMessage

# Create provider
provider = create_provider("groq", api_key="gsk_...", model="llama-3.3-70b-versatile")

# Chat
request = ChatRequest(messages=[
    ChatMessage(role="user", content="Hello!")
])
response = await provider.chat(request)
print(response.content)

# Streaming
async for token in provider.chat_stream(request):
    print(token, end="", flush=True)
```

## JSON Extraction

```python
from portfolio_llm_core import JSONExtractor

# Handles raw JSON, markdown code blocks, and common LLM formatting issues
data = JSONExtractor.extract(llm_response_text)

# Validate against Pydantic model
from myapp.models import MySchema
validated = JSONExtractor.extract_validated(llm_response_text, MySchema)
```

## Cost Estimation

```python
from portfolio_llm_core import CostCalculator

cost = CostCalculator.estimate_cost("groq", "llama-3.3-70b-versatile", response.usage)
print(CostCalculator.format_cost(cost))  # "$0.0001"
```

## Services

| Service | Uses |
|---------|------|
| portfolio-rag-backend | OpenAI (primary), Groq (fast inference) |
| recipe-extraction-service-python | Groq (text and vision extraction) |
