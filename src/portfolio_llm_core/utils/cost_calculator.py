"""Cost estimation utilities for LLM usage.

Lookup table for input/output token costs per provider and model.
Costs are in USD per 1M tokens.
"""

from ..models import TokenUsage

# Cost per 1M tokens (input, output) in USD
# Updated: 2026-02
_COSTS: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "o1": (15.00, 60.00),
        "o1-mini": (3.00, 12.00),
    },
    "groq": {
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-70b-versatile": (0.59, 0.79),
        "llama-3.2-3b-preview": (0.06, 0.06),
        "llama-3.2-1b-preview": (0.04, 0.04),
        "mixtral-8x7b-32768": (0.24, 0.24),
        "gemma2-9b-it": (0.20, 0.20),
    },
    "anthropic": {
        "claude-opus-4-6": (15.00, 75.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5-20251001": (0.25, 1.25),
    },
}


class CostCalculator:
    """Estimate LLM API costs from token usage."""

    @staticmethod
    def estimate_cost(provider: str, model: str, usage: TokenUsage) -> float:
        """
        Estimate the cost in USD for a given token usage.

        Args:
            provider: Provider name ('openai', 'groq', 'anthropic')
            model: Model name
            usage: Token usage from the response

        Returns:
            Estimated cost in USD (0.0 if provider/model not in table)
        """
        provider_costs = _COSTS.get(provider.lower(), {})
        model_costs = provider_costs.get(model)

        if model_costs is None:
            # Try prefix match (e.g., 'gpt-4o-mini-2024-07-18' matches 'gpt-4o-mini')
            for known_model, costs in provider_costs.items():
                if model.startswith(known_model):
                    model_costs = costs
                    break

        if model_costs is None:
            return 0.0

        input_cost_per_m, output_cost_per_m = model_costs
        input_cost = (usage.prompt_tokens / 1_000_000) * input_cost_per_m
        output_cost = (usage.completion_tokens / 1_000_000) * output_cost_per_m
        return input_cost + output_cost

    @staticmethod
    def format_cost(cost_usd: float) -> str:
        """Format a cost in USD for display."""
        if cost_usd < 0.001:
            return f"${cost_usd * 1000:.4f}m"  # millicents
        return f"${cost_usd:.4f}"
