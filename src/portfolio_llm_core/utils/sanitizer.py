"""Output sanitization utilities for LLM responses.

Generalized from recipe-extraction-service/src/services/groq_service.py _sanitize_recipe_dict().
Handles common LLM output issues: null values, type coercion, missing required fields.
"""

from typing import Any


class OutputSanitizer:
    """Sanitize LLM output dicts before Pydantic validation."""

    @staticmethod
    def sanitize(
        data: dict,
        required_strings: list[str] | None = None,
        required_lists: list[str] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> dict:
        """
        Sanitize a dict from LLM output.

        Args:
            data: Raw dict from LLM response
            required_strings: Fields that must be non-empty strings (set default if null/empty)
            required_lists: Fields that must be lists (set [] if missing or wrong type)
            defaults: Default values for fields that are null/missing

        Returns:
            Sanitized dict ready for Pydantic validation
        """
        result = dict(data)

        # Apply defaults for null/missing fields
        if defaults:
            for field, default_value in defaults.items():
                if result.get(field) is None:
                    result[field] = default_value

        # Ensure required string fields are non-empty
        if required_strings:
            for field in required_strings:
                if not result.get(field):
                    result[field] = defaults.get(field, "") if defaults else ""

        # Ensure required list fields are lists
        if required_lists:
            for field in required_lists:
                if not isinstance(result.get(field), list):
                    result[field] = []

        return result

    @staticmethod
    def coerce_float(value: Any, default: float = 0.0) -> float:
        """Coerce a value to float, returning default on failure."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def coerce_int(value: Any, default: int = 0) -> int:
        """Coerce a value to int, returning default on failure."""
        if value is None:
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def clamp_float(value: Any, min_val: float, max_val: float, default: float = 0.0) -> float:
        """Coerce to float and clamp to [min_val, max_val]."""
        f = OutputSanitizer.coerce_float(value, default)
        return max(min_val, min(max_val, f))
