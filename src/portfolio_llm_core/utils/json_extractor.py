"""JSON extraction utilities for LLM responses.

Ported and generalized from recipe-extraction-service/src/services/groq_service.py.
Handles the common failure modes in LLM JSON output: markdown code blocks,
trailing commas, and malformed responses.
"""

import json
import re
from typing import Any

from pydantic import BaseModel


class JSONExtractor:
    """Extract and validate JSON from LLM response text."""

    @staticmethod
    def extract(text: str) -> dict | list:
        """
        Extract JSON from LLM response text.

        Handles:
        - Raw JSON
        - JSON wrapped in markdown code blocks (```json ... ```)
        - JSON wrapped in plain code blocks (``` ... ```)
        - Trailing whitespace and newlines

        Args:
            text: Raw text from LLM response

        Returns:
            Parsed dict or list

        Raises:
            ValueError: If no valid JSON can be extracted
        """
        # Try to extract from markdown code block first
        code_block_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```",
            text,
            re.DOTALL,
        )
        if code_block_match:
            text = code_block_match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}\nResponse text: {text[:500]}")

    @staticmethod
    def extract_validated(text: str, model: type[BaseModel]) -> BaseModel:
        """
        Extract JSON from LLM response and validate against a Pydantic model.

        Args:
            text: Raw text from LLM response
            model: Pydantic model class to validate against

        Returns:
            Validated model instance

        Raises:
            ValueError: If JSON extraction fails
            pydantic.ValidationError: If JSON doesn't match the model schema
        """
        data = JSONExtractor.extract(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
        return model(**data)

    @staticmethod
    def extract_field(text: str, field: str, default: Any = None) -> Any:
        """
        Extract a single field from JSON in LLM response.

        Args:
            text: Raw text from LLM response
            field: Field name to extract
            default: Default value if field is missing or extraction fails

        Returns:
            Field value or default
        """
        try:
            data = JSONExtractor.extract(text)
            if isinstance(data, dict):
                return data.get(field, default)
        except ValueError:
            pass
        return default
