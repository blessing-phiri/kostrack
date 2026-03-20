"""
TokenLedger — Gemini Cost Calculator

Extracts token counts from Gemini API responses,
handling Gemini-specific pricing nuances:
  - Cached content tokens (context caching)
  - Separate prompt / candidates token counts
  - Thinking tokens (Gemini 2.0 Flash Thinking)
"""

from __future__ import annotations

import logging
from typing import Any

from kostrack.models import TokenBreakdown

logger = logging.getLogger("kostrack.calculators.gemini")


def extract_tokens(response: Any) -> TokenBreakdown:
    """
    Extract token counts from a Gemini GenerateContent response.

    Handles both the google-generativeai SDK response object and raw dicts.
    Gemini's usage metadata sits at response.usage_metadata.
    """
    usage = _get_usage(response)
    if usage is None:
        logger.warning("No usage_metadata in Gemini response — tokens defaulting to 0")
        return TokenBreakdown()

    # Gemini field names differ from Anthropic/OpenAI
    input_tokens = _int(usage, "prompt_token_count")
    output_tokens = _int(usage, "candidates_token_count")

    # Context caching — cached_content_token_count is a subset of prompt_token_count
    cached_tokens = _int(usage, "cached_content_token_count")

    # Thinking tokens — available on Gemini 2.0 Flash Thinking
    thinking_tokens = _int(usage, "thoughts_token_count")

    return TokenBreakdown(
        input=input_tokens,
        output=output_tokens,
        cache_read=cached_tokens,
        cache_write=0,
        thinking=thinking_tokens,
        extra={"thoughts_token_count": thinking_tokens} if thinking_tokens else {},
    )


def extract_model(response: Any) -> str:
    """
    Gemini doesn't always echo the model in the response.
    Return empty string — caller falls back to the model kwarg.
    """
    model = _get_attr(response, "model")
    return str(model) if model else ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_usage(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("usage_metadata")
    return getattr(response, "usage_metadata", None)


def _get_attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int(obj: Any, key: str) -> int:
    val = _get_attr(obj, key)
    return int(val) if val is not None else 0
