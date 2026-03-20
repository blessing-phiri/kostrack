"""
TokenLedger — OpenAI Cost Calculator

Extracts token counts from OpenAI API responses,
handling OpenAI-specific pricing nuances:
  - Cached prompt tokens (50% discount via cache_read_rate)
  - Reasoning tokens (o1, o3 models — billed as output)
  - Audio tokens (future-proofing)
"""

from __future__ import annotations

import logging
from typing import Any

from kostrack.models import TokenBreakdown

logger = logging.getLogger("kostrack.calculators.openai")


def extract_tokens(response: Any) -> TokenBreakdown:
    """
    Extract token counts from an OpenAI Chat Completions response.

    Handles both the openai-sdk response object and raw dicts.
    """
    usage = _get_usage(response)
    if usage is None:
        logger.warning("No usage field in OpenAI response — tokens defaulting to 0")
        return TokenBreakdown()

    input_tokens = _int(usage, "prompt_tokens")
    output_tokens = _int(usage, "completion_tokens")

    # OpenAI nests cache and reasoning details under prompt_tokens_details
    # and completion_tokens_details respectively
    prompt_details = _get_attr(usage, "prompt_tokens_details") or {}
    completion_details = _get_attr(usage, "completion_tokens_details") or {}

    cached_tokens = _int(prompt_details, "cached_tokens")

    # Reasoning tokens are a subset of completion_tokens (already included)
    # We track them separately for observability
    reasoning_tokens = _int(completion_details, "reasoning_tokens")

    return TokenBreakdown(
        input=input_tokens,
        output=output_tokens,
        cache_read=cached_tokens,     # OpenAI has no cache write — only cache hits
        cache_write=0,
        thinking=reasoning_tokens,    # reuse thinking field for reasoning tokens
        extra={"reasoning_tokens": reasoning_tokens} if reasoning_tokens else {},
    )


def extract_model(response: Any) -> str:
    model = _get_attr(response, "model")
    return str(model) if model else ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_usage(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _get_attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int(obj: Any, key: str) -> int:
    val = _get_attr(obj, key)
    return int(val) if val is not None else 0
