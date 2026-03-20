"""
kostrack — Anthropic Cost Calculator

Extracts token counts from Anthropic API responses,
handling all Anthropic-specific pricing nuances:
  - Cache creation (write) vs cache usage (read) tokens
  - Extended thinking tokens (billed as output)
  - Batch API pricing model
"""

from __future__ import annotations

import logging
from typing import Any

from kostrack.models import TokenBreakdown

logger = logging.getLogger("kostrack.calculators.anthropic")


def extract_tokens(response: Any) -> TokenBreakdown:
    """
    Extract token counts from an Anthropic Messages API response.

    Handles both the anthropic-sdk response object and raw dicts,
    so it works regardless of how the response was received.
    """
    usage = _get_usage(response)
    if usage is None:
        logger.warning("No usage field in Anthropic response — tokens defaulting to 0")
        return TokenBreakdown()

    # Standard tokens
    input_tokens = _int(usage, "input_tokens")
    output_tokens = _int(usage, "output_tokens")

    # Cache tokens — Anthropic distinguishes write vs read
    cache_creation = _int(usage, "cache_creation_input_tokens")   # write
    cache_read = _int(usage, "cache_read_input_tokens")           # read

    # Thinking tokens are billed as output tokens in Anthropic pricing.
    # We track them separately for observability but don't add to output_tokens
    # (they're already included in the output_tokens count from the API).
    thinking_tokens = 0
    content = _get_content(response)
    if content:
        for block in content:
            block_type = _get_attr(block, "type")
            if block_type == "thinking":
                thinking_text = _get_attr(block, "thinking") or ""
                # Rough estimate: 1 token ≈ 4 chars — exact count not in response
                thinking_tokens = len(thinking_text) // 4

    return TokenBreakdown(
        input=input_tokens,
        output=output_tokens,
        cache_write=cache_creation,
        cache_read=cache_read,
        thinking=thinking_tokens,
        extra={},
    )


def extract_model(response: Any) -> str:
    """Extract model string from response."""
    model = _get_attr(response, "model")
    return str(model) if model else ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_usage(response: Any) -> Any:
    """Get usage from response object or dict."""
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _get_content(response: Any) -> list | None:
    if isinstance(response, dict):
        return response.get("content")
    return getattr(response, "content", None)


def _get_attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int(obj: Any, key: str) -> int:
    val = _get_attr(obj, key)
    return int(val) if val is not None else 0