"""
kostrack — DeepSeek Cost Calculator

Extracts token counts from DeepSeek API responses.
DeepSeek uses the OpenAI-compatible Chat Completions format,
so the response shape is identical to OpenAI — but pricing and
token field names differ in one key way:

  - DeepSeek R1 (reasoning model) surfaces thinking tokens under
    usage.completion_tokens_details.reasoning_tokens, exactly
    like OpenAI o1/o3. We reuse the thinking field for these.
  - Cache: DeepSeek exposes prompt_cache_hit_tokens (discounted)
    and prompt_cache_miss_tokens (full price) directly on usage.
    There is no prompt_tokens_details nesting.

Supported models (March 2026):
    deepseek-chat      — DeepSeek-V3
    deepseek-reasoner  — DeepSeek-R1 (with reasoning tokens)
"""

from __future__ import annotations

import logging
from typing import Any

from kostrack.models import TokenBreakdown

logger = logging.getLogger("kostrack.calculators.deepseek")


def extract_tokens(response: Any) -> TokenBreakdown:
    """
    Extract token counts from a DeepSeek Chat Completions response.

    Handles both the openai-sdk response object (since DeepSeek is
    OpenAI-API-compatible) and raw dicts.
    """
    usage = _get_usage(response)
    if usage is None:
        logger.warning("No usage field in DeepSeek response — tokens defaulting to 0")
        return TokenBreakdown()

    input_tokens = _int(usage, "prompt_tokens")
    output_tokens = _int(usage, "completion_tokens")

    # DeepSeek cache fields sit directly on usage (not nested under details)
    cache_hit = _int(usage, "prompt_cache_hit_tokens")
    cache_miss = _int(usage, "prompt_cache_miss_tokens")  # full-price tokens

    # Reasoning tokens (DeepSeek-R1 only) — subset of completion_tokens,
    # nested identically to OpenAI o1/o3
    completion_details = _get_attr(usage, "completion_tokens_details") or {}
    reasoning_tokens = _int(completion_details, "reasoning_tokens")

    return TokenBreakdown(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_hit,
        cache_write=cache_miss,   # "miss" = not cached, billed at full rate
        thinking=reasoning_tokens,
        extra={
            **({"prompt_cache_miss_tokens": cache_miss} if cache_miss else {}),
            **({"reasoning_tokens": reasoning_tokens} if reasoning_tokens else {}),
        },
    )


def extract_model(response: Any) -> str:
    """Extract model string from response."""
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
