"""
kostrack — DeepSeek Provider Wrapper

Drop-in replacement for openai.OpenAI pointed at DeepSeek's API.
DeepSeek's API is fully OpenAI-compatible, so we wrap openai.OpenAI
with a custom base_url. Intercepts chat.completions.create() calls,
extracts token usage (including reasoning tokens for deepseek-reasoner),
calculates cost at DeepSeek pricing, and writes a CallRecord
asynchronously.

Usage:
    # Before
    from openai import OpenAI
    client = OpenAI(api_key="sk-...", base_url="https://api.deepseek.com")

    # After — one import change
    from kostrack import DeepSeek
    client = DeepSeek(
        tags={
            "project": "openmanagr",
            "feature": "journal-classification",
        }
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Classify this transaction..."}],
    )

Reasoning model (R1):
    reasoner = DeepSeek(
        model_hint="deepseek-reasoner",
        tags={"project": "openmanagr", "feature": "ifrss-interpretation"},
    )
    response = reasoner.chat.completions.create(
        model="deepseek-reasoner",
        messages=[...],
    )
    # Reasoning tokens are captured in token_breakdown.thinking
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import openai as _openai

from kostrack.calculators.deepseek_calc import extract_model, extract_tokens
from kostrack.models import CallRecord, TraceContext
from kostrack.writers.batch_writer import AsyncBatchWriter
from kostrack.calculators.pricing_engine import PricingEngine

logger = logging.getLogger("kostrack.providers.deepseek")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class ChatCompletions:
    """
    Wraps openai.resources.chat.Completions pointed at DeepSeek.
    Intercepts .create() to capture cost data.
    """

    def __init__(
        self,
        completions: Any,
        writer: AsyncBatchWriter,
        pricing: PricingEngine,
        base_tags: dict[str, str],
        service_id: str,
        trace_ctx: TraceContext | None,
        pricing_model: str,
    ) -> None:
        self._completions = completions
        self._writer = writer
        self._pricing = pricing
        self._base_tags = base_tags
        self._service_id = service_id
        self._trace_ctx = trace_ctx
        self._pricing_model = pricing_model

    def create(self, **kwargs: Any) -> Any:
        """
        Identical signature to openai chat.completions.create().
        All kwargs are passed through to DeepSeek unchanged.
        """
        call_tags = {**self._base_tags}
        extra_tags = kwargs.pop("kostrack_tags", {})
        call_tags.update(extra_tags)

        start = time.monotonic()
        response = self._completions.create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        tokens = extract_tokens(response)
        model = extract_model(response) or kwargs.get("model", "deepseek-chat")
        cost_usd = self._pricing.get_cost(
            provider="deepseek",
            model=model,
            tokens=tokens,
            pricing_model=self._pricing_model,
        )

        record = CallRecord(
            service_id=self._service_id,
            provider="deepseek",
            model=model,
            pricing_model=self._pricing_model,
            tokens=tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            tags=call_tags,
        )

        from kostrack.tracing import get_active_trace
        active_trace = self._trace_ctx or get_active_trace()
        if active_trace:
            record.trace_id = active_trace.trace_id
            record.parent_span_id = active_trace.span_id
            active_trace.record_call(cost_usd, model=model)

        self._writer.write(record.to_row())

        logger.debug(
            "DeepSeek call — model=%s input=%d output=%d reasoning=%d "
            "cache_hit=%d cost=$%.6f latency=%dms",
            model,
            tokens.input,
            tokens.output,
            tokens.thinking,
            tokens.cache_read,
            cost_usd,
            latency_ms,
        )

        return response


class Chat:
    def __init__(self, completions: ChatCompletions) -> None:
        self.completions = completions


class DeepSeek:
    """
    kostrack drop-in replacement for openai.OpenAI pointed at DeepSeek.

    All openai.OpenAI kwargs are passed through (api_key, timeout, etc.).
    kostrack-specific kwargs:
        tags (dict)         — attribution tags applied to every call
        pricing_model (str) — 'per_token' (default) or 'batch'
        trace_ctx           — active TraceContext (set by kostrack.trace())
        api_key (str)       — defaults to DEEPSEEK_API_KEY env var
    """

    def __init__(
        self,
        *,
        tags: dict[str, str] | None = None,
        pricing_model: str = "per_token",
        trace_ctx: TraceContext | None = None,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        _writer: AsyncBatchWriter | None = None,
        _pricing: PricingEngine | None = None,
        _service_id: str = "default",
        **openai_kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key required. Pass api_key= or set DEEPSEEK_API_KEY."
            )

        self._client = _openai.OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            **openai_kwargs,
        )
        self._writer = _writer or _get_global_writer()
        self._pricing = _pricing or _get_global_pricing()
        self._service_id = _service_id
        self._base_tags = tags or {}
        self._pricing_model = pricing_model
        self._trace_ctx = trace_ctx

        self.chat = Chat(
            ChatCompletions(
                completions=self._client.chat.completions,
                writer=self._writer,
                pricing=self._pricing,
                base_tags=self._base_tags,
                service_id=self._service_id,
                trace_ctx=self._trace_ctx,
                pricing_model=self._pricing_model,
            )
        )

    def with_trace(self, trace_ctx: TraceContext) -> "DeepSeek":
        """Return a new client bound to the given trace context."""
        return DeepSeek(
            tags=self._base_tags,
            pricing_model=self._pricing_model,
            trace_ctx=trace_ctx,
            api_key=self._client.api_key,
            _writer=self._writer,
            _pricing=self._pricing,
            _service_id=self._service_id,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ------------------------------------------------------------------
# Global state — set by kostrack.configure()
# ------------------------------------------------------------------

_global_writer: AsyncBatchWriter | None = None
_global_pricing: PricingEngine | None = None


def _get_global_writer() -> AsyncBatchWriter:
    if _global_writer is None:
        raise RuntimeError(
            "kostrack not configured. Call kostrack.configure(dsn=...) first."
        )
    return _global_writer


def _get_global_pricing() -> PricingEngine:
    if _global_pricing is None:
        raise RuntimeError(
            "kostrack not configured. Call kostrack.configure(dsn=...) first."
        )
    return _global_pricing
