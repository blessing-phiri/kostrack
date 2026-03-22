"""
TokenLedger — OpenAI Provider Wrapper

Drop-in replacement for openai.OpenAI.
Intercepts chat.completions.create() calls, extracts token usage,
calculates cost, and writes a CallRecord asynchronously.

Usage:
    # Before
    from openai import OpenAI
    client = OpenAI()

    # After
    from kostrack import OpenAI
    client = OpenAI(
        tags={"project": "openmanagr", "feature": "gl-classification"}
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

import openai as _openai

from kostrack.calculators.openai_calc import extract_model, extract_tokens
from kostrack.models import CallRecord, TraceContext
from kostrack.writers.batch_writer import AsyncBatchWriter
from kostrack.calculators.pricing_engine import PricingEngine

logger = logging.getLogger("kostrack.providers.openai")


class ChatCompletions:
    """
    Wraps openai.resources.chat.Completions.
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
        Identical signature to openai.chat.completions.create().
        All kwargs are passed through unchanged.
        """
        call_tags = {**self._base_tags}
        extra_tags = kwargs.pop("kostrack_tags", {})
        call_tags.update(extra_tags)

        # Detect streaming — handle separately
        stream = kwargs.get("stream", False)
        if stream:
            return self._create_streaming(kwargs, call_tags)

        start = time.monotonic()
        response = self._completions.create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        tokens = extract_tokens(response)
        model = extract_model(response) or kwargs.get("model", "")
        cost_usd = self._pricing.get_cost(
            provider="openai",
            model=model,
            tokens=tokens,
            pricing_model=self._pricing_model,
        )

        record = CallRecord(
            service_id=self._service_id,
            provider="openai",
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
            "OpenAI call — model=%s prompt=%d completion=%d cached=%d cost=$%.6f latency=%dms",
            model,
            tokens.input,
            tokens.output,
            tokens.cache_read,
            cost_usd,
            latency_ms,
        )

        return response

    def _create_streaming(self, kwargs: dict[str, Any], call_tags: dict[str, str]) -> Any:
        """
        Streaming path — yields chunks, captures usage from the final chunk.
        OpenAI includes usage in the last chunk when stream_options={"include_usage": True}.
        We inject this automatically.
        """
        # Inject usage into stream so we can capture token counts
        stream_options = kwargs.get("stream_options", {})
        stream_options["include_usage"] = True
        kwargs["stream_options"] = stream_options

        start = time.monotonic()
        final_usage_chunk = None

        with self._completions.create(**kwargs) as stream:
            for chunk in stream:
                if chunk.usage is not None:
                    final_usage_chunk = chunk
                yield chunk

        latency_ms = int((time.monotonic() - start) * 1000)

        if final_usage_chunk is None:
            logger.debug("No usage in OpenAI stream — skipping cost record")
            return

        tokens = extract_tokens(final_usage_chunk)
        model = extract_model(final_usage_chunk) or kwargs.get("model", "")
        cost_usd = self._pricing.get_cost(
            provider="openai",
            model=model,
            tokens=tokens,
            pricing_model=self._pricing_model,
        )

        record = CallRecord(
            service_id=self._service_id,
            provider="openai",
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


class Chat:
    def __init__(self, chat: Any, **kwargs: Any) -> None:
        self.completions = ChatCompletions(chat.completions, **kwargs)


class OpenAI:
    """
    TokenLedger drop-in replacement for openai.OpenAI.

    TokenLedger-specific kwargs:
        tags (dict)         — attribution tags applied to every call
        pricing_model (str) — 'per_token' (default) or 'batch'
        trace_ctx           — active TraceContext
    """

    def __init__(
        self,
        *,
        tags: dict[str, str] | None = None,
        pricing_model: str = "per_token",
        trace_ctx: TraceContext | None = None,
        _writer: AsyncBatchWriter | None = None,
        _pricing: PricingEngine | None = None,
        _service_id: str = "default",
        **openai_kwargs: Any,
    ) -> None:
        self._client = _openai.OpenAI(**openai_kwargs)
        self._writer = _writer or _get_global_writer()
        self._pricing = _pricing or _get_global_pricing()
        self._service_id = _service_id
        self._base_tags = tags or {}
        self._pricing_model = pricing_model
        self._trace_ctx = trace_ctx

        _shared = dict(
            writer=self._writer,
            pricing=self._pricing,
            base_tags=self._base_tags,
            service_id=self._service_id,
            trace_ctx=self._trace_ctx,
            pricing_model=self._pricing_model,
        )
        self.chat = Chat(self._client.chat, **_shared)

    def with_trace(self, trace_ctx: TraceContext) -> "OpenAI":
        return OpenAI(
            tags=self._base_tags,
            pricing_model=self._pricing_model,
            trace_ctx=trace_ctx,
            _writer=self._writer,
            _pricing=self._pricing,
            _service_id=self._service_id,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ------------------------------------------------------------------
# Global state — injected by kostrack.configure()
# ------------------------------------------------------------------

_global_writer: AsyncBatchWriter | None = None
_global_pricing: PricingEngine | None = None


def _get_global_writer() -> AsyncBatchWriter:
    if _global_writer is None:
        raise RuntimeError(
            "TokenLedger not configured. Call kostrack.configure(dsn=...) first."
        )
    return _global_writer


def _get_global_pricing() -> PricingEngine:
    if _global_pricing is None:
        raise RuntimeError(
            "TokenLedger not configured. Call kostrack.configure(dsn=...) first."
        )
    return _global_pricing
