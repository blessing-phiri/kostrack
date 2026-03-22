"""
kostrack — Anthropic Provider Wrapper

Drop-in replacement for anthropic.Anthropic.
Intercepts messages.create() calls, extracts token usage,
calculates cost, and writes a CallRecord asynchronously.

Usage:
    # Before
    from anthropic import Anthropic
    client = Anthropic()

    # After — one import change, everything else identical
    from kostrack import Anthropic
    client = Anthropic(
        tags={"project": "openmanagr", "feature": "invoice-extraction"}
    )
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import anthropic as _anthropic

from kostrack.calculators.anthropic_calc import extract_model, extract_tokens
from kostrack.models import CallRecord, TraceContext
from kostrack.writers.batch_writer import AsyncBatchWriter
from kostrack.calculators.pricing_engine import PricingEngine

logger = logging.getLogger("kostrack.providers.anthropic")


class Messages:
    """
    Wraps anthropic.resources.Messages.
    Intercepts .create() to capture cost data.
    """

    def __init__(
        self,
        messages: _anthropic.resources.Messages,
        writer: AsyncBatchWriter,
        pricing: PricingEngine,
        base_tags: dict[str, str],
        service_id: str,
        trace_ctx: TraceContext | None,
        pricing_model: str,
    ) -> None:
        self._messages = messages
        self._writer = writer
        self._pricing = pricing
        self._base_tags = base_tags
        self._service_id = service_id
        self._trace_ctx = trace_ctx
        self._pricing_model = pricing_model

    def create(self, **kwargs: Any) -> Any:
        """
        Identical signature to anthropic.messages.create().
        All kwargs are passed through unchanged.
        """
        call_tags = {**self._base_tags}

        # Allow per-call tag overrides via kostrack_tags kwarg
        extra_tags = kwargs.pop("kostrack_tags", {})
        call_tags.update(extra_tags)

        # Time the call
        start = time.monotonic()
        response = self._messages.create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        # Extract tokens and cost
        tokens = extract_tokens(response)
        model = extract_model(response) or kwargs.get("model", "")
        cost_usd = self._pricing.get_cost(
            provider="anthropic",
            model=model,
            tokens=tokens,
            pricing_model=self._pricing_model,
        )

        # Build CallRecord
        record = CallRecord(
            service_id=self._service_id,
            provider="anthropic",
            model=model,
            pricing_model=self._pricing_model,
            tokens=tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            tags=call_tags,
        )

        # Attach trace context if active
        from kostrack.tracing import get_active_trace
        active_trace = self._trace_ctx or get_active_trace()
        if active_trace:
            record.trace_id = active_trace.trace_id
            record.parent_span_id = active_trace.span_id
            active_trace.record_call(cost_usd, model=model)

        self._writer.write(record.to_row())

        logger.debug(
            "Anthropic call — model=%s input=%d output=%d cache_read=%d cost=$%.6f latency=%dms",
            model,
            tokens.input,
            tokens.output,
            tokens.cache_read,
            cost_usd,
            latency_ms,
        )

        return response

    def stream(self, **kwargs: Any) -> Any:
        """
        Streaming wrapper — collects usage from the final stream event.
        Falls back to non-streaming if usage not available in stream.
        """
        call_tags = {**self._base_tags}
        extra_tags = kwargs.pop("kostrack_tags", {})
        call_tags.update(extra_tags)

        start = time.monotonic()

        with self._messages.stream(**kwargs) as stream:
            yield from stream

            latency_ms = int((time.monotonic() - start) * 1000)

            # get_final_message() returns the accumulated response with usage
            try:
                final = stream.get_final_message()
                tokens = extract_tokens(final)
                model = extract_model(final) or kwargs.get("model", "")
            except Exception:
                logger.debug("Could not extract usage from stream — skipping cost record")
                return

        cost_usd = self._pricing.get_cost(
            provider="anthropic",
            model=model,
            tokens=tokens,
            pricing_model=self._pricing_model,
        )

        record = CallRecord(
            service_id=self._service_id,
            provider="anthropic",
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


class Anthropic:
    """
    kostrack drop-in replacement for anthropic.Anthropic.

    All anthropic.Anthropic kwargs are passed through.
    kostrack-specific kwargs:
        tags (dict)         — attribution tags applied to every call
        pricing_model (str) — 'per_token' (default) or 'batch'
        trace_ctx           — active TraceContext (set automatically by kostrack.trace())
    """

    def __init__(
        self,
        *,
        tags: dict[str, str] | None = None,
        pricing_model: str = "per_token",
        trace_ctx: TraceContext | None = None,
        # Injected by kostrack.configure() — not set by users directly
        _writer: AsyncBatchWriter | None = None,
        _pricing: PricingEngine | None = None,
        _service_id: str = "default",
        # Pass-through to anthropic.Anthropic
        **anthropic_kwargs: Any,
    ) -> None:
        self._client = _anthropic.Anthropic(**anthropic_kwargs)
        self._writer = _writer or _get_global_writer()
        self._pricing = _pricing or _get_global_pricing()
        self._service_id = _service_id
        self._base_tags = tags or {}
        self._pricing_model = pricing_model
        self._trace_ctx = trace_ctx

        self.messages = Messages(
            messages=self._client.messages,
            writer=self._writer,
            pricing=self._pricing,
            base_tags=self._base_tags,
            service_id=self._service_id,
            trace_ctx=self._trace_ctx,
            pricing_model=self._pricing_model,
        )

    def with_trace(self, trace_ctx: TraceContext) -> "Anthropic":
        """Return a new client bound to the given trace context."""
        return Anthropic(
            tags=self._base_tags,
            pricing_model=self._pricing_model,
            trace_ctx=trace_ctx,
            _writer=self._writer,
            _pricing=self._pricing,
            _service_id=self._service_id,
        )

    # Passthrough for any other anthropic.Anthropic attributes
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