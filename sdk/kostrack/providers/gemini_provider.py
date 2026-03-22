"""
TokenLedger — Gemini Provider Wrapper

Wraps google.genai.Client (new google-genai SDK).
Intercepts models.generate_content() calls, extracts token usage,
calculates cost, and writes a CallRecord asynchronously.

Usage:
    # Before
    from google import genai
    client = genai.Client(api_key="...")
    response = client.models.generate_content(model="gemini-2.0-flash", contents="Hello")

    # After
    from kostrack import GenerativeModel
    model = GenerativeModel(
        "gemini-2.0-flash",
        tags={"project": "openmanagr", "feature": "document-ocr"}
    )
    response = model.generate_content("Hello")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from google import genai as _genai

from kostrack.calculators.gemini_calc import extract_tokens
from kostrack.models import CallRecord, TraceContext
from kostrack.writers.batch_writer import AsyncBatchWriter
from kostrack.calculators.pricing_engine import PricingEngine

logger = logging.getLogger("kostrack.providers.gemini")


class GenerativeModel:
    """
    TokenLedger wrapper providing a model-centric interface over google.genai.Client.

    TokenLedger-specific kwargs:
        api_key (str)       — Gemini API key (defaults to GEMINI_API_KEY env var)
        tags (dict)         — attribution tags applied to every call
        pricing_model (str) — 'per_token' (default)
        trace_ctx           — active TraceContext
    """

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        tags: dict[str, str] | None = None,
        pricing_model: str = "per_token",
        trace_ctx: TraceContext | None = None,
        _writer: AsyncBatchWriter | None = None,
        _pricing: PricingEngine | None = None,
        _service_id: str = "default",
    ) -> None:
        self._model_name = model_name.replace("models/", "")
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = _genai.Client(api_key=resolved_key)
        self._writer = _writer or _get_global_writer()
        self._pricing = _pricing or _get_global_pricing()
        self._service_id = _service_id
        self._base_tags = tags or {}
        self._pricing_model = pricing_model
        self._trace_ctx = trace_ctx

    def generate_content(self, contents: Any, **kwargs: Any) -> Any:
        """
        Wraps client.models.generate_content().
        All kwargs are passed through to the Gemini API unchanged.
        """
        call_tags = {**self._base_tags}
        extra_tags = kwargs.pop("kostrack_tags", {})
        call_tags.update(extra_tags)

        start = time.monotonic()
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            **kwargs,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        tokens = extract_tokens(response)
        cost_usd = self._pricing.get_cost(
            provider="gemini",
            model=self._model_name,
            tokens=tokens,
            pricing_model=self._pricing_model,
        )

        record = CallRecord(
            service_id=self._service_id,
            provider="gemini",
            model=self._model_name,
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
            "Gemini call — model=%s prompt=%d candidates=%d cached=%d cost=$%.6f latency=%dms",
            self._model_name,
            tokens.input,
            tokens.output,
            tokens.cache_read,
            cost_usd,
            latency_ms,
        )

        return response

    def generate_content_stream(self, contents: Any, **kwargs: Any) -> Any:
        """
        Streaming variant — yields chunks, captures usage_metadata from final chunk.
        """
        call_tags = {**self._base_tags}
        extra_tags = kwargs.pop("kostrack_tags", {})
        call_tags.update(extra_tags)

        start = time.monotonic()
        last_chunk = None

        for chunk in self._client.models.generate_content_stream(
            model=self._model_name, contents=contents, **kwargs
        ):
            last_chunk = chunk
            yield chunk

        latency_ms = int((time.monotonic() - start) * 1000)
        if last_chunk is None:
            return

        tokens = extract_tokens(last_chunk)
        cost_usd = self._pricing.get_cost(
            provider="gemini",
            model=self._model_name,
            tokens=tokens,
            pricing_model=self._pricing_model,
        )

        record = CallRecord(
            service_id=self._service_id,
            provider="gemini",
            model=self._model_name,
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

    def with_trace(self, trace_ctx: TraceContext) -> "GenerativeModel":
        return GenerativeModel(
            self._model_name,
            tags=self._base_tags,
            pricing_model=self._pricing_model,
            trace_ctx=trace_ctx,
            _writer=self._writer,
            _pricing=self._pricing,
            _service_id=self._service_id,
        )


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
