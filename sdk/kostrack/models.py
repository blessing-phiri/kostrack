"""
kostrack — Core Data Models

CallRecord is the central data structure. Every LLM API call
produces one CallRecord which flows through the cost calculator
and into the async batch writer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TokenBreakdown:
    """
    Provider-specific token counts.

    Flat fields cover the common case.
    Extra dict captures anything provider-specific.
    """
    input: int = 0
    output: int = 0
    cache_write: int = 0      # Anthropic: tokens written to cache
    cache_read: int = 0       # Anthropic cache hits / OpenAI cached prompt tokens
    thinking: int = 0         # Anthropic extended thinking tokens
    extra: dict[str, int] = field(default_factory=dict)

    def to_jsonb(self) -> dict[str, Any]:
        d = {
            "input": self.input,
            "output": self.output,
        }
        if self.cache_write:
            d["cache_write"] = self.cache_write
        if self.cache_read:
            d["cache_read"] = self.cache_read
        if self.thinking:
            d["thinking"] = self.thinking
        d.update(self.extra)
        return d

    @property
    def total_cached(self) -> int:
        """Flat cached_tokens column — total cache hits."""
        return self.cache_read


@dataclass
class CallRecord:
    """
    One LLM API call — fully attributed and costed.
    Maps 1:1 to a row in llm_calls.
    """

    # Core identity
    time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    service_id: str = "default"
    provider: str = ""          # 'anthropic' | 'openai' | 'gemini'
    model: str = ""
    pricing_model: str = "per_token"

    # Token counts
    tokens: TokenBreakdown = field(default_factory=TokenBreakdown)

    # Cost — calculated by provider CostCalculator
    cost_usd: float = 0.0

    # Performance
    latency_ms: int | None = None

    # Distributed tracing
    trace_id: uuid.UUID | None = None
    span_id: uuid.UUID = field(default_factory=uuid.uuid4)
    parent_span_id: uuid.UUID | None = None

    # Attribution — arbitrary tags dict
    # Reserved keys: project, feature, user_id, team, environment
    tags: dict[str, str] = field(default_factory=dict)

    # Anything else
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Serialize to a dict ready for TimescaleDB insert."""
        return {
            "time": self.time,
            "service_id": self.service_id,
            "provider": self.provider,
            "model": self.model,
            "pricing_model": self.pricing_model,
            "input_tokens": self.tokens.input,
            "output_tokens": self.tokens.output,
            "cached_tokens": self.tokens.total_cached,
            "cost_usd": self.cost_usd,
            "token_breakdown": self.tokens.to_jsonb(),
            "latency_ms": self.latency_ms,
            "trace_id": str(self.trace_id) if self.trace_id else None,
            "span_id": str(self.span_id),
            "parent_span_id": str(self.parent_span_id) if self.parent_span_id else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class PricingEntry:
    """
    One row from the pricing table.
    Loaded at SDK init and cached in memory.
    """
    provider: str
    model: str
    pricing_model: str
    input_rate: float
    output_rate: float
    cache_write_rate: float | None = None
    cache_read_rate: float | None = None

    def calculate_cost(self, tokens: TokenBreakdown) -> float:
        cost = (
            tokens.input * self.input_rate
            + tokens.output * self.output_rate
        )
        if self.cache_write_rate and tokens.cache_write:
            cost += tokens.cache_write * self.cache_write_rate
        if self.cache_read_rate and tokens.cache_read:
            cost += tokens.cache_read * self.cache_read_rate
        return cost


@dataclass
class TraceContext:
    """
    Active trace — holds IDs for the current workflow.
    Used by the context manager in kostrack.trace().

    model_costs: per-model cost breakdown across all calls in this trace.
    Useful for multi-model agentic workflows where you want to know
    "how much did the R1 reasoning step cost vs the Claude extraction step".
    """
    trace_id: uuid.UUID = field(default_factory=uuid.uuid4)
    span_id: uuid.UUID = field(default_factory=uuid.uuid4)
    parent_span_id: uuid.UUID | None = None
    tags: dict[str, str] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    call_count: int = 0
    model_costs: dict[str, float] = field(default_factory=dict)

    def child_span(self) -> "TraceContext":
        """Create a child span inheriting this trace's ID."""
        return TraceContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            tags=self.tags.copy(),
        )

    def record_call(self, cost_usd: float, model: str = "") -> None:
        self.total_cost_usd += cost_usd
        self.call_count += 1
        if model:
            self.model_costs[model] = self.model_costs.get(model, 0.0) + cost_usd

    def cost_breakdown(self) -> list[dict[str, Any]]:
        """
        Return per-model cost sorted by spend descending.

        Example:
            [
                {"model": "deepseek-reasoner",  "cost_usd": 0.0142, "pct": 71.0},
                {"model": "claude-sonnet-4-6",   "cost_usd": 0.0058, "pct": 29.0},
            ]
        """
        if not self.model_costs or self.total_cost_usd == 0:
            return []
        return sorted(
            [
                {
                    "model": model,
                    "cost_usd": cost,
                    "pct": round(cost / self.total_cost_usd * 100, 1),
                }
                for model, cost in self.model_costs.items()
            ],
            key=lambda x: x["cost_usd"],
            reverse=True,
        )