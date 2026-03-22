"""
kostrack — Trace Context Manager

Tracks cost across multi-step agentic workflows.
All LLM calls made within a trace() block are linked
via trace_id and parent_span_id, enabling cost rollup
per workflow run in the Grafana agentic dashboard.

Usage:
    with kostrack.trace(tags={"project": "openmanagr", "feature": "month-end"}) as trace:
        result = agent.run(task)
    
    print(f"Workflow cost: ${trace.total_cost_usd:.6f}")
    print(f"API calls made: {trace.call_count}")

Nested spans (for sub-graphs):
    with kostrack.trace(tags={"feature": "invoice-pipeline"}) as root:
        with kostrack.span("validate", parent=root) as validation_span:
            client.messages.create(...)
        with kostrack.span("post", parent=root) as posting_span:
            client.messages.create(...)
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Generator

from kostrack.models import TraceContext

logger = logging.getLogger("kostrack.trace")

# Thread-local storage — each thread has its own active trace stack
_local = threading.local()


def get_active_trace() -> TraceContext | None:
    """Return the innermost active trace for the current thread."""
    stack: list[TraceContext] = getattr(_local, "stack", [])
    return stack[-1] if stack else None


def _push_trace(ctx: TraceContext) -> None:
    if not hasattr(_local, "stack"):
        _local.stack = []
    _local.stack.append(ctx)


def _pop_trace() -> None:
    if hasattr(_local, "stack") and _local.stack:
        _local.stack.pop()


@contextmanager
def trace(
    tags: dict[str, str] | None = None,
) -> Generator[TraceContext, None, None]:
    """
    Context manager that opens a root trace span.

    All LLM calls made inside this block (on the same thread)
    are automatically linked to this trace.

    Args:
        tags: Attribution tags — project, feature, team, environment, etc.
              These override any tags set on the client.
    """
    ctx = TraceContext(tags=tags or {})
    _push_trace(ctx)
    logger.debug("Trace started — id=%s tags=%s", ctx.trace_id, ctx.tags)
    try:
        yield ctx
    finally:
        _pop_trace()
        logger.debug(
            "Trace complete — id=%s calls=%d cost=$%.6f",
            ctx.trace_id,
            ctx.call_count,
            ctx.total_cost_usd,
        )


@contextmanager
def span(
    name: str,
    parent: TraceContext | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[TraceContext, None, None]:
    """
    Context manager that opens a child span within an existing trace.

    If parent is None, inherits the active trace from the current thread.
    If no active trace exists, creates a standalone trace.

    Args:
        name:   Span name — stored in tags["span_name"] for filtering.
        parent: Explicit parent TraceContext. Defaults to active thread trace.
        tags:   Additional tags merged with parent tags.
    """
    active = parent or get_active_trace()

    if active:
        ctx = active.child_span()
    else:
        ctx = TraceContext()
        logger.debug("span() called outside a trace() block — creating standalone trace")

    ctx.tags["span_name"] = name
    if tags:
        ctx.tags.update(tags)

    _push_trace(ctx)
    logger.debug("Span started — name=%s trace=%s span=%s", name, ctx.trace_id, ctx.span_id)
    try:
        yield ctx
    finally:
        _pop_trace()
        if active:
            # Roll up total cost and per-model breakdown to the parent trace.
            # Incrementing call_count by 1 here represents the span as a unit;
            # individual call counts are already tracked on the child ctx.
            active.total_cost_usd += ctx.total_cost_usd
            active.call_count += ctx.call_count
            for model, cost in ctx.model_costs.items():
                active.model_costs[model] = active.model_costs.get(model, 0.0) + cost
        logger.debug(
            "Span complete — name=%s calls=%d cost=$%.6f",
            name,
            ctx.call_count,
            ctx.total_cost_usd,
        )