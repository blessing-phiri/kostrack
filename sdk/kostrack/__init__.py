"""
kostrack — AI API Cost Governance

Quick start:
    import kostrack

    kostrack.configure(dsn="postgresql://kostrack:changeme@localhost/kostrack")

    from kostrack import Anthropic

    client = Anthropic(tags={"project": "myapp", "feature": "summariser"})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Summarise this invoice..."}]
    )

With agentic tracing:
    with kostrack.trace(tags={"project": "openmanagr", "feature": "month-end"}) as t:
        response = client.messages.create(...)

    print(f"Total cost: ${t.total_cost_usd:.6f}")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from kostrack.writers.batch_writer import AsyncBatchWriter
from kostrack.calculators.pricing_engine import PricingEngine
from kostrack.providers.anthropic_provider import Anthropic
import kostrack.providers.anthropic_provider as _anthropic_provider
from kostrack.tracing import trace, span, get_active_trace
from kostrack.models import TraceContext, CallRecord, TokenBreakdown

__version__ = "0.1.0"
__all__ = [
    "configure",
    "shutdown",
    "health",
    "Anthropic",
    "trace",
    "span",
    "get_active_trace",
    "TraceContext",
    "CallRecord",
    "TokenBreakdown",
]

logger = logging.getLogger("kostrack")

_writer: AsyncBatchWriter | None = None
_pricing: PricingEngine | None = None


def configure(
    dsn: str | None = None,
    *,
    service_id: str = "default",
    flush_interval: float = 5.0,
    max_batch_size: int = 100,
    sqlite_path: Path | None = None,
    fail_open: bool = True,
    log_level: str = "WARNING",
) -> None:
    """
    Initialise kostrack. Call once at application startup.

    Args:
        dsn:             PostgreSQL DSN for TimescaleDB.
                         Defaults to TOKENLEDGER_DSN env var.
        service_id:      Identifies this service in writer_health and queries.
        flush_interval:  Seconds between write batches.
        max_batch_size:  Max rows per TimescaleDB insert.
        sqlite_path:     Override default SQLite buffer path.
        fail_open:       If True, swallow all write errors silently.
        log_level:       Log level for kostrack loggers.
    """
    global _writer, _pricing

    logging.getLogger("kostrack").setLevel(
        getattr(logging, log_level.upper(), logging.WARNING)
    )

    resolved_dsn = dsn or os.environ.get("TOKENLEDGER_DSN")
    if not resolved_dsn:
        raise ValueError(
            "No DSN provided. Pass dsn= to configure() or set TOKENLEDGER_DSN env var."
        )

    _pricing = PricingEngine(dsn=resolved_dsn)
    _writer = AsyncBatchWriter(
        dsn=resolved_dsn,
        flush_interval=flush_interval,
        max_batch_size=max_batch_size,
        sqlite_path=sqlite_path,
        service_id=service_id,
        fail_open=fail_open,
    )

    # Inject into provider modules
    _anthropic_provider._global_writer = _writer
    _anthropic_provider._global_pricing = _pricing

    logger.info("kostrack configured — service_id=%s", service_id)


def health() -> dict[str, Any]:
    """Return current writer health state."""
    if _writer is None:
        return {"status": "not_configured"}
    return {"status": "ok", **_writer.health()}


def shutdown(timeout: float = 10.0) -> None:
    """Graceful shutdown — flush remaining queue before process exits."""
    if _writer:
        _writer.stop(timeout=timeout)