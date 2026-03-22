"""
kostrack — Pricing Sync

Keeps the pricing table in TimescaleDB current by periodically
fetching the latest model pricing from each provider.

Since providers don't publish machine-readable pricing APIs,
this module maintains a versioned table of known prices and
provides a mechanism to insert new pricing entries when prices change.

Usage:
    # Manual update (e.g. run after a provider price change)
    from kostrack.sync.pricing_sync import PricingSync
    sync = PricingSync(dsn="postgresql://...")
    sync.run()

    # Scheduled background refresh (call at app startup)
    sync.start_background(interval_hours=24)

    # Add a new model or price change manually
    sync.upsert(
        provider="anthropic",
        model="claude-new-model",
        pricing_model="per_token",
        input_rate=3.00 / 1_000_000,
        output_rate=15.00 / 1_000_000,
        cache_write_rate=3.75 / 1_000_000,
        cache_read_rate=0.30 / 1_000_000,
    )
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger("kostrack.sync.pricing")

# ──────────────────────────────────────────────────────────────────────────────
# Current pricing table — update this when providers change prices
# All rates are USD per token
# Last verified: March 2026
# ──────────────────────────────────────────────────────────────────────────────

CURRENT_PRICING: list[dict[str, Any]] = [

    # ── Anthropic ──────────────────────────────────────────────────────────────
    {
        "provider": "anthropic", "model": "claude-sonnet-4-6",
        "pricing_model": "per_token",
        "input_rate":       3.00 / 1_000_000,
        "output_rate":     15.00 / 1_000_000,
        "cache_write_rate": 3.75 / 1_000_000,
        "cache_read_rate":  0.30 / 1_000_000,
        "notes": "claude-sonnet-4-6, 200k context",
    },
    {
        "provider": "anthropic", "model": "claude-opus-4-6",
        "pricing_model": "per_token",
        "input_rate":      15.00 / 1_000_000,
        "output_rate":     75.00 / 1_000_000,
        "cache_write_rate":18.75 / 1_000_000,
        "cache_read_rate":  1.50 / 1_000_000,
        "notes": "claude-opus-4-6, 200k context",
    },
    {
        "provider": "anthropic", "model": "claude-haiku-4-5-20251001",
        "pricing_model": "per_token",
        "input_rate":      0.80 / 1_000_000,
        "output_rate":     4.00 / 1_000_000,
        "cache_write_rate":1.00 / 1_000_000,
        "cache_read_rate": 0.08 / 1_000_000,
        "notes": "claude-haiku-4-5, 200k context",
    },
    {
        "provider": "anthropic", "model": "claude-sonnet-4-6",
        "pricing_model": "batch",
        "input_rate":      1.50 / 1_000_000,
        "output_rate":     7.50 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  None,
        "notes": "Batch API — 50% discount, 24hr turnaround",
    },
    {
        "provider": "anthropic", "model": "claude-opus-4-6",
        "pricing_model": "batch",
        "input_rate":      7.50 / 1_000_000,
        "output_rate":    37.50 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  None,
        "notes": "Batch API — 50% discount",
    },
    {
        "provider": "anthropic", "model": "claude-haiku-4-5-20251001",
        "pricing_model": "batch",
        "input_rate":      0.40 / 1_000_000,
        "output_rate":     2.00 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  None,
        "notes": "Batch API — 50% discount",
    },

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    {
        "provider": "openai", "model": "gpt-4o",
        "pricing_model": "per_token",
        "input_rate":      2.50 / 1_000_000,
        "output_rate":    10.00 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  1.25 / 1_000_000,
        "notes": "gpt-4o, 128k context",
    },
    {
        "provider": "openai", "model": "gpt-4o-mini",
        "pricing_model": "per_token",
        "input_rate":      0.15 / 1_000_000,
        "output_rate":     0.60 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  0.075 / 1_000_000,
        "notes": "gpt-4o-mini, 128k context",
    },
    {
        "provider": "openai", "model": "o1",
        "pricing_model": "per_token",
        "input_rate":     15.00 / 1_000_000,
        "output_rate":    60.00 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  7.50 / 1_000_000,
        "notes": "o1 reasoning model, 200k context",
    },
    {
        "provider": "openai", "model": "o3-mini",
        "pricing_model": "per_token",
        "input_rate":      1.10 / 1_000_000,
        "output_rate":     4.40 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  0.55 / 1_000_000,
        "notes": "o3-mini reasoning model",
    },
    {
        "provider": "openai", "model": "gpt-4o",
        "pricing_model": "batch",
        "input_rate":      1.25 / 1_000_000,
        "output_rate":     5.00 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  None,
        "notes": "Batch API — 50% discount",
    },
    {
        "provider": "openai", "model": "gpt-4o-mini",
        "pricing_model": "batch",
        "input_rate":      0.075 / 1_000_000,
        "output_rate":     0.30 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  None,
        "notes": "Batch API — 50% discount",
    },

    # ── DeepSeek ───────────────────────────────────────────────────────────────
    {
        "provider": "deepseek", "model": "deepseek-chat",
        "pricing_model": "per_token",
        "input_rate":      0.27 / 1_000_000,
        "output_rate":     1.10 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  0.07 / 1_000_000,
        "notes": "DeepSeek-V3, 64k context. Cache hit = 0.07/M input tokens.",
    },
    {
        "provider": "deepseek", "model": "deepseek-reasoner",
        "pricing_model": "per_token",
        "input_rate":      0.55 / 1_000_000,
        "output_rate":     2.19 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  0.14 / 1_000_000,
        "notes": "DeepSeek-R1. Reasoning tokens billed as output. Cache hit = 0.14/M.",
    },

    # ── Gemini ─────────────────────────────────────────────────────────────────
    {
        "provider": "gemini", "model": "gemini-2.0-flash",
        "pricing_model": "per_token",
        "input_rate":      0.10 / 1_000_000,
        "output_rate":     0.40 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  0.025 / 1_000_000,
        "notes": "gemini-2.0-flash, 1M context",
    },
    {
        "provider": "gemini", "model": "gemini-2.0-flash-lite",
        "pricing_model": "per_token",
        "input_rate":      0.075 / 1_000_000,
        "output_rate":     0.30 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  None,
        "notes": "gemini-2.0-flash-lite",
    },
    {
        "provider": "gemini", "model": "gemini-1.5-pro",
        "pricing_model": "per_token",
        "input_rate":      1.25 / 1_000_000,
        "output_rate":     5.00 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  0.3125 / 1_000_000,
        "notes": "gemini-1.5-pro, 2M context",
    },
    {
        "provider": "gemini", "model": "gemini-1.5-flash",
        "pricing_model": "per_token",
        "input_rate":      0.075 / 1_000_000,
        "output_rate":     0.30 / 1_000_000,
        "cache_write_rate": None,
        "cache_read_rate":  0.01875 / 1_000_000,
        "notes": "gemini-1.5-flash, 1M context",
    },
]


class PricingSync:
    """
    Keeps the pricing table current.

    Compares CURRENT_PRICING against what's in the DB.
    If a price entry doesn't exist yet, inserts it with effective_from=now.
    Existing entries are never modified — new rows are added for price changes,
    preserving historical accuracy.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> dict[str, int]:
        """
        Sync CURRENT_PRICING to the DB once.
        Returns {"inserted": N, "already_current": M}.
        """
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = False
        try:
            result = self._sync(conn)
            conn.commit()
            logger.info(
                "Pricing sync complete — inserted=%d already_current=%d",
                result["inserted"], result["already_current"],
            )
            return result
        except Exception as exc:
            conn.rollback()
            logger.error("Pricing sync failed: %s", exc)
            raise
        finally:
            conn.close()

    def upsert(
        self,
        provider: str,
        model: str,
        pricing_model: str = "per_token",
        input_rate: float = 0.0,
        output_rate: float = 0.0,
        cache_write_rate: float | None = None,
        cache_read_rate: float | None = None,
        unit: str = "USD",
        notes: str | None = None,
    ) -> None:
        """
        Insert a new pricing entry for a model.
        Used when adding new models or recording a price change.
        The previous entry's effective_to is set to now, and a new
        entry is inserted with effective_from=now.
        """
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = False
        try:
            now = datetime.now(timezone.utc)
            with conn.cursor() as cur:
                # Close previous entry
                cur.execute(
                    """
                    UPDATE pricing SET effective_to = %s
                    WHERE provider = %s AND model = %s
                      AND pricing_model = %s AND effective_to IS NULL
                    """,
                    (now, provider, model, pricing_model),
                )
                # Insert new entry
                cur.execute(
                    """
                    INSERT INTO pricing (
                        provider, model, pricing_model,
                        input_rate, output_rate,
                        cache_write_rate, cache_read_rate,
                        unit, effective_from, metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        provider, model, pricing_model,
                        input_rate, output_rate,
                        cache_write_rate, cache_read_rate,
                        unit, now,
                        psycopg2.extras.Json({"notes": notes} if notes else {}),
                    ),
                )
            conn.commit()
            logger.info("Upserted pricing for %s/%s (%s)", provider, model, pricing_model)
        except Exception as exc:
            conn.rollback()
            logger.error("Pricing upsert failed: %s", exc)
            raise
        finally:
            conn.close()

    def start_background(self, interval_hours: float = 24.0) -> None:
        """
        Start a background thread that re-syncs pricing every N hours.
        Safe to call multiple times — only one thread runs at a time.
        """
        if self._thread and self._thread.is_alive():
            logger.debug("Pricing sync background thread already running")
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(interval_hours * 3600,),
            name="kostrack-pricing-sync",
            daemon=True,
        )
        self._thread.start()
        logger.info("Pricing sync background thread started (interval=%.1fh)", interval_hours)

    def stop_background(self) -> None:
        self._stop.set()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _sync(self, conn: psycopg2.extensions.connection) -> dict[str, int]:
        inserted = 0
        already_current = 0

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Load all currently active entries
            cur.execute(
                "SELECT provider, model, pricing_model, input_rate, output_rate "
                "FROM pricing WHERE effective_to IS NULL"
            )
            active: dict[tuple, dict] = {
                (r["provider"], r["model"], r["pricing_model"]): dict(r)
                for r in cur.fetchall()
            }

        now = datetime.now(timezone.utc)

        with conn.cursor() as cur:
            for entry in CURRENT_PRICING:
                key = (entry["provider"], entry["model"], entry["pricing_model"])
                existing = active.get(key)

                if existing:
                    # Check if rates have changed
                    rates_match = (
                        abs(float(existing["input_rate"] or 0) - (entry["input_rate"] or 0)) < 1e-12
                        and abs(float(existing["output_rate"] or 0) - (entry["output_rate"] or 0)) < 1e-12
                    )
                    if rates_match:
                        already_current += 1
                        continue
                    # Rates changed — close existing entry
                    cur.execute(
                        "UPDATE pricing SET effective_to = %s "
                        "WHERE provider=%s AND model=%s AND pricing_model=%s AND effective_to IS NULL",
                        (now, entry["provider"], entry["model"], entry["pricing_model"]),
                    )

                # Insert new entry
                cur.execute(
                    """
                    INSERT INTO pricing (
                        provider, model, pricing_model,
                        input_rate, output_rate,
                        cache_write_rate, cache_read_rate,
                        unit, effective_from, effective_to, metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,'USD',%s,NULL,%s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        entry["provider"], entry["model"], entry["pricing_model"],
                        entry["input_rate"], entry["output_rate"],
                        entry.get("cache_write_rate"), entry.get("cache_read_rate"),
                        now,
                        psycopg2.extras.Json({"notes": entry.get("notes", "")}),
                    ),
                )
                inserted += 1

        return {"inserted": inserted, "already_current": already_current}

    def _loop(self, interval_seconds: float) -> None:
        # Run immediately on start
        try:
            self.run()
        except Exception as exc:
            logger.warning("Initial pricing sync failed: %s", exc)

        while not self._stop.wait(timeout=interval_seconds):
            try:
                self.run()
            except Exception as exc:
                logger.warning("Scheduled pricing sync failed: %s", exc)
