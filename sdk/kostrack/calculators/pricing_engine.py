"""
kostrack — Pricing Engine

Loads current pricing from TimescaleDB at SDK init.
Refreshes periodically in the background.
Falls back to bundled defaults if DB is unavailable at startup.

Cost is calculated at write time — not query time — so dashboard
aggregation queries stay fast and historical costs remain accurate
even after pricing changes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import psycopg2
import psycopg2.extras

from kostrack.models import PricingEntry, TokenBreakdown

logger = logging.getLogger("kostrack.pricing")

# Refresh pricing from DB every 6 hours
REFRESH_INTERVAL = 6 * 60 * 60

# Bundled fallback pricing (March 2026)
# Used if TimescaleDB is unreachable at startup
_BUNDLED_PRICING: list[dict[str, Any]] = [
    # Anthropic
    {"provider": "anthropic", "model": "claude-sonnet-4-6",          "pricing_model": "per_token", "input_rate": 3.00/1e6,  "output_rate": 15.00/1e6, "cache_write_rate": 3.75/1e6,  "cache_read_rate": 0.30/1e6},
    {"provider": "anthropic", "model": "claude-opus-4-6",            "pricing_model": "per_token", "input_rate": 15.00/1e6, "output_rate": 75.00/1e6, "cache_write_rate": 18.75/1e6, "cache_read_rate": 1.50/1e6},
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",  "pricing_model": "per_token", "input_rate": 0.80/1e6,  "output_rate": 4.00/1e6,  "cache_write_rate": 1.00/1e6,  "cache_read_rate": 0.08/1e6},
    {"provider": "anthropic", "model": "claude-sonnet-4-6",          "pricing_model": "batch",     "input_rate": 1.50/1e6,  "output_rate": 7.50/1e6,  "cache_write_rate": None,      "cache_read_rate": None},
    {"provider": "anthropic", "model": "claude-opus-4-6",            "pricing_model": "batch",     "input_rate": 7.50/1e6,  "output_rate": 37.50/1e6, "cache_write_rate": None,      "cache_read_rate": None},
    # OpenAI
    {"provider": "openai", "model": "gpt-4o",      "pricing_model": "per_token", "input_rate": 2.50/1e6,  "output_rate": 10.00/1e6, "cache_write_rate": None, "cache_read_rate": 1.25/1e6},
    {"provider": "openai", "model": "gpt-4o-mini", "pricing_model": "per_token", "input_rate": 0.15/1e6,  "output_rate": 0.60/1e6,  "cache_write_rate": None, "cache_read_rate": 0.075/1e6},
    {"provider": "openai", "model": "o1",          "pricing_model": "per_token", "input_rate": 15.00/1e6, "output_rate": 60.00/1e6, "cache_write_rate": None, "cache_read_rate": 7.50/1e6},
    {"provider": "openai", "model": "o3-mini",     "pricing_model": "per_token", "input_rate": 1.10/1e6,  "output_rate": 4.40/1e6,  "cache_write_rate": None, "cache_read_rate": 0.55/1e6},
    # DeepSeek
    {"provider": "deepseek", "model": "deepseek-chat",      "pricing_model": "per_token", "input_rate": 0.27/1e6,  "output_rate": 1.10/1e6, "cache_write_rate": None, "cache_read_rate": 0.07/1e6},
    {"provider": "deepseek", "model": "deepseek-reasoner",  "pricing_model": "per_token", "input_rate": 0.55/1e6,  "output_rate": 2.19/1e6, "cache_write_rate": None, "cache_read_rate": 0.14/1e6},
    # Gemini
    {"provider": "gemini", "model": "gemini-2.0-flash",      "pricing_model": "per_token", "input_rate": 0.10/1e6,  "output_rate": 0.40/1e6, "cache_write_rate": None, "cache_read_rate": 0.025/1e6},
    {"provider": "gemini", "model": "gemini-2.0-flash-lite",  "pricing_model": "per_token", "input_rate": 0.075/1e6, "output_rate": 0.30/1e6, "cache_write_rate": None, "cache_read_rate": None},
    {"provider": "gemini", "model": "gemini-1.5-pro",         "pricing_model": "per_token", "input_rate": 1.25/1e6,  "output_rate": 5.00/1e6, "cache_write_rate": None, "cache_read_rate": 0.3125/1e6},
    {"provider": "gemini", "model": "gemini-1.5-flash",       "pricing_model": "per_token", "input_rate": 0.075/1e6, "output_rate": 0.30/1e6, "cache_write_rate": None, "cache_read_rate": 0.01875/1e6},
]


class PricingEngine:
    """
    In-memory pricing cache with background refresh.

    Key: (provider, model, pricing_model)
    Value: PricingEntry
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._cache: dict[tuple[str, str, str], PricingEntry] = {}
        self._lock = threading.RLock()
        self._last_refresh = 0.0

        # Load bundled defaults immediately — always available
        self._load_bundled()

        # Attempt DB load — overwrites bundled if successful
        self._load_from_db()

        # Background refresh thread
        self._refresher = threading.Thread(
            target=self._refresh_loop,
            name="kostrack-pricing",
            daemon=True,
        )
        self._refresher.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cost(
        self,
        provider: str,
        model: str,
        tokens: TokenBreakdown,
        pricing_model: str = "per_token",
    ) -> float:
        """
        Calculate USD cost for a set of tokens.
        Falls back to per_token pricing if the specific pricing_model
        isn't found. Returns 0.0 and logs a warning if model unknown.
        """
        entry = self._lookup(provider, model, pricing_model)
        if entry is None:
            entry = self._lookup(provider, model, "per_token")
        if entry is None:
            logger.warning(
                "No pricing found for %s/%s (%s) — cost recorded as 0.0",
                provider, model, pricing_model,
            )
            return 0.0
        return entry.calculate_cost(tokens)

    def known_models(self) -> list[str]:
        with self._lock:
            return sorted({model for _, model, _ in self._cache})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lookup(
        self, provider: str, model: str, pricing_model: str
    ) -> PricingEntry | None:
        with self._lock:
            return self._cache.get((provider, model, pricing_model))

    def _load_bundled(self) -> None:
        with self._lock:
            for row in _BUNDLED_PRICING:
                self._upsert(row)
        logger.debug("Loaded %d bundled pricing entries", len(_BUNDLED_PRICING))

    def _load_from_db(self) -> None:
        try:
            conn = psycopg2.connect(self.dsn)
            conn.autocommit = True
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT provider, model, pricing_model,
                           input_rate, output_rate,
                           cache_write_rate, cache_read_rate
                    FROM pricing_current
                """)
                rows = cur.fetchall()
            conn.close()

            with self._lock:
                for row in rows:
                    self._upsert(dict(row))

            self._last_refresh = time.monotonic()
            logger.info("Loaded %d pricing entries from TimescaleDB", len(rows))

        except Exception as exc:
            logger.warning(
                "Could not load pricing from DB — using bundled defaults: %s", exc
            )

    def _upsert(self, row: dict[str, Any]) -> None:
        """Write one pricing row into the cache. Caller holds lock."""
        key = (row["provider"], row["model"], row["pricing_model"])
        self._cache[key] = PricingEntry(
            provider=row["provider"],
            model=row["model"],
            pricing_model=row["pricing_model"],
            input_rate=float(row["input_rate"] or 0),
            output_rate=float(row["output_rate"] or 0),
            cache_write_rate=float(row["cache_write_rate"]) if row.get("cache_write_rate") else None,
            cache_read_rate=float(row["cache_read_rate"]) if row.get("cache_read_rate") else None,
        )

    def _refresh_loop(self) -> None:
        while True:
            time.sleep(REFRESH_INTERVAL)
            logger.info("Refreshing pricing from TimescaleDB...")
            self._load_from_db()