"""
kostrack — Budget Enforcer

Checks current spend against configured budgets.
Can be used in two modes:

1. Alert-only (default) — logs a warning when threshold is crossed
2. Enforce mode — raises BudgetExceededError before the call is made

The enforcer queries the DB periodically (not per-call) and caches
results, so it never adds meaningful latency to API calls.

Usage:
    from kostrack.budget import BudgetEnforcer, BudgetExceededError

    enforcer = BudgetEnforcer(dsn="postgresql://...")

    # Check before a call (raises if enforce=True in budgets table)
    enforcer.check(tags={"project": "openmanagr"}, estimated_cost=0.005)

    # Or use as a context manager
    with enforcer.guard(tags={"project": "openmanagr"}):
        response = client.messages.create(...)
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
import psycopg2.extras

logger = logging.getLogger("kostrack.budget")


class BudgetExceededError(Exception):
    """
    Raised when a call would exceed a configured hard budget limit.
    Only raised when the budget's enforce=True.
    """
    def __init__(self, tag_key: str, tag_value: str, period: str,
                 spent: float, limit: float) -> None:
        self.tag_key = tag_key
        self.tag_value = tag_value
        self.period = period
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"Budget exceeded: {tag_key}={tag_value} ({period}) "
            f"spent=${spent:.4f} / limit=${limit:.2f}"
        )


class BudgetEnforcer:
    """
    Caches budget state and checks it without blocking LLM calls.

    Cache refreshes every `cache_ttl` seconds (default: 60).
    This means enforcement lags by up to 60 seconds — acceptable
    for cost governance, unacceptable for financial transactions.
    """

    def __init__(
        self,
        dsn: str,
        cache_ttl: float = 60.0,
    ) -> None:
        self.dsn = dsn
        self.cache_ttl = cache_ttl
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}  # key → budget state
        self._last_refresh = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        tags: dict[str, str],
        estimated_cost: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Check all budgets that match any of the given tags.

        Returns list of budget states (for logging/monitoring).
        Raises BudgetExceededError if enforce=True and limit exceeded.
        Logs a warning if alert_threshold crossed.
        """
        self._maybe_refresh()

        triggered = []
        with self._lock:
            for key, state in self._cache.items():
                tag_key = state["tag_key"]
                tag_value = state["tag_value"]

                # Does this budget match any of the provided tags?
                if tags.get(tag_key) != tag_value:
                    continue

                spent = state["spent"] + estimated_cost
                limit = float(state["limit_usd"])
                threshold = float(state["alert_threshold"])
                enforce = state["enforce"]
                period = state["period"]

                pct = spent / limit if limit > 0 else 0

                if enforce and pct >= 1.0:
                    raise BudgetExceededError(tag_key, tag_value, period, spent, limit)

                if pct >= threshold:
                    logger.warning(
                        "Budget alert: %s=%s (%s) at %.1f%% ($%.4f / $%.2f)%s",
                        tag_key, tag_value, period,
                        pct * 100, spent, limit,
                        " — HARD LIMIT" if enforce else "",
                    )

                triggered.append({
                    "tag_key": tag_key,
                    "tag_value": tag_value,
                    "period": period,
                    "spent": spent,
                    "limit": limit,
                    "pct": pct,
                    "enforce": enforce,
                })

        return triggered

    @contextmanager
    def guard(
        self,
        tags: dict[str, str],
        estimated_cost: float = 0.0,
    ) -> Generator[None, None, None]:
        """
        Context manager that checks budget before the block executes.
        Use to wrap individual LLM calls or workflow sections.
        """
        self.check(tags, estimated_cost)
        yield

    def get_status(self) -> list[dict[str, Any]]:
        """Return current budget status for all configured budgets."""
        self._maybe_refresh()
        with self._lock:
            return list(self._cache.values())

    def set_budget(
        self,
        tag_key: str,
        tag_value: str,
        period: str,
        limit_usd: float,
        alert_threshold: float = 0.80,
        enforce: bool = False,
        service_id: str = "default",
    ) -> None:
        """
        Insert or update a budget in the DB.

        Args:
            tag_key:         e.g. "project"
            tag_value:       e.g. "openmanagr"
            period:          "daily" | "weekly" | "monthly"
            limit_usd:       Budget cap in USD
            alert_threshold: Fraction at which to alert (0.8 = 80%)
            enforce:         If True, calls raise BudgetExceededError at 100%
        """
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO budgets
                        (service_id, tag_key, tag_value, period,
                         limit_usd, alert_threshold, enforce)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (service_id, tag_key, tag_value, period)
                    DO UPDATE SET
                        limit_usd        = EXCLUDED.limit_usd,
                        alert_threshold  = EXCLUDED.alert_threshold,
                        enforce          = EXCLUDED.enforce
                    """,
                    (service_id, tag_key, tag_value, period,
                     limit_usd, alert_threshold, enforce),
                )
            conn.commit()
            logger.info(
                "Budget set: %s=%s %s $%.2f (enforce=%s)",
                tag_key, tag_value, period, limit_usd, enforce,
            )
            # Force cache refresh on next check
            self._last_refresh = 0.0
        finally:
            conn.close()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _maybe_refresh(self) -> None:
        now = time.monotonic()
        if now - self._last_refresh < self.cache_ttl:
            return
        try:
            self._refresh()
            self._last_refresh = now
        except Exception as exc:
            logger.warning("Budget cache refresh failed: %s", exc)

    def _refresh(self) -> None:
        """
        Load all budgets and current period spend from DB.
        Builds the cache keyed by (tag_key, tag_value, period).
        """
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = True
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        b.id,
                        b.service_id,
                        b.tag_key,
                        b.tag_value,
                        b.period,
                        b.limit_usd,
                        b.alert_threshold,
                        b.enforce,
                        COALESCE(SUM(c.cost_usd), 0) AS spent
                    FROM budgets b
                    LEFT JOIN llm_calls c
                        ON c.tags ->> b.tag_key = b.tag_value
                        AND c.time >= CASE b.period
                            WHEN 'daily'   THEN date_trunc('day',   NOW())
                            WHEN 'weekly'  THEN date_trunc('week',  NOW())
                            WHEN 'monthly' THEN date_trunc('month', NOW())
                            ELSE date_trunc('month', NOW())
                        END
                    GROUP BY b.id, b.service_id, b.tag_key, b.tag_value,
                             b.period, b.limit_usd, b.alert_threshold, b.enforce
                """)
                rows = cur.fetchall()

            new_cache = {}
            for row in rows:
                key = f"{row['tag_key']}:{row['tag_value']}:{row['period']}"
                new_cache[key] = dict(row)

            with self._lock:
                self._cache = new_cache

            logger.debug("Budget cache refreshed — %d budgets loaded", len(new_cache))

        finally:
            conn.close()
