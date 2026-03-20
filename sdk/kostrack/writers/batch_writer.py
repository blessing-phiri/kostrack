"""
kostrack — Async Batch Writer

Non-blocking write path. The application thread puts a CallRecord
onto the in-memory queue and returns immediately. A background
daemon thread drains the queue in batches, writing to TimescaleDB.

If TimescaleDB is unavailable the batch goes to the local SQLite
buffer. The writer periodically retries TimescaleDB and flushes
the SQLite backlog when connectivity returns.

Health state is observable via .health() — surfaced to monitoring
and to the Grafana dashboard via the writer_health table.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import psycopg2
import psycopg2.extras

from kostrack.writers.sqlite_queue import SQLiteQueue

logger = logging.getLogger("kostrack.writer")

# How often (seconds) to attempt TimescaleDB reconnect after failure
RETRY_INTERVAL = 30

# How often (seconds) to flush the SQLite backlog after reconnect
BACKLOG_FLUSH_INTERVAL = 60


class AsyncBatchWriter:
    """
    Background batch writer with SQLite fallback and health reporting.

    Usage:
        writer = AsyncBatchWriter(dsn="postgresql://...")
        writer.write(call_record.to_row())   # never blocks
        writer.health()                       # observable state
        writer.stop()                         # graceful shutdown
    """

    def __init__(
        self,
        dsn: str,
        flush_interval: float = 5.0,
        max_batch_size: int = 100,
        sqlite_path: Path | None = None,
        service_id: str = "default",
        fail_open: bool = True,
    ) -> None:
        """
        Args:
            dsn:             PostgreSQL DSN for TimescaleDB.
            flush_interval:  Seconds between queue drain cycles.
            max_batch_size:  Max rows per TimescaleDB insert.
            sqlite_path:     Override default SQLite buffer location.
            service_id:      Written to writer_health table.
            fail_open:       If True, silently drop on all failures.
                             If False, raise after SQLite also fails.
        """
        self.dsn = dsn
        self.flush_interval = flush_interval
        self.max_batch_size = max_batch_size
        self.service_id = service_id
        self.fail_open = fail_open

        self._queue: Queue[dict[str, Any]] = Queue()
        self._sqlite = SQLiteQueue(sqlite_path) if sqlite_path else SQLiteQueue()

        self._tsdb_ok = False
        self._conn: psycopg2.extensions.connection | None = None
        self._lock = threading.Lock()

        self._stats = {
            "queued": 0,
            "written_timescale": 0,
            "written_sqlite": 0,
            "failed": 0,
            "last_flush": None,
        }

        self._last_retry = 0.0
        self._last_backlog_flush = 0.0
        self._stop_event = threading.Event()

        # Try initial connection
        self._try_connect()

        # Start background worker
        self._worker = threading.Thread(
            target=self._run, name="kostrack-writer", daemon=True
        )
        self._worker.start()
        logger.info(
            "AsyncBatchWriter started — TimescaleDB: %s",
            "connected" if self._tsdb_ok else "unavailable (SQLite fallback active)",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, row: dict[str, Any]) -> None:
        """
        Enqueue a CallRecord row for async writing.
        Never blocks the calling thread.
        """
        self._queue.put(row)
        self._stats["queued"] += 1

    def health(self) -> dict[str, Any]:
        """Return current writer health state."""
        with self._lock:
            return {
                "timescale_available": self._tsdb_ok,
                "sqlite_backlog": self._sqlite.size(),
                "queue_depth": self._queue.qsize(),
                **self._stats,
            }

    def stop(self, timeout: float = 10.0) -> None:
        """
        Graceful shutdown — flush remaining queue before stopping.
        Blocks until flushed or timeout exceeded.
        """
        logger.info("AsyncBatchWriter stopping — draining queue...")
        self._stop_event.set()
        self._worker.join(timeout=timeout)
        # Final drain attempt
        self._drain_queue()
        self._sqlite.close()
        if self._conn and not self._conn.closed:
            self._conn.close()
        logger.info("AsyncBatchWriter stopped.")

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._drain_queue()
            self._maybe_retry_tsdb()
            self._maybe_flush_backlog()
            self._update_health_table()
            time.sleep(self.flush_interval)

        # Final drain on shutdown
        self._drain_queue()

    def _drain_queue(self) -> None:
        """Collect all queued rows and write them."""
        batch: list[dict[str, Any]] = []

        # Drain up to max_batch_size from the in-memory queue
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break

        if not batch:
            return

        if self._tsdb_ok:
            success = self._write_to_timescale(batch)
            if not success:
                with self._lock:
                    self._tsdb_ok = False
                logger.warning(
                    "TimescaleDB write failed — switching to SQLite fallback"
                )
                self._write_to_sqlite(batch)
        else:
            self._write_to_sqlite(batch)

        self._stats["last_flush"] = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # TimescaleDB
    # ------------------------------------------------------------------

    def _try_connect(self) -> bool:
        try:
            if self._conn and not self._conn.closed:
                self._conn.close()
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = False
            with self._lock:
                self._tsdb_ok = True
            logger.info("Connected to TimescaleDB")
            return True
        except Exception as exc:
            logger.warning("TimescaleDB connection failed: %s", exc)
            with self._lock:
                self._tsdb_ok = False
            return False

    def _write_to_timescale(self, batch: list[dict[str, Any]]) -> bool:
        try:
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO llm_calls (
                        time, service_id, provider, model, pricing_model,
                        input_tokens, output_tokens, cached_tokens, cost_usd,
                        token_breakdown, latency_ms,
                        trace_id, span_id, parent_span_id,
                        tags, metadata
                    ) VALUES %s
                    """,
                    [_row_to_tuple(r) for r in batch],
                    template="""(
                        %(time)s, %(service_id)s, %(provider)s, %(model)s, %(pricing_model)s,
                        %(input_tokens)s, %(output_tokens)s, %(cached_tokens)s, %(cost_usd)s,
                        %(token_breakdown)s::jsonb, %(latency_ms)s,
                        %(trace_id)s::uuid, %(span_id)s::uuid, %(parent_span_id)s::uuid,
                        %(tags)s::jsonb, %(metadata)s::jsonb
                    )""",
                    page_size=self.max_batch_size,
                )
            self._conn.commit()
            self._stats["written_timescale"] += len(batch)
            logger.debug("Wrote %d rows to TimescaleDB", len(batch))
            return True

        except Exception as exc:
            logger.error("TimescaleDB write error: %s", exc)
            try:
                self._conn.rollback()
            except Exception:
                pass
            self._stats["failed"] += len(batch)
            return False

    # ------------------------------------------------------------------
    # SQLite fallback
    # ------------------------------------------------------------------

    def _write_to_sqlite(self, batch: list[dict[str, Any]]) -> None:
        try:
            self._sqlite.push_batch(batch)
            self._stats["written_sqlite"] += len(batch)
            logger.debug("Wrote %d rows to SQLite buffer", len(batch))
        except Exception as exc:
            self._stats["failed"] += len(batch)
            logger.error("SQLite write error: %s", exc)
            if not self.fail_open:
                raise

    # ------------------------------------------------------------------
    # Retry + backlog flush
    # ------------------------------------------------------------------

    def _maybe_retry_tsdb(self) -> None:
        if self._tsdb_ok:
            return
        now = time.monotonic()
        if now - self._last_retry < RETRY_INTERVAL:
            return
        self._last_retry = now
        logger.info("Retrying TimescaleDB connection...")
        self._try_connect()

    def _maybe_flush_backlog(self) -> None:
        if not self._tsdb_ok:
            return
        if self._sqlite.size() == 0:
            return
        now = time.monotonic()
        if now - self._last_backlog_flush < BACKLOG_FLUSH_INTERVAL:
            return
        self._last_backlog_flush = now
        self._flush_sqlite_backlog()

    def _flush_sqlite_backlog(self) -> None:
        logger.info("Flushing SQLite backlog (%d rows)...", self._sqlite.size())
        flushed = 0
        while True:
            batch = self._sqlite.pop_batch(size=self.max_batch_size)
            if not batch:
                break
            ids = [row_id for row_id, _ in batch]
            rows = [row for _, row in batch]
            success = self._write_to_timescale(rows)
            if success:
                self._sqlite.ack(ids)
                flushed += len(rows)
            else:
                self._sqlite.increment_attempts(ids)
                logger.warning("Backlog flush failed — will retry later")
                with self._lock:
                    self._tsdb_ok = False
                break
        if flushed:
            logger.info("Flushed %d rows from SQLite backlog to TimescaleDB", flushed)

    # ------------------------------------------------------------------
    # Health table
    # ------------------------------------------------------------------

    def _update_health_table(self) -> None:
        if not self._tsdb_ok or not self._conn:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO writer_health (
                        service_id, timescale_ok, sqlite_backlog,
                        queued, written_timescale, written_sqlite, failed,
                        last_flush, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (service_id) DO UPDATE SET
                        timescale_ok       = EXCLUDED.timescale_ok,
                        sqlite_backlog     = EXCLUDED.sqlite_backlog,
                        queued             = EXCLUDED.queued,
                        written_timescale  = EXCLUDED.written_timescale,
                        written_sqlite     = EXCLUDED.written_sqlite,
                        failed             = EXCLUDED.failed,
                        last_flush         = EXCLUDED.last_flush,
                        updated_at         = EXCLUDED.updated_at
                    """,
                    (
                        self.service_id,
                        self._tsdb_ok,
                        self._sqlite.size(),
                        self._stats["queued"],
                        self._stats["written_timescale"],
                        self._stats["written_sqlite"],
                        self._stats["failed"],
                        self._stats["last_flush"],
                    ),
                )
            self._conn.commit()
        except Exception as exc:
            logger.debug("Could not update writer_health: %s", exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _row_to_tuple(row: dict[str, Any]) -> dict[str, Any]:
    """Prepare a row dict for psycopg2 execute_values."""
    r = row.copy()
    # Serialize JSONB fields
    for key in ("token_breakdown", "tags", "metadata"):
        if isinstance(r.get(key), dict):
            r[key] = json.dumps(r[key])
    # None UUIDs stay None — cast to ::uuid handles them
    return r