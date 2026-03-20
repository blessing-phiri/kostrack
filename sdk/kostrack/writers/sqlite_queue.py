"""
TokenLedger — SQLite Fallback Queue

Durable local buffer for CallRecords when TimescaleDB is unavailable.
Writes survive process restarts. Flushed to TimescaleDB automatically
when connectivity is restored.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("kostrack.sqlite")

DEFAULT_PATH = Path.home() / ".kostrack" / "buffer.db"


class SQLiteQueue:
    """
    Thread-safe persistent queue backed by SQLite.

    Records are written here when TimescaleDB is unreachable,
    then flushed back in FIFO order once connectivity returns.
    """

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=10,
        )
        conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent writes
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS buffer (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload     TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL,
                    attempts    INTEGER NOT NULL DEFAULT 0
                )
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, row: dict[str, Any]) -> None:
        """Serialize and persist one CallRecord row."""
        payload = _serialize(row)
        with self._lock:
            self._conn.execute(
                "INSERT INTO buffer (payload, created_at) VALUES (?, ?)",
                (payload, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def push_batch(self, rows: list[dict[str, Any]]) -> None:
        """Persist a batch of CallRecord rows atomically."""
        now = datetime.now(timezone.utc).isoformat()
        records = [(_serialize(r), now) for r in rows]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO buffer (payload, created_at) VALUES (?, ?)",
                records,
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read / flush
    # ------------------------------------------------------------------

    def pop_batch(self, size: int = 100) -> list[tuple[int, dict[str, Any]]]:
        """
        Return up to `size` records as (id, row) pairs.
        Records are NOT deleted until ack() is called.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, payload FROM buffer ORDER BY id ASC LIMIT ?",
                (size,),
            )
            rows = cursor.fetchall()

        return [(row_id, _deserialize(payload)) for row_id, payload in rows]

    def ack(self, ids: list[int]) -> None:
        """Delete successfully flushed records."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            self._conn.execute(
                f"DELETE FROM buffer WHERE id IN ({placeholders})", ids
            )
            self._conn.commit()

    def increment_attempts(self, ids: list[int]) -> None:
        """Track retry attempts for observability."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            self._conn.execute(
                f"UPDATE buffer SET attempts = attempts + 1 WHERE id IN ({placeholders})",
                ids,
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def size(self) -> int:
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM buffer")
            return cursor.fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------

def _serialize(row: dict[str, Any]) -> str:
    """JSON-serialize a CallRecord row, handling non-serializable types."""

    def default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Not serializable: {type(obj)}")

    return json.dumps(row, default=default)


def _deserialize(payload: str) -> dict[str, Any]:
    """Deserialize and restore datetime fields."""
    row = json.loads(payload)

    # Restore datetime
    if isinstance(row.get("time"), str):
        row["time"] = datetime.fromisoformat(row["time"])

    return row