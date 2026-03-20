#!/usr/bin/env python3
"""
TokenLedger — Live Integration Test

Runs against a live TimescaleDB instance and real LLM API keys.
Verifies the full pipeline: API call → SDK → TimescaleDB → query back.

Prerequisites:
    docker compose up -d          (from kostrack/ root)
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...      (optional)
    export KOSTRACK_DSN=postgresql://kostrack:changeme@localhost/kostrack

Run:
    cd sdk
    pip install -e .
    python tests/integration_test.py

Flags:
    --skip-anthropic     Skip Anthropic calls
    --skip-openai        Skip OpenAI calls
    --skip-gemini        Skip Gemini calls (skipped by default if no key)
    --no-cleanup         Leave test rows in DB after run
"""

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"
INFO = "\033[94m·\033[0m"

results = {"passed": 0, "failed": 0, "skipped": 0}


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")
    results["passed"] += 1


def fail(msg: str, exc: Exception | None = None) -> None:
    print(f"  {FAIL} {msg}")
    if exc:
        print(f"      {type(exc).__name__}: {exc}")
    results["failed"] += 1


def skip(msg: str) -> None:
    print(f"  {SKIP} {msg}")
    results["skipped"] += 1


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def wait_for_row(conn, tag_key: str, tag_value: str, timeout: float = 15.0) -> dict | None:
    """Poll DB until a row with the given tag appears."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM llm_calls
                WHERE tags->>%s = %s
                ORDER BY time DESC LIMIT 1
                """,
                (tag_key, tag_value),
            )
            row = cur.fetchone()
        if row:
            return dict(row)
        time.sleep(0.5)
    return None


def cleanup_test_rows(conn, run_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM llm_calls WHERE tags->>'integration_run' = %s", (run_id,))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Test sections
# ─────────────────────────────────────────────────────────────────────────────

def test_db_connection(dsn: str) -> psycopg2.extensions.connection | None:
    section("1. Database Connection")
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        ok("Connected to TimescaleDB")
    except Exception as e:
        fail("Cannot connect to TimescaleDB", e)
        print("\n  Is the stack running?  →  docker compose up -d")
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            row = cur.fetchone()
            assert row, "TimescaleDB extension not found"
        ok(f"TimescaleDB extension present (v{row[0]})")
    except Exception as e:
        fail("TimescaleDB extension check failed", e)
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM llm_calls")
            count = cur.fetchone()[0]
        ok(f"llm_calls table exists ({count} existing rows)")
    except Exception as e:
        fail("llm_calls table missing — did init-db.sql run?", e)
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pricing")
            pcount = cur.fetchone()[0]
        ok(f"pricing table exists ({pcount} pricing entries seeded)")
        if pcount == 0:
            fail("No pricing entries — init-db.sql may not have run correctly")
    except Exception as e:
        fail("pricing table check failed", e)

    return conn


def test_sdk_configure(dsn: str) -> bool:
    section("2. SDK Configuration")
    try:
        import kostrack
        kostrack.configure(
            dsn=dsn,
            service_id="integration-test",
            flush_interval=1.0,
            log_level="DEBUG",
        )
        ok("kostrack.configure() succeeded")
    except Exception as e:
        fail("configure() failed", e)
        return False

    try:
        h = kostrack.health()
        assert h["status"] == "ok"
        ok(f"health() → {h['status']} | timescale={h['timescale_available']}")
    except Exception as e:
        fail("health() check failed", e)
        return False

    return True


def test_anthropic(conn, run_id: str, skip_flag: bool) -> None:
    section("3. Anthropic — End-to-End")

    if skip_flag:
        skip("Skipped via --skip-anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        skip("ANTHROPIC_API_KEY not set")
        return

    try:
        from kostrack import Anthropic
        client = Anthropic(
            tags={
                "project": "integration-test",
                "feature": "anthropic-basic",
                "integration_run": run_id,
            }
        )
        ok("Anthropic client instantiated")
    except Exception as e:
        fail("Anthropic client init failed", e)
        return

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",   # cheapest model for tests
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: integration test ok"}],
        )
        ok(f"API call succeeded — response: {response.content[0].text[:50]!r}")
    except Exception as e:
        fail("Anthropic API call failed", e)
        return

    row = wait_for_row(conn, "integration_run", run_id)
    if row:
        ok(f"Row written to DB — model={row['model']} cost=${row['cost_usd']:.8f}")
        assert row["provider"] == "anthropic", f"Wrong provider: {row['provider']}"
        assert row["input_tokens"] > 0, "No input tokens recorded"
        assert row["output_tokens"] > 0, "No output tokens recorded"
        assert row["cost_usd"] > 0, "Zero cost recorded"
        assert row["latency_ms"] > 0, "Zero latency recorded"
        ok(f"Row validated — input={row['input_tokens']} output={row['output_tokens']} latency={row['latency_ms']}ms")
    else:
        fail("Row not found in DB after 15s — writer may be buffering or failing")


def test_anthropic_with_trace(conn, run_id: str, skip_flag: bool) -> None:
    section("4. Anthropic — Agentic Trace")

    if skip_flag:
        skip("Skipped via --skip-anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        skip("ANTHROPIC_API_KEY not set")
        return

    try:
        import kostrack
        from kostrack import Anthropic

        trace_run_id = f"{run_id}-trace"
        client = Anthropic(
            tags={
                "project": "integration-test",
                "feature": "agentic-trace",
                "integration_run": trace_run_id,
            }
        )

        with kostrack.trace(tags={"project": "integration-test"}) as t:
            with kostrack.span("step-1", parent=t):
                client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "Say: step one"}],
                )
            with kostrack.span("step-2", parent=t):
                client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "Say: step two"}],
                )

        ok(f"Traced workflow complete — calls={t.call_count} total_cost=${t.total_cost_usd:.8f}")
        assert t.call_count == 2, f"Expected 2 calls, got {t.call_count}"

        # Wait and verify both rows share a trace_id
        time.sleep(3)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT trace_id, COUNT(*) as call_count, SUM(cost_usd) as total_cost
                FROM llm_calls
                WHERE tags->>'integration_run' = %s
                GROUP BY trace_id
                """,
                (trace_run_id,),
            )
            rows = cur.fetchall()

        if rows:
            r = dict(rows[0])
            ok(f"Trace verified in DB — trace_id={str(r['trace_id'])[:8]}... calls={r['call_count']} total=${float(r['total_cost']):.8f}")
            assert r["call_count"] == 2, f"Expected 2 rows, got {r['call_count']}"
            ok("Trace rollup validated ✓")
        else:
            fail(f"No traced rows found in DB for run_id={trace_run_id}")

    except Exception as e:
        fail("Agentic trace test failed", e)


def test_openai(conn, run_id: str, skip_flag: bool) -> None:
    section("5. OpenAI — End-to-End")

    if skip_flag:
        skip("Skipped via --skip-openai")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        skip("OPENAI_API_KEY not set")
        return

    try:
        from kostrack import OpenAI
        oai_run_id = f"{run_id}-openai"
        client = OpenAI(
            tags={
                "project": "integration-test",
                "feature": "openai-basic",
                "integration_run": oai_run_id,
            }
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # cheapest for tests
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with exactly: integration test ok"}],
        )
        ok(f"OpenAI call succeeded — response: {response.choices[0].message.content[:50]!r}")

        row = wait_for_row(conn, "integration_run", oai_run_id)
        if row:
            assert row["provider"] == "openai"
            assert row["cost_usd"] > 0
            ok(f"Row validated — model={row['model']} cost=${row['cost_usd']:.8f}")
        else:
            fail("OpenAI row not found in DB after 15s")

    except Exception as e:
        fail("OpenAI test failed", e)


def test_gemini(conn, run_id: str, skip_flag: bool) -> None:
    section("6. Gemini — End-to-End")

    if skip_flag:
        skip("Skipped via --skip-gemini")
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        skip("GEMINI_API_KEY not set")
        return

    try:
        from kostrack import GenerativeModel
        gem_run_id = f"{run_id}-gemini"
        model = GenerativeModel(
            "gemini-2.0-flash-lite",  # cheapest for tests
            tags={
                "project": "integration-test",
                "feature": "gemini-basic",
                "integration_run": gem_run_id,
            },
        )
        response = model.generate_content("Reply with exactly: integration test ok")
        ok(f"Gemini call succeeded — response: {response.text[:50]!r}")

        row = wait_for_row(conn, "integration_run", gem_run_id)
        if row:
            assert row["provider"] == "gemini"
            assert row["cost_usd"] >= 0   # Gemini has a free tier
            ok(f"Row validated — model={row['model']} cost=${row['cost_usd']:.8f}")
        else:
            fail("Gemini row not found in DB after 15s")

    except Exception as e:
        fail("Gemini test failed", e)


def test_multi_provider_trace(conn, run_id: str, args: argparse.Namespace) -> None:
    section("7. Multi-Provider Agentic Trace")

    has_anthropic = os.environ.get("ANTHROPIC_API_KEY") and not args.skip_anthropic
    has_openai = os.environ.get("OPENAI_API_KEY") and not args.skip_openai

    if not (has_anthropic and has_openai):
        skip("Requires both ANTHROPIC_API_KEY and OPENAI_API_KEY")
        return

    try:
        import kostrack
        from kostrack import Anthropic, OpenAI

        mp_run_id = f"{run_id}-multiprovider"
        anthropic_client = Anthropic(tags={"integration_run": mp_run_id, "project": "integration-test"})
        openai_client = OpenAI(tags={"integration_run": mp_run_id, "project": "integration-test"})

        with kostrack.trace(tags={"feature": "multi-provider-workflow"}) as t:
            with kostrack.span("extract", parent=t):
                anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "Say: extract done"}],
                )
            with kostrack.span("classify", parent=t):
                openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "Say: classify done"}],
                )

        ok(f"Multi-provider trace — calls={t.call_count} cost=${t.total_cost_usd:.8f}")
        assert t.call_count == 2

        time.sleep(3)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT provider, trace_id, cost_usd FROM llm_calls
                WHERE tags->>'integration_run' = %s
                ORDER BY time
                """,
                (mp_run_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]

        if len(rows) == 2:
            trace_ids = {str(r["trace_id"]) for r in rows}
            providers = {r["provider"] for r in rows}
            assert len(trace_ids) == 1, f"Trace IDs differ: {trace_ids}"
            assert providers == {"anthropic", "openai"}
            ok(f"Multi-provider trace verified — providers={providers} shared trace_id={list(trace_ids)[0][:8]}...")
        else:
            fail(f"Expected 2 rows, found {len(rows)}")

    except Exception as e:
        fail("Multi-provider trace test failed", e)


def test_grafana_reachable() -> None:
    section("8. Grafana Dashboard")
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:3000/api/health", timeout=5)
        data = req.read().decode()
        if "ok" in data.lower():
            ok("Grafana is reachable at http://localhost:3000")
            ok("Login: admin / admin  (or your GRAFANA_PASSWORD)")
            ok("TokenLedger Overview dashboard pre-loaded")
        else:
            fail(f"Grafana health unexpected response: {data[:100]}")
    except Exception as e:
        fail("Grafana not reachable at localhost:3000", e)
        print("    Is docker compose running?  →  docker compose up -d")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TokenLedger live integration test")
    parser.add_argument("--skip-anthropic", action="store_true")
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true", default=True,
                        help="Skip Gemini (default: skipped unless GEMINI_API_KEY set)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Leave test rows in DB after run")
    args = parser.parse_args()

    # Auto-enable Gemini if key is present
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        args.skip_gemini = False

    dsn = os.environ.get(
        "KOSTRACK_DSN",
        "postgresql://kostrack:changeme@localhost/kostrack",
    )
    run_id = str(uuid.uuid4())[:8]

    print(f"\n{'═' * 60}")
    print(f"  TokenLedger Integration Test")
    print(f"  Run ID: {run_id}")
    print(f"  DSN:    {dsn}")
    print(f"{'═' * 60}")

    # 1. DB connection (must pass to continue)
    conn = test_db_connection(dsn)
    if not conn:
        print("\n  Cannot proceed without DB connection.\n")
        sys.exit(1)

    # 2. SDK configure (must pass to continue)
    if not test_sdk_configure(dsn):
        print("\n  Cannot proceed without SDK configured.\n")
        sys.exit(1)

    # 3–7. Provider tests
    test_anthropic(conn, run_id, args.skip_anthropic)
    test_anthropic_with_trace(conn, run_id, args.skip_anthropic)
    test_openai(conn, run_id, args.skip_openai)
    test_gemini(conn, run_id, args.skip_gemini)
    test_multi_provider_trace(conn, run_id, args)
    test_grafana_reachable()

    # Cleanup
    if not args.no_cleanup:
        try:
            cleanup_test_rows(conn, run_id)
            # Also clean trace and multi-provider rows
            for suffix in ["-trace", "-openai", "-gemini", "-multiprovider"]:
                cleanup_test_rows(conn, f"{run_id}{suffix}")
            print(f"\n  {INFO} Test rows cleaned up (use --no-cleanup to keep them)")
        except Exception as e:
            print(f"\n  {INFO} Cleanup failed: {e}")

    conn.close()

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  Results: {results['passed']} passed  {results['failed']} failed  {results['skipped']} skipped")
    print(f"{'═' * 60}\n")

    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
