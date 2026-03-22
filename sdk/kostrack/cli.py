"""
kostrack CLI — operator tooling for cost governance without writing Python.

Usage:
    kostrack status
    kostrack spend [--project PROJECT] [--days N] [--provider PROVIDER]
    kostrack budgets
    kostrack budget set <tag_key> <tag_value> <period> <limit_usd> [--alert 0.8] [--enforce]
    kostrack budget delete <tag_key> <tag_value> <period>
    kostrack pricing
    kostrack pricing sync
    kostrack pricing add <provider> <model> <input_rate> <output_rate> [--cache-read RATE]
    kostrack models
    kostrack traces [--project PROJECT] [--limit N]
    kostrack health

All commands require either:
    --dsn postgresql://user:pass@host/db
    KOSTRACK_DSN env var

Options:
    --json     Output raw JSON (default: pretty table)
    --service  service_id filter (default: all)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

# ─── output helpers ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if _supports_color() else text


def _header(text: str) -> None:
    print(f"\n{_c(text, BOLD + CYAN)}\n{'─' * len(text)}")


def _ok(text: str) -> None:
    print(f"  {_c('✓', GREEN)}  {text}")


def _warn(text: str) -> None:
    print(f"  {_c('!', YELLOW)}  {text}")


def _err(text: str) -> None:
    print(f"  {_c('✗', RED)}  {text}")


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct * width)
    color = RED if pct >= 1.0 else YELLOW if pct >= 0.8 else GREEN
    bar = "█" * filled + "░" * (width - filled)
    return _c(bar, color) if _supports_color() else bar


def _table(rows: list[dict], columns: list[tuple[str, str, int]]) -> None:
    """Print a simple aligned table. columns = [(key, header, width), ...]"""
    headers = [h.ljust(w) for _, h, w in columns]
    print("  " + "  ".join(_c(h, BOLD) for h in headers))
    print("  " + "  ".join("─" * w for _, _, w in columns))
    for row in rows:
        cells = []
        for key, _, width in columns:
            val = str(row.get(key, ""))
            cells.append(val[:width].ljust(width))
        print("  " + "  ".join(cells))


# ─── DB helpers ──────────────────────────────────────────────────────────────

def _connect(dsn: str):
    try:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        return conn
    except ImportError:
        _err("psycopg2 not installed. Run: pip install kostrack")
        sys.exit(1)
    except Exception as exc:
        _err(f"Cannot connect to TimescaleDB: {exc}")
        sys.exit(1)


def _query(dsn: str, sql: str, params=()) -> list[dict]:
    import psycopg2.extras
    conn = _connect(dsn)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _execute(dsn: str, sql: str, params=()) -> None:
    conn = _connect(dsn)
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.close()


# ─── commands ────────────────────────────────────────────────────────────────

def cmd_health(args) -> None:
    rows = _query(args.dsn, "SELECT * FROM writer_health ORDER BY updated_at DESC")
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return

    _header("Writer health")
    if not rows:
        _warn("No writer_health entries — has any service connected yet?")
        return

    for row in rows:
        status = _c("CONNECTED", GREEN) if row["timescale_ok"] else _c("DEGRADED", RED)
        print(f"\n  Service: {_c(row['service_id'], BOLD)}  [{status}]")
        print(f"    Written to TimescaleDB : {row['written_timescale']:,}")
        print(f"    SQLite backlog         : {row['sqlite_backlog']:,}")
        print(f"    Failed writes          : {row['failed']:,}")
        if row["last_flush"]:
            print(f"    Last flush             : {row['last_flush']}")
        print(f"    Updated                : {row['updated_at']}")


def cmd_status(args) -> None:
    rows = _query(args.dsn, """
        SELECT
            provider,
            COUNT(*)                    AS calls,
            SUM(cost_usd)               AS cost_usd,
            SUM(input_tokens)           AS input_tokens,
            SUM(output_tokens)          AS output_tokens,
            ROUND(AVG(latency_ms))      AS avg_latency_ms
        FROM llm_calls
        WHERE time >= NOW() - INTERVAL '24 hours'
        GROUP BY provider
        ORDER BY cost_usd DESC
    """)

    total_cost = _query(args.dsn, """
        SELECT SUM(cost_usd) AS total FROM llm_calls
        WHERE time >= NOW() - INTERVAL '24 hours'
    """)
    total = float(total_cost[0]["total"] or 0) if total_cost else 0.0

    if args.json:
        print(json.dumps({"total_24h_usd": total, "by_provider": rows}, indent=2, default=str))
        return

    _header("Last 24 hours")
    print(f"\n  Total spend : {_c(f'${total:.4f}', BOLD + GREEN)}\n")
    if not rows:
        _warn("No calls recorded in the last 24 hours.")
        return

    _table(rows, [
        ("provider",       "Provider",   12),
        ("calls",          "Calls",       8),
        ("cost_usd",       "Cost (USD)", 12),
        ("input_tokens",   "In tokens",  12),
        ("output_tokens",  "Out tokens", 12),
        ("avg_latency_ms", "Avg ms",      8),
    ])


def cmd_spend(args) -> None:
    days  = args.days or 30
    where = ["time >= NOW() - INTERVAL %s"]
    params: list[Any] = [f"{days} days"]

    if args.project:
        where.append("tags->>'project' = %s")
        params.append(args.project)
    if args.provider:
        where.append("provider = %s")
        params.append(args.provider)
    if args.service:
        where.append("service_id = %s")
        params.append(args.service)

    where_clause = " AND ".join(where)

    rows = _query(args.dsn, f"""
        SELECT
            time_bucket('1 day', time)  AS day,
            provider,
            model,
            tags->>'project'            AS project,
            COUNT(*)                    AS calls,
            SUM(cost_usd)               AS cost_usd,
            SUM(input_tokens)           AS input_tokens,
            SUM(output_tokens)          AS output_tokens
        FROM llm_calls
        WHERE {where_clause}
        GROUP BY day, provider, model, tags->>'project'
        ORDER BY day DESC, cost_usd DESC
    """, params)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return

    label = f"Spend — last {days} days"
    if args.project:
        label += f" · project={args.project}"
    if args.provider:
        label += f" · provider={args.provider}"
    _header(label)

    if not rows:
        _warn("No data for the given filters.")
        return

    _table(rows, [
        ("day",           "Day",        12),
        ("provider",      "Provider",   12),
        ("model",         "Model",      30),
        ("project",       "Project",    16),
        ("calls",         "Calls",       8),
        ("cost_usd",      "Cost (USD)", 12),
        ("input_tokens",  "In tokens",  12),
        ("output_tokens", "Out tokens", 12),
    ])

    total = sum(float(r["cost_usd"] or 0) for r in rows)
    print(f"\n  Total: {_c(f'${total:.4f}', BOLD + GREEN)}")


def cmd_budgets(args) -> None:
    # Pull budgets joined with current period spend
    rows = _query(args.dsn, """
        SELECT
            b.service_id,
            b.tag_key,
            b.tag_value,
            b.period,
            b.limit_usd,
            b.alert_threshold,
            b.enforce,
            COALESCE(
                (SELECT SUM(lc.cost_usd)
                 FROM llm_calls lc
                 WHERE lc.tags->>b.tag_key = b.tag_value
                   AND lc.time >= CASE b.period
                       WHEN 'daily'   THEN date_trunc('day',   NOW())
                       WHEN 'weekly'  THEN date_trunc('week',  NOW())
                       WHEN 'monthly' THEN date_trunc('month', NOW())
                       ELSE NOW() - INTERVAL '30 days'
                   END
                ), 0
            ) AS spent
        FROM budgets b
        ORDER BY b.tag_key, b.tag_value, b.period
    """)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return

    _header("Budgets")
    if not rows:
        _warn("No budgets configured. Use: kostrack budget set <key> <value> <period> <limit>")
        return

    print()
    for row in rows:
        spent = float(row["spent"] or 0)
        limit = float(row["limit_usd"])
        pct   = spent / limit if limit > 0 else 0
        threshold = float(row["alert_threshold"])
        enforce = row["enforce"]

        status = ""
        if pct >= 1.0:
            status = _c(" EXCEEDED", RED + BOLD) if enforce else _c(" OVER", RED)
        elif pct >= threshold:
            status = _c(" ALERT", YELLOW)

        tag_str   = f"{row['tag_key']}={_c(row['tag_value'], BOLD)}"
        period    = row["period"].upper()
        enforce_s = _c("hard", RED) if enforce else _c("soft", DIM)
        print(f"  {tag_str}  [{period}]{status}  ({enforce_s} limit)")
        print(f"    ${spent:.4f} / ${limit:.2f}  {_bar(pct)}  {pct*100:.1f}%")
        print()


def cmd_budget_set(args) -> None:
    from kostrack.budget import BudgetEnforcer

    enforcer = BudgetEnforcer(dsn=args.dsn)
    enforcer.set_budget(
        tag_key=args.tag_key,
        tag_value=args.tag_value,
        period=args.period,
        limit_usd=args.limit_usd,
        alert_threshold=args.alert,
        enforce=args.enforce,
        service_id=args.service or "default",
    )
    _ok(f"Budget set: {args.tag_key}={args.tag_value}  {args.period}  ${args.limit_usd:.2f}"
        f"  alert={int(args.alert*100)}%  enforce={args.enforce}")


def cmd_budget_delete(args) -> None:
    _execute(args.dsn, """
        DELETE FROM budgets
        WHERE tag_key = %s AND tag_value = %s AND period = %s
    """, (args.tag_key, args.tag_value, args.period))
    _ok(f"Deleted budget: {args.tag_key}={args.tag_value} [{args.period}]")


def cmd_pricing(args) -> None:
    rows = _query(args.dsn, """
        SELECT provider, model, pricing_model,
               input_rate, output_rate, cache_read_rate, effective_from
        FROM pricing_current
        ORDER BY provider, model, pricing_model
    """)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return

    _header("Active pricing  (USD per token)")
    print()
    current_provider = None
    for row in rows:
        if row["provider"] != current_provider:
            current_provider = row["provider"]
            print(f"  {_c(current_provider.upper(), BOLD + BLUE)}")

        ir = float(row["input_rate"] or 0) * 1_000_000
        or_ = float(row["output_rate"] or 0) * 1_000_000
        cr = float(row["cache_read_rate"] or 0) * 1_000_000 if row["cache_read_rate"] else None
        cache_s = f"  cache_read=${cr:.3f}/M" if cr else ""
        pm = f"[{row['pricing_model']}]"
        print(f"    {row['model']:<40} {pm:<12}  "
              f"in=${ir:.3f}/M  out=${or_:.3f}/M{cache_s}")
    print()


def cmd_pricing_sync(args) -> None:
    from kostrack.sync.pricing_sync import PricingSync

    sync = PricingSync(dsn=args.dsn)
    result = sync.run()
    _ok(f"Pricing sync complete — inserted={result['inserted']}  already_current={result['already_current']}")


def cmd_pricing_add(args) -> None:
    from kostrack.sync.pricing_sync import PricingSync

    sync = PricingSync(dsn=args.dsn)
    sync.upsert(
        provider=args.provider,
        model=args.model,
        pricing_model="per_token",
        input_rate=args.input_rate / 1_000_000,
        output_rate=args.output_rate / 1_000_000,
        cache_read_rate=args.cache_read / 1_000_000 if args.cache_read else None,
        notes=f"Added via CLI {datetime.now(timezone.utc).date()}",
    )
    _ok(f"Added pricing: {args.provider}/{args.model}  "
        f"in=${args.input_rate}/M  out=${args.output_rate}/M")


def cmd_models(args) -> None:
    rows = _query(args.dsn, """
        SELECT provider, model,
               COUNT(*)       AS calls,
               SUM(cost_usd)  AS total_cost,
               MIN(time)      AS first_seen,
               MAX(time)      AS last_seen
        FROM llm_calls
        GROUP BY provider, model
        ORDER BY total_cost DESC
    """)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return

    _header("Models seen (all time)")
    if not rows:
        _warn("No calls recorded yet.")
        return

    _table(rows, [
        ("provider",   "Provider",  12),
        ("model",      "Model",     36),
        ("calls",      "Calls",      8),
        ("total_cost", "Cost (USD)", 12),
        ("first_seen", "First seen", 24),
        ("last_seen",  "Last seen",  24),
    ])


def cmd_traces(args) -> None:
    limit = args.limit or 20
    where = []
    params: list[Any] = []

    if args.project:
        where.append("tags->>'project' = %s")
        params.append(args.project)
    if args.service:
        where.append("service_id = %s")
        params.append(args.service)

    where_clause = f"WHERE trace_id IS NOT NULL{' AND ' + ' AND '.join(where) if where else ''}"

    rows = _query(args.dsn, f"""
        SELECT
            trace_id,
            tags->>'project'      AS project,
            tags->>'feature'      AS feature,
            MIN(time)             AS started_at,
            ROUND(EXTRACT(EPOCH FROM (MAX(time) - MIN(time))) * 1000) AS duration_ms,
            COUNT(*)              AS calls,
            COUNT(DISTINCT model) AS models,
            SUM(cost_usd)         AS cost_usd
        FROM llm_calls
        {where_clause}
        GROUP BY trace_id, tags->>'project', tags->>'feature'
        ORDER BY started_at DESC
        LIMIT %s
    """, params + [limit])

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return

    label = f"Recent traces (last {limit})"
    if args.project:
        label += f" · project={args.project}"
    _header(label)

    if not rows:
        _warn("No traced workflows found.")
        return

    _table(rows, [
        ("started_at",   "Started",      24),
        ("project",      "Project",      16),
        ("feature",      "Feature",      20),
        ("calls",        "Calls",         6),
        ("models",       "Models",        7),
        ("duration_ms",  "Duration ms",  12),
        ("cost_usd",     "Cost (USD)",   12),
    ])


# ─── argument parser ─────────────────────────────────────────────────────────

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dsn",
        default=os.environ.get("KOSTRACK_DSN"),
        help="PostgreSQL DSN (default: $KOSTRACK_DSN)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--service", default=None, help="Filter by service_id")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="kostrack",
        description="kostrack — AI API cost governance CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  kostrack status
  kostrack spend --project openmanagr --days 7
  kostrack budgets
  kostrack budget set project openmanagr monthly 50.00 --enforce
  kostrack pricing sync
  kostrack pricing add deepseek deepseek-v4 0.30 1.20
  kostrack traces --project openmanagr --limit 50
  kostrack health
        """,
    )
    _add_common(root)
    sub = root.add_subparsers(dest="command", metavar="COMMAND")

    # status
    p = sub.add_parser("status", help="24-hour spend summary by provider")
    _add_common(p)

    # spend
    p = sub.add_parser("spend", help="Spend breakdown by day/model/project")
    _add_common(p)
    p.add_argument("--project",  default=None, help="Filter by project tag")
    p.add_argument("--provider", default=None, help="Filter by provider (anthropic/openai/gemini/deepseek)")
    p.add_argument("--days",     type=int, default=30, help="Lookback window in days (default: 30)")

    # budgets
    p = sub.add_parser("budgets", help="Show all budgets with current spend")
    _add_common(p)

    # budget set / delete
    budget = sub.add_parser("budget", help="Manage budgets")
    _add_common(budget)
    bsub = budget.add_subparsers(dest="budget_command", metavar="ACTION")

    bs = bsub.add_parser("set", help="Set or update a budget")
    _add_common(bs)
    bs.add_argument("tag_key",   help="Tag dimension, e.g. project")
    bs.add_argument("tag_value", help="Tag value, e.g. openmanagr")
    bs.add_argument("period",    choices=["daily", "weekly", "monthly"])
    bs.add_argument("limit_usd", type=float, help="Budget cap in USD")
    bs.add_argument("--alert",   type=float, default=0.80, help="Alert threshold 0–1 (default: 0.80)")
    bs.add_argument("--enforce", action="store_true", help="Hard limit — raise on breach")

    bd = bsub.add_parser("delete", help="Delete a budget")
    _add_common(bd)
    bd.add_argument("tag_key")
    bd.add_argument("tag_value")
    bd.add_argument("period", choices=["daily", "weekly", "monthly"])

    # pricing
    p = sub.add_parser("pricing", help="Manage pricing table")
    _add_common(p)
    psub = p.add_subparsers(dest="pricing_command", metavar="ACTION")

    psub.add_parser("sync", help="Sync bundled pricing to TimescaleDB")

    pa = psub.add_parser("add", help="Add or update a model's pricing")
    _add_common(pa)
    pa.add_argument("provider")
    pa.add_argument("model")
    pa.add_argument("input_rate",  type=float, help="USD per million input tokens")
    pa.add_argument("output_rate", type=float, help="USD per million output tokens")
    pa.add_argument("--cache-read", type=float, default=None,
                    dest="cache_read", help="USD per million cache-read tokens")

    # models
    p = sub.add_parser("models", help="All models seen, with lifetime cost")
    _add_common(p)

    # traces
    p = sub.add_parser("traces", help="Recent agentic workflow traces")
    _add_common(p)
    p.add_argument("--project", default=None)
    p.add_argument("--limit",   type=int, default=20)

    # health
    p = sub.add_parser("health", help="Writer health for all connected services")
    _add_common(p)

    return root


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if not args.dsn:
        _err("No DSN provided. Use --dsn or set KOSTRACK_DSN env var.")
        sys.exit(1)

    dispatch = {
        "status":  cmd_status,
        "spend":   cmd_spend,
        "budgets": cmd_budgets,
        "models":  cmd_models,
        "traces":  cmd_traces,
        "health":  cmd_health,
        "budget": lambda a: (
            cmd_budget_set(a) if a.budget_command == "set"
            else cmd_budget_delete(a) if a.budget_command == "delete"
            else parser.parse_args(["budget", "--help"])
        ),
        "pricing": lambda a: (
            cmd_pricing_sync(a) if a.pricing_command == "sync"
            else cmd_pricing_add(a) if a.pricing_command == "add"
            else cmd_pricing(a)
        ),
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print()
        sys.exit(0)
    except Exception as exc:
        _err(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()