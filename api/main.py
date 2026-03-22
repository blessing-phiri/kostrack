"""
kostrack Platform API

A thin REST service that sits in front of TimescaleDB and exposes
cost governance primitives to any language, team, or service —
not just Python apps with the SDK installed.

All SDK features are available via HTTP:
    GET  /spend                    — cost by provider/model/project/day
    GET  /spend/trace/{trace_id}   — full trace cost breakdown
    GET  /budgets                  — all budgets with live spend
    POST /budgets                  — create or update a budget
    DELETE /budgets/{id}           — delete a budget
    POST /budgets/check            — check tags against all budgets (for pre-call enforcement)
    GET  /models                   — all models seen with lifetime stats
    GET  /pricing                  — active pricing table
    POST /pricing/sync             — trigger a pricing sync from bundled defaults
    GET  /health                   — writer health for all services
    GET  /health/live              — liveness probe (no DB required)

Start:
    uvicorn kostrack_api.main:app --host 0.0.0.0 --port 8080

Environment:
    KOSTRACK_DSN        — TimescaleDB DSN (required)
    KOSTRACK_API_KEY    — Bearer token for auth (optional but recommended)
    KOSTRACK_SERVICE_ID — service_id label for this API instance (default: api)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# ─── config ──────────────────────────────────────────────────────────────────

from kostrack.sync.pricing_sync import PricingSync


def _dsn() -> str:
    return os.environ.get("KOSTRACK_DSN", "")

def _api_key() -> str:
    return os.environ.get("KOSTRACK_API_KEY", "")

# ─── DB pool (simple — psycopg2, one conn per request) ───────────────────────

def _conn():
    dsn = _dsn()
    if not dsn:
        raise HTTPException(503, "KOSTRACK_DSN not configured")
    try:
        c = psycopg2.connect(dsn)
        c.autocommit = True
        return c
    except Exception as exc:
        raise HTTPException(503, f"Database unavailable: {exc}")


def _q(sql: str, params=()) -> list[dict]:
    c = _conn()
    try:
        with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        c.close()


def _exec(sql: str, params=()) -> None:
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(sql, params)
    finally:
        c.close()


# ─── auth ─────────────────────────────────────────────────────────────────────

bearer = HTTPBearer(auto_error=False)


def auth(creds: HTTPAuthorizationCredentials | None = Security(bearer)):
    key = _api_key()
    if not key:
        return  # auth disabled — no key configured
    if creds is None or creds.credentials != key:
        raise HTTPException(401, "Invalid or missing API key")


# ─── lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate DSN on startup — fail fast, don't serve 503s silently
    if _dsn():
        try:
            c = psycopg2.connect(_dsn())
            c.close()
        except Exception as exc:
            import sys
            print(f"[kostrack-api] WARNING: Cannot reach TimescaleDB: {exc}", file=sys.stderr)
    yield


# ─── app ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="kostrack Platform API",
    version="0.2.0",
    description="AI API cost governance — HTTP interface for cross-language platform integration",
    lifespan=lifespan,
)

# ─── schemas ─────────────────────────────────────────────────────────────────

class BudgetCreate(BaseModel):
    tag_key:         str
    tag_value:       str
    period:          str  = Field(..., pattern="^(daily|weekly|monthly)$")
    limit_usd:       float
    alert_threshold: float = 0.80
    enforce:         bool  = False
    service_id:      str   = "default"


class BudgetCheckRequest(BaseModel):
    tags:           dict[str, str]
    estimated_cost: float = 0.0


class PricingAddRequest(BaseModel):
    provider:        str
    model:           str
    pricing_model:   str   = "per_token"
    input_rate:      float  # USD per million tokens
    output_rate:     float  # USD per million tokens
    cache_read_rate: float | None = None
    cache_write_rate: float | None = None
    notes:           str | None = None


# ─── routes ──────────────────────────────────────────────────────────────────

@app.get("/health/live")
def liveness():
    """Liveness probe — no DB required."""
    return {"status": "ok", "service": "kostrack-api", "version": "0.2.0"}


@app.get("/health", dependencies=[Depends(auth)])
def health():
    """Writer health for all connected services."""
    rows = _q("SELECT * FROM writer_health ORDER BY updated_at DESC")
    return {"services": rows}


@app.get("/spend", dependencies=[Depends(auth)])
def spend(
    project:  str | None = Query(None, description="Filter by project tag"),
    provider: str | None = Query(None, description="Filter by provider"),
    service:  str | None = Query(None, description="Filter by service_id"),
    days:     int        = Query(30,   description="Lookback window in days"),
    granularity: str     = Query("day", pattern="^(hour|day)$"),
):
    """
    Spend breakdown by provider/model/project.
    Groups by the requested granularity (hour or day).
    """
    bucket = "1 hour" if granularity == "hour" else "1 day"
    where = ["time >= NOW() - INTERVAL %s"]
    params: list[Any] = [f"{days} days"]

    if project:
        where.append("tags->>'project' = %s")
        params.append(project)
    if provider:
        where.append("provider = %s")
        params.append(provider)
    if service:
        where.append("service_id = %s")
        params.append(service)

    rows = _q(f"""
        SELECT
            time_bucket('{bucket}', time)  AS bucket,
            service_id,
            provider,
            model,
            tags->>'project'               AS project,
            tags->>'feature'               AS feature,
            COUNT(*)                       AS calls,
            SUM(cost_usd)                  AS cost_usd,
            SUM(input_tokens)              AS input_tokens,
            SUM(output_tokens)             AS output_tokens,
            SUM(cached_tokens)             AS cached_tokens,
            ROUND(AVG(latency_ms))         AS avg_latency_ms
        FROM llm_calls
        WHERE {" AND ".join(where)}
        GROUP BY bucket, service_id, provider, model,
                 tags->>'project', tags->>'feature'
        ORDER BY bucket DESC, cost_usd DESC
    """, params)

    total = sum(float(r["cost_usd"] or 0) for r in rows)
    return {"total_cost_usd": total, "rows": rows, "filters": {
        "days": days, "project": project, "provider": provider, "granularity": granularity,
    }}


@app.get("/spend/trace/{trace_id}", dependencies=[Depends(auth)])
def spend_trace(trace_id: str):
    """
    Full per-call breakdown for a single workflow trace.
    Returns all spans sorted by time, with per-model cost rollup.
    """
    rows = _q("""
        SELECT
            time, provider, model, span_id, parent_span_id,
            tags->>'span_name' AS span_name,
            tags->>'feature'   AS feature,
            input_tokens, output_tokens, cached_tokens,
            cost_usd, latency_ms, token_breakdown
        FROM llm_calls
        WHERE trace_id = %s::uuid
        ORDER BY time ASC
    """, (trace_id,))

    if not rows:
        raise HTTPException(404, f"Trace {trace_id} not found")

    total = sum(float(r["cost_usd"] or 0) for r in rows)
    model_costs: dict[str, float] = {}
    for r in rows:
        m = r["model"]
        model_costs[m] = model_costs.get(m, 0.0) + float(r["cost_usd"] or 0)

    breakdown = sorted(
        [{"model": m, "cost_usd": c, "pct": round(c / total * 100, 1) if total else 0}
         for m, c in model_costs.items()],
        key=lambda x: x["cost_usd"], reverse=True,
    )

    return {
        "trace_id": trace_id,
        "total_cost_usd": total,
        "call_count": len(rows),
        "model_breakdown": breakdown,
        "spans": rows,
    }


@app.get("/budgets", dependencies=[Depends(auth)])
def list_budgets(service: str | None = Query(None)):
    """All budgets with live period spend."""
    where = "WHERE 1=1"
    params: list[Any] = []
    if service:
        where += " AND b.service_id = %s"
        params.append(service)

    rows = _q(f"""
        SELECT
            b.id,
            b.service_id,
            b.tag_key,
            b.tag_value,
            b.period,
            b.limit_usd,
            b.alert_threshold,
            b.enforce,
            b.created_at,
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
            ) AS spent_usd
        FROM budgets b
        {where}
        ORDER BY b.tag_key, b.tag_value, b.period
    """, params)

    for row in rows:
        limit = float(row["limit_usd"])
        spent = float(row["spent_usd"])
        row["pct"] = round(spent / limit * 100, 1) if limit > 0 else 0.0
        row["status"] = (
            "exceeded" if spent >= limit
            else "alert" if spent / limit >= float(row["alert_threshold"])
            else "ok"
        )

    return {"budgets": rows}


@app.post("/budgets", status_code=201, dependencies=[Depends(auth)])
def create_budget(body: BudgetCreate):
    """Create or update a budget (upserts on service_id + tag_key + tag_value + period)."""
    _exec("""
        INSERT INTO budgets
            (service_id, tag_key, tag_value, period,
             limit_usd, alert_threshold, enforce)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (service_id, tag_key, tag_value, period)
        DO UPDATE SET
            limit_usd        = EXCLUDED.limit_usd,
            alert_threshold  = EXCLUDED.alert_threshold,
            enforce          = EXCLUDED.enforce
    """, (
        body.service_id, body.tag_key, body.tag_value, body.period,
        body.limit_usd, body.alert_threshold, body.enforce,
    ))
    return {"status": "ok", "budget": body.model_dump()}


@app.delete("/budgets/{budget_id}", dependencies=[Depends(auth)])
def delete_budget(budget_id: int):
    """Delete a budget by its numeric ID."""
    _exec("DELETE FROM budgets WHERE id = %s", (budget_id,))
    return {"status": "deleted", "id": budget_id}


@app.post("/budgets/check", dependencies=[Depends(auth)])
def check_budgets(body: BudgetCheckRequest):
    """
    Check tags against all matching budgets.
    Returns budget states and raises 402 if any enforce=True budget is exceeded.

    Use this for pre-call enforcement from any language/service:
        POST /budgets/check
        {"tags": {"project": "openmanagr"}, "estimated_cost": 0.01}

    Returns 402 Payment Required if a hard budget is breached.
    Returns 200 with warnings if alert threshold crossed but enforce=False.
    """
    where_parts = []
    params: list[Any] = []
    for key, value in body.tags.items():
        where_parts.append("(tag_key = %s AND tag_value = %s)")
        params.extend([key, value])

    if not where_parts:
        return {"status": "ok", "budgets_checked": 0, "triggered": []}

    rows = _q(f"""
        SELECT
            b.id, b.tag_key, b.tag_value, b.period,
            b.limit_usd, b.alert_threshold, b.enforce,
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
            ) AS spent_usd
        FROM budgets b
        WHERE {" OR ".join(where_parts)}
    """, params)

    triggered = []
    for row in rows:
        spent = float(row["spent_usd"]) + body.estimated_cost
        limit = float(row["limit_usd"])
        pct   = spent / limit if limit > 0 else 0.0

        state = {
            "tag_key": row["tag_key"],
            "tag_value": row["tag_value"],
            "period": row["period"],
            "spent_usd": spent,
            "limit_usd": limit,
            "pct": round(pct * 100, 1),
            "enforce": row["enforce"],
            "status": (
                "exceeded" if pct >= 1.0
                else "alert" if pct >= float(row["alert_threshold"])
                else "ok"
            ),
        }

        if row["enforce"] and pct >= 1.0:
            raise HTTPException(402, {
                "error": "budget_exceeded",
                "tag_key": row["tag_key"],
                "tag_value": row["tag_value"],
                "period": row["period"],
                "spent_usd": spent,
                "limit_usd": limit,
            })

        if pct >= float(row["alert_threshold"]):
            triggered.append(state)

    return {
        "status": "ok",
        "budgets_checked": len(rows),
        "triggered": triggered,
    }


@app.get("/models", dependencies=[Depends(auth)])
def list_models():
    """All models ever seen, with lifetime call count and total cost."""
    rows = _q("""
        SELECT
            provider,
            model,
            COUNT(*)       AS total_calls,
            SUM(cost_usd)  AS total_cost_usd,
            MIN(time)      AS first_seen,
            MAX(time)      AS last_seen,
            ROUND(AVG(latency_ms)) AS avg_latency_ms
        FROM llm_calls
        GROUP BY provider, model
        ORDER BY total_cost_usd DESC
    """)
    return {"models": rows}


@app.get("/pricing", dependencies=[Depends(auth)])
def list_pricing():
    """Active pricing for all known models."""
    rows = _q("""
        SELECT provider, model, pricing_model,
               input_rate, output_rate,
               cache_write_rate, cache_read_rate,
               effective_from
        FROM pricing_current
        ORDER BY provider, model, pricing_model
    """)
    return {"pricing": rows}


@app.post("/pricing/sync", dependencies=[Depends(auth)])
def pricing_sync():
    """Sync bundled pricing into TimescaleDB. Idempotent."""
    import sys
    result = PricingSync(dsn=_dsn()).run()
    return {"status": "ok", **result}


@app.post("/pricing", status_code=201, dependencies=[Depends(auth)])
def add_pricing(body: PricingAddRequest):
    """Add or update pricing for a model. Rates in USD per million tokens."""
    import sys
    PricingSync(dsn=_dsn()).upsert(
        provider=body.provider,
        model=body.model,
        pricing_model=body.pricing_model,
        input_rate=body.input_rate / 1_000_000,
        output_rate=body.output_rate / 1_000_000,
        cache_read_rate=body.cache_read_rate / 1_000_000 if body.cache_read_rate else None,
        cache_write_rate=body.cache_write_rate / 1_000_000 if body.cache_write_rate else None,
        notes=body.notes,
    )
    return {"status": "ok", "model": f"{body.provider}/{body.model}"}