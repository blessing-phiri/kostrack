-- =============================================================================
-- kostrack — Database Initialisation
-- TimescaleDB schema + seed pricing data
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Primary call log — every LLM API call becomes one row
CREATE TABLE IF NOT EXISTS llm_calls (
    time              TIMESTAMPTZ       NOT NULL,
    service_id        TEXT              NOT NULL DEFAULT 'default',
    provider          TEXT              NOT NULL,   -- 'anthropic' | 'openai' | 'gemini'
    model             TEXT              NOT NULL,   -- e.g. 'claude-sonnet-4-6'
    pricing_model     TEXT              NOT NULL DEFAULT 'per_token', -- 'per_token' | 'batch'

    -- Flat token columns — fast aggregation
    input_tokens      INTEGER           NOT NULL DEFAULT 0,
    output_tokens     INTEGER           NOT NULL DEFAULT 0,
    cached_tokens     INTEGER           NOT NULL DEFAULT 0,   -- total cache hits (read)
    cost_usd          NUMERIC(12, 8)    NOT NULL DEFAULT 0,

    -- Provider-specific token breakdown
    -- Anthropic: {"input": 150, "output": 300, "cache_write": 50, "cache_read": 150, "thinking": 0}
    -- OpenAI:    {"input": 150, "output": 300, "cached_prompt": 100}
    -- Gemini:    {"input": 150, "output": 300, "context_cache": 0}
    token_breakdown   JSONB             NOT NULL DEFAULT '{}'::jsonb,

    latency_ms        INTEGER,

    -- Distributed tracing — hierarchical spans
    trace_id          UUID,             -- root workflow identifier
    span_id           UUID,             -- this call's identifier
    parent_span_id    UUID,             -- parent call if nested

    -- Attribution — extensible tag dict
    -- Reserved keys: project, feature, user_id, team, environment
    -- Arbitrary keys allowed: ab_test, version, region, etc.
    tags              JSONB             NOT NULL DEFAULT '{}'::jsonb,

    -- Anything else — request params, response metadata, etc.
    metadata          JSONB             NOT NULL DEFAULT '{}'::jsonb
);

-- Convert to TimescaleDB hypertable — partition by time (1 week chunks)
SELECT create_hypertable(
    'llm_calls',
    'time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_llm_calls_service_time
    ON llm_calls (service_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_llm_calls_provider_model
    ON llm_calls (provider, model, time DESC);

CREATE INDEX IF NOT EXISTS idx_llm_calls_trace
    ON llm_calls (trace_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_llm_calls_tags
    ON llm_calls USING GIN (tags);

-- =============================================================================
-- PRICING TABLE
-- Versioned — effective_from / effective_to allows historical cost recalculation
-- =============================================================================

CREATE TABLE IF NOT EXISTS pricing (
    provider          TEXT              NOT NULL,
    model             TEXT              NOT NULL,
    pricing_model     TEXT              NOT NULL DEFAULT 'per_token',
    input_rate        NUMERIC(12, 8),   -- USD per token
    output_rate       NUMERIC(12, 8),   -- USD per token
    cache_write_rate  NUMERIC(12, 8),   -- USD per token (Anthropic cache creation)
    cache_read_rate   NUMERIC(12, 8),   -- USD per token (Anthropic cache hits / OpenAI cached)
    unit              TEXT              NOT NULL DEFAULT 'USD',
    effective_from    TIMESTAMPTZ       NOT NULL,
    effective_to      TIMESTAMPTZ,      -- NULL = currently active
    metadata          JSONB,            -- provider-specific notes
    PRIMARY KEY (provider, model, pricing_model, effective_from)
);

CREATE INDEX IF NOT EXISTS idx_pricing_active
    ON pricing (provider, model, pricing_model, effective_to)
    WHERE effective_to IS NULL;

-- =============================================================================
-- BUDGETS TABLE
-- Tag-based budget scoping — e.g. project=openmanagr, team=engineering
-- =============================================================================

CREATE TABLE IF NOT EXISTS budgets (
    id                SERIAL            PRIMARY KEY,
    service_id        TEXT              NOT NULL DEFAULT 'default',
    tag_key           TEXT              NOT NULL,   -- e.g. 'project'
    tag_value         TEXT              NOT NULL,   -- e.g. 'openmanagr'
    period            TEXT              NOT NULL,   -- 'daily' | 'weekly' | 'monthly'
    limit_usd         NUMERIC(12, 2)    NOT NULL,
    alert_threshold   NUMERIC(3, 2)     NOT NULL DEFAULT 0.80,  -- alert at 80% by default
    enforce           BOOLEAN           NOT NULL DEFAULT FALSE,  -- hard limit vs alert only
    created_at        TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
    UNIQUE (service_id, tag_key, tag_value, period)
);

-- =============================================================================
-- WRITER HEALTH TABLE
-- AsyncBatchWriter surfaces its state here
-- =============================================================================

CREATE TABLE IF NOT EXISTS writer_health (
    service_id        TEXT              PRIMARY KEY,
    timescale_ok      BOOLEAN           NOT NULL DEFAULT TRUE,
    sqlite_backlog    INTEGER           NOT NULL DEFAULT 0,
    queued            BIGINT            NOT NULL DEFAULT 0,
    written_timescale BIGINT            NOT NULL DEFAULT 0,
    written_sqlite    BIGINT            NOT NULL DEFAULT 0,
    failed            BIGINT            NOT NULL DEFAULT 0,
    last_flush        TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ       NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- CONTINUOUS AGGREGATES
-- Pre-computed hourly and daily rollups for fast dashboard queries
-- =============================================================================

-- Hourly rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS llm_calls_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time)   AS bucket,
    service_id,
    provider,
    model,
    tags->>'project'              AS project,
    tags->>'feature'              AS feature,
    tags->>'team'                 AS team,
    tags->>'environment'          AS environment,
    COUNT(*)                      AS call_count,
    SUM(input_tokens)             AS input_tokens,
    SUM(output_tokens)            AS output_tokens,
    SUM(cached_tokens)            AS cached_tokens,
    SUM(cost_usd)                 AS cost_usd,
    AVG(latency_ms)               AS avg_latency_ms,
    PERCENTILE_CONT(0.95)
        WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
FROM llm_calls
GROUP BY bucket, service_id, provider, model,
         tags->>'project', tags->>'feature',
         tags->>'team', tags->>'environment'
WITH NO DATA;

-- Daily rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS llm_calls_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)    AS bucket,
    service_id,
    provider,
    model,
    tags->>'project'              AS project,
    tags->>'feature'              AS feature,
    tags->>'team'                 AS team,
    tags->>'environment'          AS environment,
    COUNT(*)                      AS call_count,
    SUM(input_tokens)             AS input_tokens,
    SUM(output_tokens)            AS output_tokens,
    SUM(cached_tokens)            AS cached_tokens,
    SUM(cost_usd)                 AS cost_usd,
    AVG(latency_ms)               AS avg_latency_ms
FROM llm_calls
GROUP BY bucket, service_id, provider, model,
         tags->>'project', tags->>'feature',
         tags->>'team', tags->>'environment'
WITH NO DATA;

-- Refresh policies — keep aggregates current
SELECT add_continuous_aggregate_policy('llm_calls_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy('llm_calls_daily',
    start_offset => INTERVAL '3 days',
    end_offset   => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Data retention — keep raw data 90 days, aggregates indefinitely
SELECT add_retention_policy('llm_calls',
    drop_after => INTERVAL '90 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- SEED DATA — Pricing (as of March 2026)
-- All rates are USD per token
-- =============================================================================

-- -------------------------
-- Anthropic
-- -------------------------
INSERT INTO pricing (provider, model, pricing_model, input_rate, output_rate, cache_write_rate, cache_read_rate, effective_from, metadata)
VALUES
-- Claude Sonnet 4.6
('anthropic', 'claude-sonnet-4-6', 'per_token',
    3.00 / 1000000,   -- $3.00 per 1M input tokens
    15.00 / 1000000,  -- $15.00 per 1M output tokens
    3.75 / 1000000,   -- $3.75 per 1M cache write tokens
    0.30 / 1000000,   -- $0.30 per 1M cache read tokens
    '2025-01-01 00:00:00+00',
    '{"context_window": 200000, "notes": "claude-sonnet-4-6"}'::jsonb),

-- Claude Opus 4.6
('anthropic', 'claude-opus-4-6', 'per_token',
    15.00 / 1000000,
    75.00 / 1000000,
    18.75 / 1000000,
    1.50 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"context_window": 200000, "notes": "claude-opus-4-6"}'::jsonb),

-- Claude Haiku 4.5
('anthropic', 'claude-haiku-4-5-20251001', 'per_token',
    0.80 / 1000000,
    4.00 / 1000000,
    1.00 / 1000000,
    0.08 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"context_window": 200000, "notes": "claude-haiku-4-5"}'::jsonb),

-- Anthropic Batch API — 50% discount on input/output, no caching
('anthropic', 'claude-sonnet-4-6', 'batch',
    1.50 / 1000000,
    7.50 / 1000000,
    NULL,
    NULL,
    '2025-01-01 00:00:00+00',
    '{"notes": "Batch API — 50% discount, 24hr turnaround"}'::jsonb),

('anthropic', 'claude-opus-4-6', 'batch',
    7.50 / 1000000,
    37.50 / 1000000,
    NULL,
    NULL,
    '2025-01-01 00:00:00+00',
    '{"notes": "Batch API — 50% discount, 24hr turnaround"}'::jsonb),

('anthropic', 'claude-haiku-4-5-20251001', 'batch',
    0.40 / 1000000,
    2.00 / 1000000,
    NULL,
    NULL,
    '2025-01-01 00:00:00+00',
    '{"notes": "Batch API — 50% discount, 24hr turnaround"}'::jsonb)

ON CONFLICT DO NOTHING;

-- -------------------------
-- OpenAI
-- -------------------------
INSERT INTO pricing (provider, model, pricing_model, input_rate, output_rate, cache_read_rate, effective_from, metadata)
VALUES
-- GPT-4o
('openai', 'gpt-4o', 'per_token',
    2.50 / 1000000,
    10.00 / 1000000,
    1.25 / 1000000,   -- cached input 50% discount
    '2025-01-01 00:00:00+00',
    '{"context_window": 128000}'::jsonb),

-- GPT-4o mini
('openai', 'gpt-4o-mini', 'per_token',
    0.15 / 1000000,
    0.60 / 1000000,
    0.075 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"context_window": 128000}'::jsonb),

-- o1
('openai', 'o1', 'per_token',
    15.00 / 1000000,
    60.00 / 1000000,
    7.50 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"context_window": 200000, "notes": "reasoning model"}'::jsonb),

-- o3-mini
('openai', 'o3-mini', 'per_token',
    1.10 / 1000000,
    4.40 / 1000000,
    0.55 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"context_window": 200000, "notes": "reasoning model"}'::jsonb),

-- Batch API
('openai', 'gpt-4o', 'batch',
    1.25 / 1000000,
    5.00 / 1000000,
    NULL,
    '2025-01-01 00:00:00+00',
    '{"notes": "Batch API — 50% discount"}'::jsonb),

('openai', 'gpt-4o-mini', 'batch',
    0.075 / 1000000,
    0.30 / 1000000,
    NULL,
    '2025-01-01 00:00:00+00',
    '{"notes": "Batch API — 50% discount"}'::jsonb)

ON CONFLICT DO NOTHING;

-- -------------------------
-- Gemini
-- -------------------------
INSERT INTO pricing (provider, model, pricing_model, input_rate, output_rate, cache_read_rate, effective_from, metadata)
VALUES
-- Gemini 2.0 Flash
('gemini', 'gemini-2.0-flash', 'per_token',
    0.10 / 1000000,
    0.40 / 1000000,
    0.025 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"context_window": 1048576, "notes": "text/image/video input"}'::jsonb),

-- Gemini 2.0 Flash Lite
('gemini', 'gemini-2.0-flash-lite', 'per_token',
    0.075 / 1000000,
    0.30 / 1000000,
    NULL,
    '2025-01-01 00:00:00+00',
    '{"context_window": 1048576}'::jsonb),

-- Gemini 1.5 Pro
('gemini', 'gemini-1.5-pro', 'per_token',
    1.25 / 1000000,
    5.00 / 1000000,
    0.3125 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"context_window": 2097152, "notes": "up to 128k tokens"}'::jsonb),

-- Gemini 1.5 Flash
('gemini', 'gemini-1.5-flash', 'per_token',
    0.075 / 1000000,
    0.30 / 1000000,
    0.01875 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"context_window": 1048576}'::jsonb)

ON CONFLICT DO NOTHING;


-- -------------------------
-- DeepSeek
-- -------------------------
INSERT INTO pricing (provider, model, pricing_model, input_rate, output_rate, cache_read_rate, effective_from, metadata)
VALUES
-- DeepSeek-V3
('deepseek', 'deepseek-chat', 'per_token',
    0.27 / 1000000,
    1.10 / 1000000,
    0.07 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"notes": "DeepSeek-V3, 64k context. Cache hit = $0.07/M input tokens."}'::jsonb),

-- DeepSeek-R1
('deepseek', 'deepseek-reasoner', 'per_token',
    0.55 / 1000000,
    2.19 / 1000000,
    0.14 / 1000000,
    '2025-01-01 00:00:00+00',
    '{"notes": "DeepSeek-R1. Reasoning tokens billed as output. Cache hit = $0.14/M."}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- Current active pricing — latest effective entry per provider/model
CREATE OR REPLACE VIEW pricing_current AS
SELECT DISTINCT ON (provider, model, pricing_model)
    provider, model, pricing_model,
    input_rate, output_rate,
    cache_write_rate, cache_read_rate,
    unit, effective_from, metadata
FROM pricing
WHERE effective_to IS NULL
ORDER BY provider, model, pricing_model, effective_from DESC;

-- Trace cost rollup — total cost and call count per trace
CREATE OR REPLACE VIEW trace_costs AS
SELECT
    trace_id,
    service_id,
    tags->>'project'    AS project,
    tags->>'feature'    AS feature,
    MIN(time)           AS started_at,
    MAX(time)           AS last_call_at,
    EXTRACT(EPOCH FROM (MAX(time) - MIN(time))) * 1000 AS duration_ms,
    COUNT(*)            AS call_count,
    SUM(input_tokens)   AS input_tokens,
    SUM(output_tokens)  AS output_tokens,
    SUM(cost_usd)       AS total_cost_usd
FROM llm_calls
WHERE trace_id IS NOT NULL
GROUP BY trace_id, service_id, tags->>'project', tags->>'feature';

-- =============================================================================
-- DONE
-- =============================================================================