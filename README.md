# kostrack

**AI API cost governance** — track, attribute, and govern LLM API spend across Anthropic, OpenAI, and Gemini, with full agentic workflow cost rollup.

---

## What it does

Kostrack sits between your application and your LLM providers. Every API call is intercepted, costed, attributed to a project/feature/user, and written asynchronously to TimescaleDB. Grafana dashboards are pre-provisioned and ready the moment `docker compose up -d` finishes.

```
Your App → kostrack SDK → LLM Provider (Anthropic / OpenAI / Gemini)
                ↓
          TimescaleDB
                ↓
            Grafana
```

---

## Why not Helicone or LangSmith?

Those tools are built for ML engineers — prompt logging, evals, model quality. Kostrack is built for **financial governance**: cost per feature, cost per workflow run, budget alerts, CFO-exportable reports. Nobody else does agentic cost rollup or targets the African enterprise market where USD-denominated API costs hit differently.

---

## Quick start

### 1. Start the stack

```bash
cp .env.example .env        # set your passwords
docker compose up -d
```

TimescaleDB starts on `localhost:5432`, Grafana on `localhost:3000`.
Schema and seed pricing load automatically on first run.

### 2. Install the SDK

```bash
pip install kostrack
```

### 3. Configure and instrument

```python
import kostrack

kostrack.configure(
    dsn="postgresql://kostrack:yourpassword@localhost/kostrack",
    service_id="your-app-name",
)
```

### 4. Swap one import

```python
# Before
from anthropic import Anthropic
client = Anthropic()

# After — identical API, costs captured
from kostrack import Anthropic
client = Anthropic(
    tags={
        "project":     "openmanagr",
        "feature":     "invoice-extraction",
        "environment": "production",
    }
)
```

Same pattern for OpenAI and Gemini:

```python
from kostrack import OpenAI, GenerativeModel

oai = OpenAI(tags={"project": "openmanagr", "feature": "gl-classification"})
gem = GenerativeModel("gemini-2.0-flash", tags={"project": "openmanagr"})
```

### 5. Open Grafana

→ **http://localhost:3000** · login: `admin` / your `GRAFANA_PASSWORD`

The **Kostrack — Overview** dashboard loads automatically.

---

## Agentic tracing

Wrap multi-step workflows to get total cost per workflow run:

```python
import kostrack
from kostrack import Anthropic

client = Anthropic(tags={"project": "openmanagr"})

with kostrack.trace(tags={"feature": "month-end-close"}) as t:
    with kostrack.span("validate", parent=t):
        client.messages.create(...)
    with kostrack.span("classify", parent=t):
        client.messages.create(...)
    with kostrack.span("post", parent=t):
        client.messages.create(...)

print(f"Workflow cost: ${t.total_cost_usd:.6f} across {t.call_count} calls")
```

All child calls share a `trace_id`. Query the rollup:

```sql
SELECT * FROM trace_costs ORDER BY total_cost_usd DESC LIMIT 10;
```

---

## FastAPI integration

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import kostrack

@asynccontextmanager
async def lifespan(app: FastAPI):
    kostrack.configure(
        dsn="postgresql://kostrack:password@localhost/kostrack",
        service_id="openmanagr",
    )
    yield
    kostrack.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/kostrack/health")
def health():
    return kostrack.health()
```

---

## Attribution tags

```python
tags = {
    # Reserved — shown in Grafana dropdowns
    "project":     "openmanagr",
    "feature":     "invoice-extraction",
    "team":        "engineering",
    "environment": "production",
    "user_id":     "user_123",
    # Any additional keys allowed
    "ab_test":     "variant-b",
}
```

Override tags per call:

```python
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[...],
    kostrack_tags={"feature": "override-for-this-call"},
)
```

---

## Architecture

| Layer      | Technology                          |
| ---------- | ----------------------------------- |
| SDK        | Python 3.11+, provider SDKs         |
| Write path | Async batch writer, SQLite fallback |
| Database   | TimescaleDB (Postgres extension)    |
| Dashboards | Grafana — pre-provisioned           |
| Deployment | Docker Compose                      |

**Resilience:** If TimescaleDB is unreachable, records buffer to `~/.kostrack/buffer.db` and flush automatically when connectivity returns. Write overhead is under 5ms — never blocks your LLM calls.

---

## Supported providers

| Provider  | Models                                                        | Special tokens                    |
| --------- | ------------------------------------------------------------- | --------------------------------- |
| Anthropic | claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5 + batch  | cache_write, cache_read, thinking |
| OpenAI    | gpt-4o, gpt-4o-mini, o1, o3-mini + batch                      | cached_prompt, reasoning_tokens   |
| Gemini    | gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-pro/flash | context_cache, thoughts_tokens    |

---

## Environment variables

| Variable            | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `KOSTRACK_DSN`      | PostgreSQL DSN — alternative to passing `dsn=` to `configure()` |
| `ANTHROPIC_API_KEY` | Anthropic API key                                               |
| `OPENAI_API_KEY`    | OpenAI API key                                                  |
| `GEMINI_API_KEY`    | Gemini API key                                                  |
| `TSDB_PASSWORD`     | TimescaleDB password (Docker Compose)                           |
| `GRAFANA_PASSWORD`  | Grafana admin password                                          |

---

## Tests

```bash
cd sdk
pip install -e ".[dev]"

# Offline unit tests — no keys or DB needed
python -m pytest tests/test_e2e.py -v

# Live integration test — requires running stack and API keys
python tests/integration_test.py
```

41 unit tests, 0 failures.

---

## Project structure

```
kostrack/
├── docker-compose.yml
├── init-db.sql
├── .env.example
├── grafana-provisioning/
│   ├── datasources/kostrack.yml
│   └── dashboards/overview.json
└── sdk/
    ├── pyproject.toml
    └── kostrack/
        ├── __init__.py
        ├── models.py
        ├── tracing.py
        ├── providers/
        ├── calculators/
        └── writers/
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

_Built by [Blessing Phiri](https://github.com/bphiri) · Harare, Zimbabwe._
