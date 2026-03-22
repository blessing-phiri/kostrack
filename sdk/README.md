<div align="center">

<img src="https://kostrack.netlify.app/assets/kostrack-logo-dark.png" alt="Kostrack" width="280">

**AI API cost governance** — track, attribute, and govern LLM API spend across Anthropic, OpenAI, Gemini, and DeepSeek, with full agentic workflow cost rollup.

[![PyPI](https://img.shields.io/pypi/v/kostrack?color=F5A623&labelColor=0D1B2E)](https://pypi.org/project/kostrack)
[![License](https://img.shields.io/badge/license-Apache%202.0-6B7FA3?labelColor=0D1B2E)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-149%20passed-2A8A4A?labelColor=0D1B2E)](sdk/tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-6B7FA3?labelColor=0D1B2E)](pyproject.toml)

[**Docs**](https://kostrack.netlify.app/docs/) · [**Quick Start**](https://kostrack.netlify.app/docs/quickstart.html) · [**PyPI**](https://pypi.org/project/kostrack)

</div>

---

## What it does

Kostrack sits between your application and your LLM providers. Every API call is intercepted, costed, attributed to a project/feature/user, and written asynchronously to TimescaleDB. Grafana dashboards are pre-provisioned and ready the moment `docker compose up -d` finishes.

```
Your App → kostrack SDK → LLM Provider (Anthropic / OpenAI / Gemini / DeepSeek)
                ↓  (async, <5ms)
          TimescaleDB
            ↙       ↘
     Grafana       Platform API  ←  any language / service
        ↑                ↑
     CLI ──────────────────
```

## Why not Helicone or LangSmith?

Those tools are built for ML engineers — prompt logging, evals, model quality. Kostrack is built for **financial governance**: cost per feature, cost per workflow run, budget alerts, CFO-exportable reports. Nobody else does agentic cost rollup or targets the African enterprise market where USD-denominated API costs hit differently.

---

## Quick start

```bash
# 1. Start the stack
cp .env.example .env       # set your passwords
docker compose up -d

# 2. Install the SDK
pip install kostrack

# 3. Instrument your app
```

```python
import kostrack

kostrack.configure(
    dsn="postgresql://kostrack:yourpassword@localhost/kostrack",
    service_id="your-app-name",
)

# Before: from anthropic import Anthropic
from kostrack import Anthropic

client = Anthropic(
    tags={
        "project":     "openmanagr",
        "feature":     "invoice-extraction",
        "environment": "production",
    }
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarise this invoice..."}],
)
```

```bash
# 4. Open Grafana → http://localhost:3000
```

---

## Agentic tracing with multi-model cost breakdown

Wrap multi-step workflows to get total cost per run — attributed by model:

```python
import kostrack
from kostrack import Anthropic, DeepSeek

anthropicc = Anthropic(tags={"project": "openmanagr"})
deepseek   = DeepSeek(tags={"project": "openmanagr"})

with kostrack.trace(tags={"feature": "month-end-close"}) as t:
    with kostrack.span("reason", parent=t):
        deepseek.chat.completions.create(model="deepseek-reasoner", ...)
    with kostrack.span("extract", parent=t):
        anthropiccc.messages.create(model="claude-haiku-4-5-20251001", ...)
    with kostrack.span("post", parent=t):
        anthropiccc.messages.create(model="claude-sonnet-4-6", ...)

print(f"Workflow cost: ${t.total_cost_usd:.6f} across {t.call_count} calls")
for item in t.cost_breakdown():
    print(f"  {item['model']}: ${item['cost_usd']:.6f} ({item['pct']}%)")
# deepseek-reasoner:          $0.0142  (71.0%)
# claude-haiku-4-5-20251001:  $0.0038  (19.0%)
# claude-sonnet-4-6:          $0.0020  (10.0%)
```

---

## All four providers

```python
from kostrack import Anthropic, OpenAI, GenerativeModel, DeepSeek

anthropic = Anthropic(tags={"project": "myapp"})
openai    = OpenAI(tags={"project": "myapp"})
gemini    = GenerativeModel("gemini-2.0-flash", tags={"project": "myapp"})
deepseek  = DeepSeek(tags={"project": "myapp"})          # V3 and R1

# DeepSeek-R1 reasoning tokens tracked in token_breakdown.thinking
reasoner = DeepSeek(tags={"project": "myapp", "feature": "ifrs-interpretation"})
response = reasoner.chat.completions.create(model="deepseek-reasoner", messages=[...])
```

---

## Architecture

| Layer | Technology |
|-------|-----------|
| SDK | Python 3.11+, provider SDKs |
| Write path | Async batch writer, SQLite fallback |
| Database | TimescaleDB (Postgres extension) |
| Dashboards | Grafana — pre-provisioned |
| Platform API | FastAPI — language-agnostic governance |
| CLI | `kostrack` command — operator tooling |
| Deployment | Docker Compose |

**Resilience:** If TimescaleDB is unreachable, records buffer to `~/.kostrack/buffer.db` and flush automatically when connectivity returns. Write overhead is under 5ms.

---

## Tests

```bash
cd sdk
pip install -e ".[dev]"

python -m pytest tests/ -v          # 149 tests, fully offline
python tests/integration_test.py    # live integration test (requires DB)
```

---

## Documentation

Full documentation at **[kostrack.netlify.app/docs](https://kostrack.netlify.app/docs/)**

**SDK** — [Anthropic](https://kostrack.netlify.app/docs/anthropic.html) · [OpenAI](https://kostrack.netlify.app/docs/openai.html) · [Gemini](https://kostrack.netlify.app/docs/gemini.html) · [DeepSeek](https://kostrack.netlify.app/docs/deepseek.html) · [Tracing](https://kostrack.netlify.app/docs/tracing.html) · [Tags](https://kostrack.netlify.app/docs/tags.html)

**Governance** — [Budgets](https://kostrack.netlify.app/docs/budgets.html) · [Pricing Sync](https://kostrack.netlify.app/docs/pricing-sync.html)

**Platform** — [Platform API](https://kostrack.netlify.app/docs/platform-api.html) · [CLI Reference](https://kostrack.netlify.app/docs/cli.html)

**Integrations** — [FastAPI](https://kostrack.netlify.app/docs/fastapi.html) · [LangGraph](https://kostrack.netlify.app/docs/langgraph.html) · [Grafana](https://kostrack.netlify.app/docs/grafana.html) · [Queries](https://kostrack.netlify.app/docs/queries.html)

---

## CLI

Full cost governance from the terminal — no Python required:

```bash
kostrack status                                    # 24h spend by provider
kostrack spend --project openmanagr --days 7       # daily breakdown
kostrack budgets                                   # all budgets + live spend
kostrack budget set project openmanagr monthly 50 --enforce
kostrack pricing sync                              # push bundled prices to DB
kostrack models                                    # all models, lifetime cost
kostrack traces --project openmanagr               # recent workflow traces
kostrack health                                    # writer health per service
```

---

## Platform API

A REST service on port 8080 — any language, any service, no SDK required:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/spend` | Cost breakdown with filters |
| `GET` | `/spend/trace/{id}` | Per-model breakdown for one trace |
| `POST` | `/budgets/check` | Pre-call enforcement — returns 402 if exceeded |
| `GET` | `/models` | All models with lifetime stats |
| `POST` | `/pricing/sync` | Sync bundled pricing to DB |
| `GET` | `/health/live` | Liveness probe |

Interactive docs at **http://localhost:8080/docs**.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

*Built by [Blessing Phiri](https://github.com/bphiri) · Applied AI Engineer · Init Data Solutions · Harare, Zimbabwe.*
