<div align="center">

<img src="https://kostrack.netlify.app/assets/kostrack-logo-dark.png" alt="Kostrack" width="280">

**AI API cost governance** — track, attribute, and govern LLM API spend across Anthropic, OpenAI, and Gemini, with full agentic workflow cost rollup.

[![PyPI](https://img.shields.io/pypi/v/kostrack?color=F5A623&labelColor=0D1B2E)](https://pypi.org/project/kostrack)
[![License](https://img.shields.io/badge/license-Apache%202.0-6B7FA3?labelColor=0D1B2E)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-41%20passed-2A8A4A?labelColor=0D1B2E)](sdk/tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-6B7FA3?labelColor=0D1B2E)](pyproject.toml)

[**Docs**](https://kostrack.netlify.app/docs/) · [**Quick Start**](https://kostrack.netlify.app/docs/quickstart.html) · [**PyPI**](https://pypi.org/project/kostrack)

</div>

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

---

## All three providers

```python
from kostrack import Anthropic, OpenAI, GenerativeModel

anthropic = Anthropic(tags={"project": "myapp"})
openai    = OpenAI(tags={"project": "myapp"})
gemini    = GenerativeModel("gemini-2.0-flash", tags={"project": "myapp"})
```

---

## Architecture

| Layer | Technology |
|-------|-----------|
| SDK | Python 3.11+, provider SDKs |
| Write path | Async batch writer, SQLite fallback |
| Database | TimescaleDB (Postgres extension) |
| Dashboards | Grafana — pre-provisioned |
| Deployment | Docker Compose |

**Resilience:** If TimescaleDB is unreachable, records buffer to `~/.kostrack/buffer.db` and flush automatically when connectivity returns. Write overhead is under 5ms.

---

## Tests

```bash
cd sdk
pip install -e ".[dev]"

python -m pytest tests/test_e2e.py -v         # 41 tests, offline
python tests/integration_test.py              # live integration test
```

---

## Documentation

Full documentation at **[kostrack.netlify.app/docs](https://kostrack.netlify.app/docs/)**

- [Introduction](https://kostrack.netlify.app/docs/)
- [Quick Start](https://kostrack.netlify.app/docs/quickstart.html)
- [configure()](https://kostrack.netlify.app/docs/configure.html)
- [Anthropic provider](https://kostrack.netlify.app/docs/anthropic.html)
- [OpenAI provider](https://kostrack.netlify.app/docs/openai.html)
- [Gemini provider](https://kostrack.netlify.app/docs/gemini.html)
- [Tracing & Spans](https://kostrack.netlify.app/docs/tracing.html)
- [LangGraph integration](https://kostrack.netlify.app/docs/langgraph.html)
- [FastAPI integration](https://kostrack.netlify.app/docs/fastapi.html)
- [Useful Queries](https://kostrack.netlify.app/docs/queries.html)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

*Built by [Blessing Phiri](https://github.com/bphiri) · Applied AI Engineer · Init Data Solutions · Harare, Zimbabwe.*
