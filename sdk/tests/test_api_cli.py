"""
kostrack — API & CLI Test Suite

Tests for:
- Platform REST API: all routes, auth, budget check enforcement, trace breakdown
- CLI: argument parsing, all subcommands, JSON output, error handling
- Integration: API budget check returns 402 on exceeded enforce=True budget

All tests run fully offline — TimescaleDB is mocked via monkeypatching.

Run:
    cd sdk
    python -m pytest tests/test_api_cli.py -v
"""

import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make both the SDK and API importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))

# ─── fixtures ────────────────────────────────────────────────────────────────

MOCK_DSN = "postgresql://mock:mock@localhost/mock"

SAMPLE_SPEND_ROWS = [
    {
        "bucket": "2026-03-22T00:00:00",
        "service_id": "openmanagr",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "project": "openmanagr",
        "feature": "invoice-extraction",
        "calls": 142,
        "cost_usd": 0.8512,
        "input_tokens": 284000,
        "output_tokens": 71000,
        "cached_tokens": 42000,
        "avg_latency_ms": 812,
    },
    {
        "bucket": "2026-03-22T00:00:00",
        "service_id": "openmanagr",
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "project": "openmanagr",
        "feature": "ifrs-interpretation",
        "calls": 28,
        "cost_usd": 0.3240,
        "input_tokens": 56000,
        "output_tokens": 89600,
        "cached_tokens": 0,
        "avg_latency_ms": 3421,
    },
]

SAMPLE_BUDGETS = [
    {
        "id": 1,
        "service_id": "default",
        "tag_key": "project",
        "tag_value": "openmanagr",
        "period": "monthly",
        "limit_usd": 50.00,
        "alert_threshold": 0.80,
        "enforce": False,
        "created_at": "2026-03-01T00:00:00Z",
        "spent_usd": 42.10,
    },
    {
        "id": 2,
        "service_id": "default",
        "tag_key": "team",
        "tag_value": "engineering",
        "period": "monthly",
        "limit_usd": 200.00,
        "alert_threshold": 0.80,
        "enforce": True,
        "created_at": "2026-03-01T00:00:00Z",
        "spent_usd": 201.50,   # exceeded
    },
]

SAMPLE_MODELS = [
    {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "total_calls": 1420,
        "total_cost_usd": 8.512,
        "first_seen": "2026-03-01T00:00:00Z",
        "last_seen": "2026-03-22T11:00:00Z",
        "avg_latency_ms": 812,
    },
    {
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "total_calls": 280,
        "total_cost_usd": 3.240,
        "first_seen": "2026-03-15T00:00:00Z",
        "last_seen": "2026-03-22T10:30:00Z",
        "avg_latency_ms": 3421,
    },
]

SAMPLE_PRICING = [
    {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "pricing_model": "per_token",
        "input_rate": 0.000003,
        "output_rate": 0.000015,
        "cache_write_rate": 0.000003750,
        "cache_read_rate": 0.00000030,
        "effective_from": "2025-01-01T00:00:00Z",
    },
    {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "pricing_model": "per_token",
        "input_rate": 0.00000027,
        "output_rate": 0.0000011,
        "cache_write_rate": None,
        "cache_read_rate": 0.00000007,
        "effective_from": "2025-01-01T00:00:00Z",
    },
]

SAMPLE_TRACES = [
    {
        "trace_id": "11111111-1111-1111-1111-111111111111",
        "project": "openmanagr",
        "feature": "month-end-close",
        "started_at": "2026-03-22T09:00:00Z",
        "duration_ms": 8420,
        "calls": 6,
        "models": 2,
        "cost_usd": 0.0384,
    },
]

SAMPLE_HEALTH = [
    {
        "service_id": "openmanagr",
        "timescale_ok": True,
        "sqlite_backlog": 0,
        "queued": 0,
        "written_timescale": 12470,
        "written_sqlite": 0,
        "failed": 0,
        "last_flush": "2026-03-22T11:00:01Z",
        "updated_at": "2026-03-22T11:00:01Z",
    },
]

SAMPLE_TRACE_SPANS = [
    {
        "time": "2026-03-22T09:00:00Z",
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "span_id": "aaaa-aaaa",
        "parent_span_id": None,
        "span_name": "reason",
        "feature": "month-end-close",
        "input_tokens": 1200,
        "output_tokens": 3600,
        "cached_tokens": 0,
        "cost_usd": 0.0142,
        "latency_ms": 4200,
        "token_breakdown": {"input": 1200, "output": 3600, "thinking": 2800},
    },
    {
        "time": "2026-03-22T09:04:12Z",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "span_id": "bbbb-bbbb",
        "parent_span_id": "aaaa-aaaa",
        "span_name": "extract",
        "feature": "month-end-close",
        "input_tokens": 800,
        "output_tokens": 200,
        "cached_tokens": 0,
        "cost_usd": 0.0058,
        "latency_ms": 812,
        "token_breakdown": {"input": 800, "output": 200},
    },
]


# ─── API client fixture ───────────────────────────────────────────────────────

@pytest.fixture()
def client(monkeypatch):
    """FastAPI TestClient with DB calls mocked."""
    monkeypatch.setenv("KOSTRACK_DSN", MOCK_DSN)
    monkeypatch.delenv("KOSTRACK_API_KEY", raising=False)  # auth disabled

    from fastapi.testclient import TestClient
    import main as api_main

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture()
def authed_client(monkeypatch):
    """TestClient with API key auth enabled."""
    monkeypatch.setenv("KOSTRACK_DSN", MOCK_DSN)
    monkeypatch.setenv("KOSTRACK_API_KEY", "test-secret-key")

    from fastapi.testclient import TestClient
    import importlib
    import main as api_main
    importlib.reload(api_main)  # re-read env vars

    with TestClient(api_main.app) as c:
        yield c


# ─── API: liveness ────────────────────────────────────────────────────────────

class TestAPILiveness:

    def test_liveness_no_db_required(self, client):
        r = client.get("/health/live")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "kostrack-api"
        assert data["version"] == "0.2.0"

    def test_liveness_does_not_need_auth(self, authed_client):
        """Liveness probe must work without a bearer token."""
        r = authed_client.get("/health/live")
        assert r.status_code == 200


# ─── API: auth ────────────────────────────────────────────────────────────────

class TestAPIAuth:

    def test_no_api_key_configured_allows_all(self, client, monkeypatch):
        """When KOSTRACK_API_KEY is unset, every route is open."""
        with patch("main._q", return_value=SAMPLE_HEALTH):
            r = client.get("/health")
        assert r.status_code == 200

    def test_wrong_key_returns_401(self, authed_client):
        r = authed_client.get("/health", headers={"Authorization": "Bearer wrong-key"})
        assert r.status_code == 401

    def test_correct_key_allows_access(self, authed_client):
        with patch("main._q", return_value=SAMPLE_HEALTH):
            r = authed_client.get(
                "/health",
                headers={"Authorization": "Bearer test-secret-key"},
            )
        assert r.status_code == 200

    def test_missing_auth_header_returns_401(self, authed_client):
        r = authed_client.get("/health")
        assert r.status_code == 401


# ─── API: health ──────────────────────────────────────────────────────────────

class TestAPIHealth:

    def test_health_returns_services(self, client):
        with patch("main._q", return_value=SAMPLE_HEALTH):
            r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "services" in data
        assert data["services"][0]["service_id"] == "openmanagr"
        assert data["services"][0]["timescale_ok"] is True

    def test_health_empty_when_no_services(self, client):
        with patch("main._q", return_value=[]):
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["services"] == []


# ─── API: spend ───────────────────────────────────────────────────────────────

class TestAPISpend:

    def test_spend_returns_rows_and_total(self, client):
        with patch("main._q", return_value=SAMPLE_SPEND_ROWS):
            r = client.get("/spend")
        assert r.status_code == 200
        data = r.json()
        assert "total_cost_usd" in data
        assert abs(data["total_cost_usd"] - 1.1752) < 0.001
        assert len(data["rows"]) == 2

    def test_spend_project_filter_passed_to_query(self, client):
        with patch("main._q", return_value=[SAMPLE_SPEND_ROWS[0]]) as mock_q:
            r = client.get("/spend?project=openmanagr&days=7")
        assert r.status_code == 200
        data = r.json()
        assert data["filters"]["project"] == "openmanagr"
        assert data["filters"]["days"] == 7

    def test_spend_invalid_granularity_rejected(self, client):
        r = client.get("/spend?granularity=minute")
        assert r.status_code == 422

    def test_spend_empty_is_zero(self, client):
        with patch("main._q", return_value=[]):
            r = client.get("/spend")
        assert r.status_code == 200
        assert r.json()["total_cost_usd"] == 0.0


# ─── API: spend/trace ─────────────────────────────────────────────────────────

class TestAPITraceSpend:
    TRACE_ID = "11111111-1111-1111-1111-111111111111"

    def test_trace_returns_breakdown(self, client):
        with patch("main._q", return_value=SAMPLE_TRACE_SPANS):
            r = client.get(f"/spend/trace/{self.TRACE_ID}")
        assert r.status_code == 200
        data = r.json()
        assert data["trace_id"] == self.TRACE_ID
        assert data["call_count"] == 2
        assert abs(data["total_cost_usd"] - 0.02) < 0.001

    def test_trace_model_breakdown_sorted_descending(self, client):
        with patch("main._q", return_value=SAMPLE_TRACE_SPANS):
            r = client.get(f"/spend/trace/{self.TRACE_ID}")
        breakdown = r.json()["model_breakdown"]
        assert breakdown[0]["model"] == "deepseek-reasoner"
        assert breakdown[0]["cost_usd"] > breakdown[1]["cost_usd"]

    def test_trace_percentages_sum_to_100(self, client):
        with patch("main._q", return_value=SAMPLE_TRACE_SPANS):
            r = client.get(f"/spend/trace/{self.TRACE_ID}")
        pcts = [b["pct"] for b in r.json()["model_breakdown"]]
        assert abs(sum(pcts) - 100.0) < 0.2

    def test_trace_not_found_returns_404(self, client):
        with patch("main._q", return_value=[]):
            r = client.get(f"/spend/trace/{self.TRACE_ID}")
        assert r.status_code == 404


# ─── API: budgets ─────────────────────────────────────────────────────────────

class TestAPIBudgets:

    def test_list_budgets_with_status(self, client):
        with patch("main._q", return_value=SAMPLE_BUDGETS):
            r = client.get("/budgets")
        assert r.status_code == 200
        budgets = r.json()["budgets"]
        assert len(budgets) == 2

        soft = next(b for b in budgets if b["tag_value"] == "openmanagr")
        assert soft["status"] == "alert"     # 84.2% > 80% threshold
        assert soft["pct"] == pytest.approx(84.2, abs=0.5)

        hard = next(b for b in budgets if b["tag_value"] == "engineering")
        assert hard["status"] == "exceeded"  # 100.75%

    def test_create_budget(self, client):
        with patch("main._exec") as mock_exec:
            r = client.post("/budgets", json={
                "tag_key": "project",
                "tag_value": "openmanagr",
                "period": "monthly",
                "limit_usd": 50.0,
                "alert_threshold": 0.80,
                "enforce": False,
            })
        assert r.status_code == 201
        assert r.json()["status"] == "ok"
        mock_exec.assert_called_once()

    def test_create_budget_invalid_period_rejected(self, client):
        r = client.post("/budgets", json={
            "tag_key": "project",
            "tag_value": "test",
            "period": "yearly",    # invalid
            "limit_usd": 100.0,
        })
        assert r.status_code == 422

    def test_delete_budget(self, client):
        with patch("main._exec") as mock_exec:
            r = client.delete("/budgets/1")
        assert r.status_code == 200
        assert r.json()["id"] == 1
        mock_exec.assert_called_once()


# ─── API: budget check ────────────────────────────────────────────────────────

class TestAPIBudgetCheck:

    def _over_budget_row(self, enforce: bool):
        return [{
            "id": 1,
            "tag_key": "project",
            "tag_value": "openmanagr",
            "period": "monthly",
            "limit_usd": 50.0,
            "alert_threshold": 0.80,
            "enforce": enforce,
            "spent_usd": 51.0,
        }]

    def _under_budget_row(self):
        return [{
            "id": 1,
            "tag_key": "project",
            "tag_value": "openmanagr",
            "period": "monthly",
            "limit_usd": 50.0,
            "alert_threshold": 0.80,
            "enforce": True,
            "spent_usd": 10.0,
        }]

    def test_check_ok_when_under_limit(self, client):
        with patch("main._q", return_value=self._under_budget_row()):
            r = client.post("/budgets/check", json={
                "tags": {"project": "openmanagr"},
                "estimated_cost": 0.01,
            })
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_check_returns_402_on_enforce_exceeded(self, client):
        """The critical platform enforcement path — must return 402, not 200."""
        with patch("main._q", return_value=self._over_budget_row(enforce=True)):
            r = client.post("/budgets/check", json={
                "tags": {"project": "openmanagr"},
                "estimated_cost": 0.01,
            })
        assert r.status_code == 402
        detail = r.json()["detail"]
        assert detail["error"] == "budget_exceeded"
        assert detail["tag_value"] == "openmanagr"
        assert detail["spent_usd"] > detail["limit_usd"]

    def test_check_returns_200_with_warning_when_enforce_false(self, client):
        """Soft budget exceeded — warns but does not block (200, not 402)."""
        with patch("main._q", return_value=self._over_budget_row(enforce=False)):
            r = client.post("/budgets/check", json={
                "tags": {"project": "openmanagr"},
            })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert len(data["triggered"]) == 1
        assert data["triggered"][0]["status"] == "exceeded"

    def test_check_empty_tags_returns_ok(self, client):
        r = client.post("/budgets/check", json={"tags": {}})
        assert r.status_code == 200
        assert r.json()["budgets_checked"] == 0

    def test_check_counts_budgets_checked(self, client):
        with patch("main._q", return_value=self._under_budget_row()):
            r = client.post("/budgets/check", json={
                "tags": {"project": "openmanagr"},
            })
        assert r.json()["budgets_checked"] == 1


# ─── API: models ──────────────────────────────────────────────────────────────

class TestAPIModels:

    def test_models_returned(self, client):
        with patch("main._q", return_value=SAMPLE_MODELS):
            r = client.get("/models")
        assert r.status_code == 200
        models = r.json()["models"]
        assert len(models) == 2
        assert models[0]["provider"] == "anthropic"
        assert models[1]["provider"] == "deepseek"


# ─── API: pricing ─────────────────────────────────────────────────────────────

class TestAPIPricing:

    def test_pricing_returned(self, client):
        with patch("main._q", return_value=SAMPLE_PRICING):
            r = client.get("/pricing")
        assert r.status_code == 200
        pricing = r.json()["pricing"]
        assert len(pricing) == 2
        providers = {p["provider"] for p in pricing}
        assert "anthropic" in providers
        assert "deepseek" in providers

    def test_pricing_sync(self, client):
        mock_sync = MagicMock()
        mock_sync.return_value.run.return_value = {"inserted": 3, "already_current": 14}
        with patch("main.PricingSync", mock_sync):
            r = client.post("/pricing/sync")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["inserted"] == 3
        assert data["already_current"] == 14

    def test_add_pricing(self, client):
        mock_sync = MagicMock()
        with patch("main.PricingSync", mock_sync):
            r = client.post("/pricing", json={
                "provider": "deepseek",
                "model": "deepseek-v4",
                "input_rate": 0.30,
                "output_rate": 1.40,
            })
        assert r.status_code == 201
        assert r.json()["model"] == "deepseek/deepseek-v4"


# ─── CLI tests ────────────────────────────────────────────────────────────────

class TestCLIParsing:
    """Test argument parsing — no DB connection needed."""

    def _parse(self, args: list[str]):
        from kostrack.cli import build_parser
        parser = build_parser()
        return parser.parse_args(["--dsn", MOCK_DSN] + args)

    def test_status_command_parsed(self):
        args = self._parse(["status"])
        assert args.command == "status"

    def test_spend_with_filters(self):
        args = self._parse(["spend", "--project", "openmanagr", "--days", "7"])
        assert args.command == "spend"
        assert args.project == "openmanagr"
        assert args.days == 7

    def test_spend_defaults(self):
        args = self._parse(["spend"])
        assert args.days == 30
        assert args.project is None

    def test_budget_set_parsed(self):
        args = self._parse(["budget", "set", "project", "openmanagr", "monthly", "50.0", "--enforce"])
        assert args.budget_command == "set"
        assert args.tag_key == "project"
        assert args.tag_value == "openmanagr"
        assert args.period == "monthly"
        assert args.limit_usd == 50.0
        assert args.enforce is True

    def test_budget_set_defaults(self):
        args = self._parse(["budget", "set", "project", "openmanagr", "monthly", "50.0"])
        assert args.alert == 0.80
        assert args.enforce is False

    def test_budget_period_choices_validated(self):
        from kostrack.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dsn", MOCK_DSN, "budget", "set",
                               "project", "openmanagr", "yearly", "50.0"])

    def test_budget_delete_parsed(self):
        args = self._parse(["budget", "delete", "project", "openmanagr", "monthly"])
        assert args.budget_command == "delete"
        assert args.tag_key == "project"

    def test_pricing_add_parsed(self):
        args = self._parse(["pricing", "add", "deepseek", "deepseek-v4", "0.30", "1.40",
                            "--cache-read", "0.07"])
        assert args.pricing_command == "add"
        assert args.provider == "deepseek"
        assert args.input_rate == 0.30
        assert args.cache_read == 0.07

    def test_traces_with_limit(self):
        args = self._parse(["traces", "--project", "openmanagr", "--limit", "50"])
        assert args.limit == 50

    def test_json_flag(self):
        args = self._parse(["status", "--json"])
        assert args.json is True

    def test_dsn_from_env(self, monkeypatch):
        monkeypatch.setenv("KOSTRACK_DSN", "postgresql://env/db")
        from kostrack.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.dsn == "postgresql://env/db"

    def test_no_command_exits_zero(self):
        from kostrack.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--dsn", MOCK_DSN])
        assert args.command is None


class TestCLIOutput:
    """Test that CLI commands produce correct output when given mock DB data."""

    def _run_cmd(self, args: list[str], mock_rows=None):
        """Run a CLI command with mocked _query, capture stdout.

        --dsn is injected after the subcommand so argparse routes it to
        the correct subparser's namespace (root-level --dsn is shadowed
        by subparser --dsn which defaults to $KOSTRACK_DSN).
        """
        import io
        from contextlib import redirect_stdout
        from kostrack import cli

        mock_rows = mock_rows or []
        buf = io.StringIO()

        # Insert --dsn after the first positional (the subcommand name)
        # so the subparser picks it up correctly.
        if args:
            argv = ["kostrack", args[0], "--dsn", MOCK_DSN] + args[1:]
        else:
            argv = ["kostrack", "--dsn", MOCK_DSN]

        argv_backup = sys.argv
        sys.argv = argv
        try:
            with patch("kostrack.cli._query", return_value=mock_rows), \
                 patch("kostrack.cli._execute"), \
                 patch("kostrack.cli._connect"):
                with redirect_stdout(buf):
                    try:
                        cli.main()
                    except SystemExit:
                        pass
        finally:
            sys.argv = argv_backup

        return buf.getvalue()

    def test_status_shows_total(self):
        # cmd_status makes two _query calls: by-provider then total
        import io
        from contextlib import redirect_stdout
        from kostrack import cli
        from unittest.mock import patch

        buf = io.StringIO()
        sys.argv = ["kostrack", "status", "--dsn", MOCK_DSN]
        total_row = [{"total": 1.1752}]
        with patch("kostrack.cli._query", side_effect=[SAMPLE_SPEND_ROWS, total_row]),              patch("kostrack.cli._execute"), patch("kostrack.cli._connect"):
            with redirect_stdout(buf):
                try:
                    cli.main()
                except SystemExit:
                    pass
        output = buf.getvalue()
        assert "Last 24 hours" in output or "Total" in output or "anthropic" in output.lower()

    def test_status_json_output(self):
        import io
        from contextlib import redirect_stdout
        from kostrack import cli
        from unittest.mock import patch

        buf = io.StringIO()
        sys.argv = ["kostrack", "status", "--json", "--dsn", MOCK_DSN]
        total_row = [{"total": 1.1752}]
        with patch("kostrack.cli._query", side_effect=[SAMPLE_SPEND_ROWS, total_row]),              patch("kostrack.cli._execute"), patch("kostrack.cli._connect"):
            with redirect_stdout(buf):
                try:
                    cli.main()
                except SystemExit:
                    pass
        data = json.loads(buf.getvalue())
        assert "by_provider" in data
        assert "total_24h_usd" in data

    def test_spend_json_output(self):
        output = self._run_cmd(["spend", "--json"], mock_rows=SAMPLE_SPEND_ROWS)
        data = json.loads(output)
        assert isinstance(data, list)
        assert data[0]["provider"] == "anthropic"

    def test_budgets_empty_warning(self):
        output = self._run_cmd(["budgets"], mock_rows=[])
        assert "No budgets" in output or "budget" in output.lower()

    def test_models_json_output(self):
        output = self._run_cmd(["models", "--json"], mock_rows=SAMPLE_MODELS)
        data = json.loads(output)
        assert isinstance(data, list)
        providers = {m["provider"] for m in data}
        assert "deepseek" in providers

    def test_traces_json_output(self):
        output = self._run_cmd(["traces", "--json"], mock_rows=SAMPLE_TRACES)
        data = json.loads(output)
        assert isinstance(data, list)
        assert data[0]["trace_id"] == "11111111-1111-1111-1111-111111111111"

    def test_health_json_output(self):
        output = self._run_cmd(["health", "--json"], mock_rows=SAMPLE_HEALTH)
        data = json.loads(output)
        assert isinstance(data, list)
        assert data[0]["service_id"] == "openmanagr"

    def test_pricing_json_output(self):
        output = self._run_cmd(["pricing", "--json"], mock_rows=SAMPLE_PRICING)
        data = json.loads(output)
        assert isinstance(data, list)
        assert any(p["provider"] == "deepseek" for p in data)

    def test_missing_dsn_exits_with_error(self):
        import io
        from contextlib import redirect_stdout
        from kostrack import cli
        import os

        buf = io.StringIO()
        old = os.environ.pop("KOSTRACK_DSN", None)
        try:
            sys.argv = ["kostrack", "status"]
            with redirect_stdout(buf):
                with pytest.raises(SystemExit) as exc:
                    cli.main()
            assert exc.value.code == 1
        finally:
            if old:
                os.environ["KOSTRACK_DSN"] = old


# ─── End-to-end: SDK + API budget enforcement ─────────────────────────────────

class TestPlatformBudgetEnforcement:
    """
    Simulates the cross-language enforcement pattern:
    Node/Go/Ruby service calls POST /budgets/check before making an LLM call.
    """

    def test_enforcement_flow_402_blocks_call(self, client):
        """
        The critical platform use case: a non-Python service checks budget
        before an LLM call. Exceeded enforce=True budget → 402 → call blocked.
        """
        exceeded_budget = [{
            "id": 1,
            "tag_key": "project",
            "tag_value": "openmanagr",
            "period": "monthly",
            "limit_usd": 10.0,
            "alert_threshold": 0.80,
            "enforce": True,
            "spent_usd": 10.50,
        }]

        with patch("main._q", return_value=exceeded_budget):
            r = client.post("/budgets/check", json={
                "tags": {"project": "openmanagr", "team": "engineering"},
                "estimated_cost": 0.005,
            })

        assert r.status_code == 402
        detail = r.json()["detail"]
        assert detail["error"] == "budget_exceeded"
        assert detail["period"] == "monthly"

    def test_enforcement_flow_200_allows_call(self, client):
        """Under budget → 200 → call proceeds."""
        under_budget = [{
            "id": 1,
            "tag_key": "project",
            "tag_value": "openmanagr",
            "period": "monthly",
            "limit_usd": 100.0,
            "alert_threshold": 0.80,
            "enforce": True,
            "spent_usd": 5.0,
        }]

        with patch("main._q", return_value=under_budget):
            r = client.post("/budgets/check", json={
                "tags": {"project": "openmanagr"},
            })

        assert r.status_code == 200
        assert r.json()["status"] == "ok"
