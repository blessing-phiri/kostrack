"""
kostrack — Phase 2 Test Suite

Tests for:
- PricingSync: current pricing table, insert logic, upsert
- BudgetEnforcer: cache, check, BudgetExceededError, guard context manager
- SDK configure() with sync_pricing and budget_enforcer options
- Provider kostrack_tags per-call override

Run:
    cd sdk
    python -m pytest tests/test_phase2.py -v
"""

import sys
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# PricingSync
# ─────────────────────────────────────────────────────────────────────────────

class TestPricingSync:

    def test_current_pricing_covers_all_providers(self):
        from kostrack.sync.pricing_sync import CURRENT_PRICING
        providers = {e["provider"] for e in CURRENT_PRICING}
        assert "anthropic" in providers
        assert "openai" in providers
        assert "gemini" in providers

    def test_current_pricing_has_no_missing_rates(self):
        from kostrack.sync.pricing_sync import CURRENT_PRICING
        for entry in CURRENT_PRICING:
            assert entry["input_rate"] is not None, f"Missing input_rate: {entry}"
            assert entry["output_rate"] is not None, f"Missing output_rate: {entry}"
            assert entry["pricing_model"] in ("per_token", "batch"), \
                f"Unknown pricing_model: {entry['pricing_model']}"

    def test_batch_pricing_cheaper_than_per_token(self):
        from kostrack.sync.pricing_sync import CURRENT_PRICING
        by_key = {}
        for e in CURRENT_PRICING:
            key = (e["provider"], e["model"])
            by_key.setdefault(key, {})[e["pricing_model"]] = e

        for key, variants in by_key.items():
            if "per_token" in variants and "batch" in variants:
                pt = variants["per_token"]
                ba = variants["batch"]
                assert ba["input_rate"] < pt["input_rate"], \
                    f"Batch input not cheaper for {key}"
                assert ba["output_rate"] < pt["output_rate"], \
                    f"Batch output not cheaper for {key}"

    def test_all_anthropic_models_have_cache_rates(self):
        from kostrack.sync.pricing_sync import CURRENT_PRICING
        for e in CURRENT_PRICING:
            if e["provider"] == "anthropic" and e["pricing_model"] == "per_token":
                assert e.get("cache_read_rate") is not None, \
                    f"Anthropic per_token missing cache_read_rate: {e['model']}"

    def test_sync_run_with_mock_db(self):
        """PricingSync._sync inserts all entries when DB is empty."""
        from kostrack.sync.pricing_sync import PricingSync, CURRENT_PRICING

        # Mock connection that returns empty active pricing
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = []  # empty DB
        mock_conn.cursor.return_value = mock_cur

        sync = PricingSync.__new__(PricingSync)
        sync.dsn = "mock"
        result = sync._sync(mock_conn)

        assert result["inserted"] == len(CURRENT_PRICING)
        assert result["already_current"] == 0

    def test_sync_skips_unchanged_entries(self):
        """PricingSync._sync skips entries already in DB with same rates."""
        from kostrack.sync.pricing_sync import PricingSync, CURRENT_PRICING

        # Simulate all entries already present with correct rates
        existing = {
            (e["provider"], e["model"], e["pricing_model"]): {
                "provider": e["provider"],
                "model": e["model"],
                "pricing_model": e["pricing_model"],
                "input_rate": e["input_rate"],
                "output_rate": e["output_rate"],
            }
            for e in CURRENT_PRICING
        }

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [
            {"provider": v["provider"], "model": v["model"],
             "pricing_model": v["pricing_model"],
             "input_rate": v["input_rate"], "output_rate": v["output_rate"]}
            for v in existing.values()
        ]
        mock_conn.cursor.return_value = mock_cur

        sync = PricingSync.__new__(PricingSync)
        sync.dsn = "mock"
        result = sync._sync(mock_conn)

        assert result["inserted"] == 0
        assert result["already_current"] == len(CURRENT_PRICING)

    def test_pricing_entry_count(self):
        from kostrack.sync.pricing_sync import CURRENT_PRICING
        # Should have entries for all 3 providers with per_token + batch where applicable
        per_token = [e for e in CURRENT_PRICING if e["pricing_model"] == "per_token"]
        batch = [e for e in CURRENT_PRICING if e["pricing_model"] == "batch"]
        assert len(per_token) >= 10, f"Expected >=10 per_token entries, got {len(per_token)}"
        assert len(batch) >= 4, f"Expected >=4 batch entries, got {len(batch)}"


# ─────────────────────────────────────────────────────────────────────────────
# BudgetEnforcer
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetEnforcer:

    def _make_enforcer(self, budgets: list[dict]) -> "BudgetEnforcer":
        """Create a BudgetEnforcer with pre-populated cache, no DB needed."""
        from kostrack.budget import BudgetEnforcer
        enforcer = BudgetEnforcer.__new__(BudgetEnforcer)
        enforcer.dsn = "mock"
        enforcer.cache_ttl = 9999  # never expires during test
        enforcer._lock = threading.RLock()
        enforcer._last_refresh = time.monotonic()

        enforcer._cache = {}
        for b in budgets:
            key = f"{b['tag_key']}:{b['tag_value']}:{b['period']}"
            enforcer._cache[key] = b

        return enforcer

    def test_no_budgets_returns_empty(self):
        enforcer = self._make_enforcer([])
        result = enforcer.check({"project": "openmanagr"})
        assert result == []

    def test_under_threshold_no_warning(self):
        enforcer = self._make_enforcer([{
            "tag_key": "project", "tag_value": "openmanagr",
            "period": "monthly", "limit_usd": 100.0,
            "alert_threshold": 0.80, "enforce": False, "spent": 50.0,
        }])
        result = enforcer.check({"project": "openmanagr"}, estimated_cost=0.01)
        assert len(result) == 1
        assert result[0]["pct"] < 0.80

    def test_over_alert_threshold_warning(self, caplog):
        import logging
        enforcer = self._make_enforcer([{
            "tag_key": "project", "tag_value": "openmanagr",
            "period": "monthly", "limit_usd": 100.0,
            "alert_threshold": 0.80, "enforce": False, "spent": 82.0,
        }])
        with caplog.at_level(logging.WARNING, logger="kostrack.budget"):
            result = enforcer.check({"project": "openmanagr"})
        assert len(result) == 1
        assert result[0]["pct"] >= 0.80
        assert "Budget alert" in caplog.text

    def test_enforce_raises_when_exceeded(self):
        from kostrack.budget import BudgetExceededError
        enforcer = self._make_enforcer([{
            "tag_key": "project", "tag_value": "openmanagr",
            "period": "monthly", "limit_usd": 100.0,
            "alert_threshold": 0.80, "enforce": True, "spent": 101.0,
        }])
        with pytest.raises(BudgetExceededError) as exc_info:
            enforcer.check({"project": "openmanagr"})
        assert exc_info.value.tag_value == "openmanagr"
        assert exc_info.value.spent >= 100.0

    def test_enforce_false_does_not_raise_when_exceeded(self):
        enforcer = self._make_enforcer([{
            "tag_key": "project", "tag_value": "openmanagr",
            "period": "monthly", "limit_usd": 100.0,
            "alert_threshold": 0.80, "enforce": False, "spent": 105.0,
        }])
        # Should not raise even though over limit
        result = enforcer.check({"project": "openmanagr"})
        assert len(result) == 1
        assert result[0]["pct"] > 1.0

    def test_unmatched_tags_not_checked(self):
        from kostrack.budget import BudgetExceededError
        enforcer = self._make_enforcer([{
            "tag_key": "project", "tag_value": "other-project",
            "period": "monthly", "limit_usd": 1.0,
            "alert_threshold": 0.80, "enforce": True, "spent": 999.0,
        }])
        # openmanagr tags should not trigger other-project budget
        result = enforcer.check({"project": "openmanagr"})
        assert result == []

    def test_guard_context_manager_passes(self):
        enforcer = self._make_enforcer([{
            "tag_key": "project", "tag_value": "openmanagr",
            "period": "monthly", "limit_usd": 100.0,
            "alert_threshold": 0.80, "enforce": True, "spent": 10.0,
        }])
        executed = False
        with enforcer.guard({"project": "openmanagr"}):
            executed = True
        assert executed

    def test_guard_context_manager_raises_before_block(self):
        from kostrack.budget import BudgetExceededError
        enforcer = self._make_enforcer([{
            "tag_key": "project", "tag_value": "openmanagr",
            "period": "monthly", "limit_usd": 100.0,
            "alert_threshold": 0.80, "enforce": True, "spent": 105.0,
        }])
        executed = False
        with pytest.raises(BudgetExceededError):
            with enforcer.guard({"project": "openmanagr"}):
                executed = True  # should never reach here
        assert not executed

    def test_multiple_budgets_all_checked(self):
        from kostrack.budget import BudgetExceededError
        enforcer = self._make_enforcer([
            {"tag_key": "project", "tag_value": "openmanagr",
             "period": "monthly", "limit_usd": 100.0,
             "alert_threshold": 0.80, "enforce": False, "spent": 85.0},
            {"tag_key": "team", "tag_value": "engineering",
             "period": "monthly", "limit_usd": 200.0,
             "alert_threshold": 0.80, "enforce": True, "spent": 199.0},
        ])
        # Both tags match — team budget is over threshold but not at limit
        result = enforcer.check({"project": "openmanagr", "team": "engineering"})
        assert len(result) == 2

    def test_get_status_returns_all_budgets(self):
        budgets = [
            {"tag_key": "project", "tag_value": "a", "period": "monthly",
             "limit_usd": 100.0, "alert_threshold": 0.80, "enforce": False, "spent": 10.0},
            {"tag_key": "project", "tag_value": "b", "period": "monthly",
             "limit_usd": 200.0, "alert_threshold": 0.80, "enforce": False, "spent": 20.0},
        ]
        enforcer = self._make_enforcer(budgets)
        status = enforcer.get_status()
        assert len(status) == 2

    def test_budget_exceeded_error_message(self):
        from kostrack.budget import BudgetExceededError
        err = BudgetExceededError("project", "openmanagr", "monthly", 105.0, 100.0)
        assert "openmanagr" in str(err)
        assert "monthly" in str(err)
        assert "105" in str(err)

    def test_cache_ttl_triggers_refresh(self):
        """When cache is stale, _maybe_refresh is called."""
        from kostrack.budget import BudgetEnforcer
        enforcer = BudgetEnforcer.__new__(BudgetEnforcer)
        enforcer.dsn = "mock"
        enforcer.cache_ttl = 0  # always stale
        enforcer._lock = threading.RLock()
        enforcer._cache = {}
        enforcer._last_refresh = 0.0

        refresh_called = []
        def mock_refresh():
            refresh_called.append(True)

        enforcer._refresh = mock_refresh
        enforcer._maybe_refresh()
        assert len(refresh_called) == 1


# ─────────────────────────────────────────────────────────────────────────────
# kostrack_tags per-call override
# ─────────────────────────────────────────────────────────────────────────────

class TestPerCallTagOverride:

    def _make_mock_writer(self):
        from kostrack.writers.sqlite_queue import SQLiteQueue
        from kostrack.writers.batch_writer import AsyncBatchWriter
        import queue as q

        written = []
        writer = AsyncBatchWriter.__new__(AsyncBatchWriter)
        writer.dsn = "mock"
        writer.flush_interval = 0.1
        writer.max_batch_size = 100
        writer.service_id = "test"
        writer.fail_open = True
        writer._queue = q.Queue()
        writer._sqlite = MagicMock()
        writer._sqlite.size.return_value = 0
        writer._tsdb_ok = True
        writer._conn = None
        writer._lock = threading.Lock()
        writer._stats = {"queued": 0, "written_timescale": 0, "written_sqlite": 0, "failed": 0, "last_flush": None}
        writer._last_retry = 0.0
        writer._last_backlog_flush = 0.0
        writer._stop_event = threading.Event()
        writer._write_to_timescale = lambda batch: (written.extend(batch), True)[1]
        writer._update_health_table = lambda: None
        writer._worker = threading.Thread(target=writer._run, daemon=True)
        writer._worker.start()
        writer.written = written
        return writer

    def _make_mock_pricing(self):
        from kostrack.calculators.pricing_engine import PricingEngine
        pricing = PricingEngine.__new__(PricingEngine)
        pricing._lock = threading.RLock()
        pricing._cache = {}
        pricing._load_bundled()
        pricing._load_from_db = lambda: None
        return pricing

    def _wait(self, writer, n, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(writer.written) >= n:
                return True
            time.sleep(0.05)
        return False

    def _make_response(self):
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cache_creation_input_tokens = 0
        usage.cache_read_input_tokens = 0
        resp = MagicMock()
        resp.usage = usage
        resp.model = "claude-sonnet-4-6"
        resp.content = []
        return resp

    def test_anthropic_per_call_tag_override(self):
        from kostrack.providers.anthropic_provider import Anthropic

        writer = self._make_mock_writer()
        pricing = self._make_mock_pricing()

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = self._make_response()
            client = Anthropic(
                tags={"project": "base-project", "feature": "base-feature"},
                _writer=writer, _pricing=pricing,
            )
            client.messages.create(
                model="claude-sonnet-4-6", max_tokens=256, messages=[],
                kostrack_tags={"feature": "override-feature", "ab_test": "variant-b"},
            )

        assert self._wait(writer, 1)
        row = writer.written[0]
        # Base tag preserved, feature overridden, new key added
        assert row["tags"]["project"] == "base-project"
        assert row["tags"]["feature"] == "override-feature"
        assert row["tags"]["ab_test"] == "variant-b"

    def test_openai_per_call_tag_override(self):
        from kostrack.providers.openai_provider import OpenAI

        writer = self._make_mock_writer()
        pricing = self._make_mock_pricing()

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.prompt_tokens_details = MagicMock(cached_tokens=0)
        usage.completion_tokens_details = MagicMock(reasoning_tokens=0)
        resp = MagicMock()
        resp.usage = usage
        resp.model = "gpt-4o"

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = resp
            client = OpenAI(
                tags={"project": "base-project"},
                _writer=writer, _pricing=pricing,
            )
            client.chat.completions.create(
                model="gpt-4o", max_tokens=256, messages=[],
                kostrack_tags={"feature": "openai-feature"},
            )

        assert self._wait(writer, 1)
        row = writer.written[0]
        assert row["tags"]["project"] == "base-project"
        assert row["tags"]["feature"] == "openai-feature"

    def test_tags_not_leaked_between_calls(self):
        """Per-call tags should not persist to subsequent calls."""
        from kostrack.providers.anthropic_provider import Anthropic

        writer = self._make_mock_writer()
        pricing = self._make_mock_pricing()

        responses = [self._make_response(), self._make_response()]

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = responses
            client = Anthropic(
                tags={"project": "myapp", "feature": "base"},
                _writer=writer, _pricing=pricing,
            )
            # First call with override
            client.messages.create(
                model="claude-sonnet-4-6", max_tokens=256, messages=[],
                kostrack_tags={"feature": "override"},
            )
            # Second call without override
            client.messages.create(
                model="claude-sonnet-4-6", max_tokens=256, messages=[],
            )

        assert self._wait(writer, 2)
        assert writer.written[0]["tags"]["feature"] == "override"
        assert writer.written[1]["tags"]["feature"] == "base"


# ─────────────────────────────────────────────────────────────────────────────
# PricingSync × BudgetEnforcer integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2Integration:

    def test_budget_enforcer_with_anthropic_provider(self):
        """BudgetEnforcer.guard() blocks a call when limit is exceeded."""
        from kostrack.budget import BudgetEnforcer, BudgetExceededError
        from kostrack.providers.anthropic_provider import Anthropic
        import threading

        # Build enforcer with exceeded budget
        enforcer = BudgetEnforcer.__new__(BudgetEnforcer)
        enforcer.dsn = "mock"
        enforcer.cache_ttl = 9999
        enforcer._lock = threading.RLock()
        enforcer._last_refresh = time.monotonic()
        enforcer._cache = {
            "project:openmanagr:monthly": {
                "tag_key": "project", "tag_value": "openmanagr",
                "period": "monthly", "limit_usd": 10.0,
                "alert_threshold": 0.80, "enforce": True, "spent": 10.50,
            }
        }

        tags = {"project": "openmanagr", "feature": "invoice-extraction"}

        # Guard should raise before the LLM call is made
        with pytest.raises(BudgetExceededError):
            with enforcer.guard(tags):
                pytest.fail("Should not reach here")

    def test_budget_enforcer_allows_call_under_limit(self):
        from kostrack.budget import BudgetEnforcer
        import threading

        enforcer = BudgetEnforcer.__new__(BudgetEnforcer)
        enforcer.dsn = "mock"
        enforcer.cache_ttl = 9999
        enforcer._lock = threading.RLock()
        enforcer._last_refresh = time.monotonic()
        enforcer._cache = {
            "project:openmanagr:monthly": {
                "tag_key": "project", "tag_value": "openmanagr",
                "period": "monthly", "limit_usd": 100.0,
                "alert_threshold": 0.80, "enforce": True, "spent": 5.0,
            }
        }

        executed = False
        with enforcer.guard({"project": "openmanagr"}):
            executed = True
        assert executed

    def test_pricing_sync_and_engine_consistency(self):
        """Every model in CURRENT_PRICING should be in the bundled pricing engine."""
        from kostrack.sync.pricing_sync import CURRENT_PRICING
        from kostrack.calculators.pricing_engine import _BUNDLED_PRICING

        sync_keys = {(e["provider"], e["model"], e["pricing_model"]) for e in CURRENT_PRICING}
        bundled_keys = {(e["provider"], e["model"], e["pricing_model"]) for e in _BUNDLED_PRICING}

        # All bundled entries should be in sync pricing
        missing = bundled_keys - sync_keys
        assert not missing, f"Models in bundled but not in sync: {missing}"

    def test_budget_exceeded_error_attributes(self):
        from kostrack.budget import BudgetExceededError
        err = BudgetExceededError(
            tag_key="project",
            tag_value="openmanagr",
            period="monthly",
            spent=105.50,
            limit=100.0,
        )
        assert err.tag_key == "project"
        assert err.tag_value == "openmanagr"
        assert err.period == "monthly"
        assert err.spent == 105.50
        assert err.limit == 100.0
        assert isinstance(err, Exception)
