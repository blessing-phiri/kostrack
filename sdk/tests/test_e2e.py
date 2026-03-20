"""
TokenLedger — End-to-End Test Suite

Tests the full call flow from provider wrapper through to
serialized row output, without requiring a live database.

Run:
    cd sdk
    pip install -e ".[dev]"
    pytest tests/test_e2e.py -v
"""

import json
import sys
import uuid
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_sqlite(tmp_path):
    return tmp_path / "test_buffer.db"


@pytest.fixture()
def mock_writer(tmp_sqlite):
    """
    AsyncBatchWriter with TimescaleDB replaced by an in-memory capture list.
    Never actually connects to Postgres.
    """
    from kostrack.writers.sqlite_queue import SQLiteQueue
    from kostrack.writers.batch_writer import AsyncBatchWriter

    written_rows = []

    writer = AsyncBatchWriter.__new__(AsyncBatchWriter)
    writer.dsn = "postgresql://fake/fake"
    writer.flush_interval = 0.1
    writer.max_batch_size = 100
    writer.service_id = "test"
    writer.fail_open = True
    writer._queue = __import__("queue").Queue()
    writer._sqlite = SQLiteQueue(tmp_sqlite)
    writer._tsdb_ok = True
    writer._conn = None
    writer._lock = threading.Lock()
    writer._stats = {
        "queued": 0, "written_timescale": 0,
        "written_sqlite": 0, "failed": 0, "last_flush": None,
    }
    writer._last_retry = 0.0
    writer._last_backlog_flush = 0.0
    writer._stop_event = threading.Event()

    # Replace DB write with in-memory capture
    def fake_write_to_timescale(batch):
        written_rows.extend(batch)
        writer._stats["written_timescale"] += len(batch)
        return True

    writer._write_to_timescale = fake_write_to_timescale
    writer._update_health_table = lambda: None

    # Start worker
    writer._worker = threading.Thread(
        target=writer._run, name="test-writer", daemon=True
    )
    writer._worker.start()

    writer.written_rows = written_rows
    return writer


@pytest.fixture()
def mock_pricing():
    from kostrack.calculators.pricing_engine import PricingEngine
    from kostrack.models import TokenBreakdown

    pricing = PricingEngine.__new__(PricingEngine)
    pricing._lock = threading.RLock()
    pricing._cache = {}
    pricing._load_bundled()

    def fake_refresh():
        pass
    pricing._load_from_db = fake_refresh
    return pricing


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_rows(writer, count: int, timeout: float = 3.0) -> bool:
    """Wait until at least `count` rows have been written."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(writer.written_rows) >= count:
            return True
        time.sleep(0.05)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class TestModels:

    def test_token_breakdown_total_cached(self):
        from kostrack.models import TokenBreakdown
        tb = TokenBreakdown(input=100, output=200, cache_write=50, cache_read=150)
        assert tb.total_cached == 150

    def test_token_breakdown_jsonb(self):
        from kostrack.models import TokenBreakdown
        tb = TokenBreakdown(input=100, output=200, cache_write=50, cache_read=150, thinking=30)
        j = tb.to_jsonb()
        assert j == {"input": 100, "output": 200, "cache_write": 50, "cache_read": 150, "thinking": 30}

    def test_token_breakdown_omits_zeros(self):
        from kostrack.models import TokenBreakdown
        tb = TokenBreakdown(input=100, output=200)
        j = tb.to_jsonb()
        assert "cache_write" not in j
        assert "cache_read" not in j

    def test_pricing_entry_cost_calculation(self):
        from kostrack.models import PricingEntry, TokenBreakdown
        pe = PricingEntry(
            provider="anthropic", model="claude-sonnet-4-6", pricing_model="per_token",
            input_rate=3/1e6, output_rate=15/1e6,
            cache_write_rate=3.75/1e6, cache_read_rate=0.30/1e6,
        )
        tb = TokenBreakdown(input=1000, output=500, cache_write=200, cache_read=800)
        cost = pe.calculate_cost(tb)
        expected = (1000*3 + 500*15 + 200*3.75 + 800*0.30) / 1e6
        assert abs(cost - expected) < 1e-10

    def test_call_record_to_row(self):
        from kostrack.models import CallRecord, TokenBreakdown
        tb = TokenBreakdown(input=100, output=200)
        record = CallRecord(
            service_id="svc", provider="anthropic", model="claude-sonnet-4-6",
            tokens=tb, cost_usd=0.001234,
            tags={"project": "test", "env": "prod"},
        )
        row = record.to_row()
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 200
        assert row["cost_usd"] == 0.001234
        assert row["tags"]["project"] == "test"
        assert isinstance(row["token_breakdown"], dict)
        assert row["span_id"] is not None

    def test_trace_context_child_span(self):
        from kostrack.models import TraceContext
        root = TraceContext(tags={"project": "x"})
        child = root.child_span()
        assert child.trace_id == root.trace_id
        assert child.parent_span_id == root.span_id
        assert child.span_id != root.span_id
        assert child.tags["project"] == "x"

    def test_trace_context_cost_rollup(self):
        from kostrack.models import TraceContext
        ctx = TraceContext()
        ctx.record_call(0.001)
        ctx.record_call(0.002)
        ctx.record_call(0.0005)
        assert ctx.call_count == 3
        assert abs(ctx.total_cost_usd - 0.0035) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# SQLite Queue
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLiteQueue:

    def test_push_and_pop(self, tmp_sqlite):
        from kostrack.writers.sqlite_queue import SQLiteQueue
        q = SQLiteQueue(tmp_sqlite)
        row = {"time": datetime.now(timezone.utc), "cost_usd": 0.001, "model": "test"}
        q.push(row)
        assert q.size() == 1
        batch = q.pop_batch()
        assert len(batch) == 1
        _, recovered = batch[0]
        assert recovered["model"] == "test"
        q.close()

    def test_ack_removes_records(self, tmp_sqlite):
        from kostrack.writers.sqlite_queue import SQLiteQueue
        q = SQLiteQueue(tmp_sqlite)
        for i in range(5):
            q.push({"time": datetime.now(timezone.utc), "i": i})
        batch = q.pop_batch(size=3)
        ids = [i for i, _ in batch]
        q.ack(ids)
        assert q.size() == 2
        q.close()

    def test_push_batch_atomic(self, tmp_sqlite):
        from kostrack.writers.sqlite_queue import SQLiteQueue
        q = SQLiteQueue(tmp_sqlite)
        rows = [{"time": datetime.now(timezone.utc), "n": i} for i in range(10)]
        q.push_batch(rows)
        assert q.size() == 10
        q.close()

    def test_survives_reopen(self, tmp_sqlite):
        from kostrack.writers.sqlite_queue import SQLiteQueue
        q = SQLiteQueue(tmp_sqlite)
        q.push({"time": datetime.now(timezone.utc), "persist": True})
        q.close()
        q2 = SQLiteQueue(tmp_sqlite)
        assert q2.size() == 1
        q2.close()


# ─────────────────────────────────────────────────────────────────────────────
# Pricing Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestPricingEngine:

    def test_bundled_pricing_covers_all_providers(self, mock_pricing):
        models = mock_pricing.known_models()
        providers = {e[0] for e in mock_pricing._cache}
        assert "anthropic" in providers
        assert "openai" in providers
        assert "gemini" in providers

    def test_known_models_includes_sonnet(self, mock_pricing):
        assert "claude-sonnet-4-6" in mock_pricing.known_models()

    def test_get_cost_anthropic_sonnet(self, mock_pricing):
        from kostrack.models import TokenBreakdown
        tokens = TokenBreakdown(input=1000, output=500)
        cost = mock_pricing.get_cost("anthropic", "claude-sonnet-4-6", tokens)
        expected = (1000 * 3 + 500 * 15) / 1e6
        assert abs(cost - expected) < 1e-10

    def test_get_cost_with_cache(self, mock_pricing):
        from kostrack.models import TokenBreakdown
        tokens = TokenBreakdown(input=1000, output=500, cache_write=200, cache_read=800)
        cost = mock_pricing.get_cost("anthropic", "claude-sonnet-4-6", tokens)
        expected = (1000*3 + 500*15 + 200*3.75 + 800*0.30) / 1e6
        assert abs(cost - expected) < 1e-10

    def test_get_cost_unknown_model_returns_zero(self, mock_pricing):
        from kostrack.models import TokenBreakdown
        tokens = TokenBreakdown(input=100, output=100)
        cost = mock_pricing.get_cost("anthropic", "claude-nonexistent", tokens)
        assert cost == 0.0

    def test_batch_pricing_cheaper_than_per_token(self, mock_pricing):
        from kostrack.models import TokenBreakdown
        tokens = TokenBreakdown(input=1000, output=500)
        per_token = mock_pricing.get_cost("anthropic", "claude-sonnet-4-6", tokens, "per_token")
        batch = mock_pricing.get_cost("anthropic", "claude-sonnet-4-6", tokens, "batch")
        assert batch < per_token
        assert abs(batch - per_token / 2) < 1e-10  # batch = 50% discount


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Calculator
# ─────────────────────────────────────────────────────────────────────────────

class TestAnthropicCalculator:

    def _make_response(self, input=150, output=300, cache_write=0, cache_read=0, model="claude-sonnet-4-6"):
        usage = MagicMock()
        usage.input_tokens = input
        usage.output_tokens = output
        usage.cache_creation_input_tokens = cache_write
        usage.cache_read_input_tokens = cache_read
        resp = MagicMock()
        resp.usage = usage
        resp.model = model
        resp.content = []
        return resp

    def test_basic_extraction(self):
        from kostrack.calculators.anthropic_calc import extract_tokens, extract_model
        resp = self._make_response(150, 300)
        tokens = extract_tokens(resp)
        assert tokens.input == 150
        assert tokens.output == 300
        assert tokens.cache_write == 0
        assert tokens.cache_read == 0

    def test_cache_tokens(self):
        from kostrack.calculators.anthropic_calc import extract_tokens
        resp = self._make_response(150, 300, cache_write=50, cache_read=100)
        tokens = extract_tokens(resp)
        assert tokens.cache_write == 50
        assert tokens.cache_read == 100
        assert tokens.total_cached == 100

    def test_model_extraction(self):
        from kostrack.calculators.anthropic_calc import extract_model
        resp = self._make_response(model="claude-opus-4-6")
        assert extract_model(resp) == "claude-opus-4-6"

    def test_dict_response(self):
        from kostrack.calculators.anthropic_calc import extract_tokens
        resp = {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 400,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 50,
            },
            "model": "claude-sonnet-4-6",
            "content": [],
        }
        tokens = extract_tokens(resp)
        assert tokens.input == 200
        assert tokens.cache_read == 50

    def test_missing_usage_returns_zeros(self):
        from kostrack.calculators.anthropic_calc import extract_tokens
        tokens = extract_tokens({"no_usage": True})
        assert tokens.input == 0
        assert tokens.output == 0


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Calculator
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAICalculator:

    def _make_response(self, prompt=200, completion=400, cached=80, reasoning=0, model="gpt-4o"):
        prompt_details = MagicMock()
        prompt_details.cached_tokens = cached
        completion_details = MagicMock()
        completion_details.reasoning_tokens = reasoning
        usage = MagicMock()
        usage.prompt_tokens = prompt
        usage.completion_tokens = completion
        usage.prompt_tokens_details = prompt_details
        usage.completion_tokens_details = completion_details
        resp = MagicMock()
        resp.usage = usage
        resp.model = model
        return resp

    def test_basic_extraction(self):
        from kostrack.calculators.openai_calc import extract_tokens
        tokens = extract_tokens(self._make_response())
        assert tokens.input == 200
        assert tokens.output == 400

    def test_cached_tokens(self):
        from kostrack.calculators.openai_calc import extract_tokens
        tokens = extract_tokens(self._make_response(cached=80))
        assert tokens.cache_read == 80
        assert tokens.cache_write == 0  # OpenAI has no cache write

    def test_reasoning_tokens(self):
        from kostrack.calculators.openai_calc import extract_tokens
        tokens = extract_tokens(self._make_response(reasoning=150))
        assert tokens.thinking == 150

    def test_cost_includes_cache_discount(self, mock_pricing):
        from kostrack.calculators.openai_calc import extract_tokens
        tokens = extract_tokens(self._make_response(prompt=1000, completion=500, cached=500))
        cost = mock_pricing.get_cost("openai", "gpt-4o", tokens)
        # input: 1000 * 2.5/1e6, output: 500 * 10/1e6, cache_read: 500 * 1.25/1e6
        expected = (1000*2.5 + 500*10 + 500*1.25) / 1e6
        assert abs(cost - expected) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Calculator
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiCalculator:

    def _make_response(self, prompt=300, candidates=150, cached=100, thinking=0):
        usage = MagicMock()
        usage.prompt_token_count = prompt
        usage.candidates_token_count = candidates
        usage.cached_content_token_count = cached
        usage.thoughts_token_count = thinking
        resp = MagicMock()
        resp.usage_metadata = usage
        return resp

    def test_basic_extraction(self):
        from kostrack.calculators.gemini_calc import extract_tokens
        tokens = extract_tokens(self._make_response())
        assert tokens.input == 300
        assert tokens.output == 150
        assert tokens.cache_read == 100

    def test_thinking_tokens(self):
        from kostrack.calculators.gemini_calc import extract_tokens
        tokens = extract_tokens(self._make_response(thinking=75))
        assert tokens.thinking == 75

    def test_missing_usage_metadata(self):
        from kostrack.calculators.gemini_calc import extract_tokens
        tokens = extract_tokens(MagicMock(usage_metadata=None))
        assert tokens.input == 0


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Provider — Full Call Flow
# ─────────────────────────────────────────────────────────────────────────────

class TestAnthropicProvider:

    def _make_sdk_response(self, input=150, output=300, cache_write=0, cache_read=0):
        usage = MagicMock()
        usage.input_tokens = input
        usage.output_tokens = output
        usage.cache_creation_input_tokens = cache_write
        usage.cache_read_input_tokens = cache_read
        resp = MagicMock()
        resp.usage = usage
        resp.model = "claude-sonnet-4-6"
        resp.content = []
        return resp

    def test_basic_call_writes_row(self, mock_writer, mock_pricing):
        from kostrack.providers.anthropic_provider import Anthropic

        with patch("anthropic.Anthropic") as MockClient:
            mock_msgs = MockClient.return_value.messages
            mock_msgs.create.return_value = self._make_sdk_response(150, 300)

            client = Anthropic(
                tags={"project": "test", "feature": "summarise"},
                _writer=mock_writer,
                _pricing=mock_pricing,
            )
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert wait_for_rows(mock_writer, 1)
        row = mock_writer.written_rows[0]
        assert row["provider"] == "anthropic"
        assert row["model"] == "claude-sonnet-4-6"
        assert row["input_tokens"] == 150
        assert row["output_tokens"] == 300
        assert row["cost_usd"] > 0
        assert row["tags"]["project"] == "test"
        assert row["tags"]["feature"] == "summarise"
        assert row["latency_ms"] >= 0

    def test_per_call_tag_override(self, mock_writer, mock_pricing):
        from kostrack.providers.anthropic_provider import Anthropic

        with patch("anthropic.Anthropic") as MockClient:
            mock_msgs = MockClient.return_value.messages
            mock_msgs.create.return_value = self._make_sdk_response()

            client = Anthropic(
                tags={"project": "base"},
                _writer=mock_writer,
                _pricing=mock_pricing,
            )
            client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[],
                kostrack_tags={"feature": "override-feature"},
            )

        assert wait_for_rows(mock_writer, 1)
        row = mock_writer.written_rows[0]
        assert row["tags"]["project"] == "base"
        assert row["tags"]["feature"] == "override-feature"

    def test_trace_context_attached(self, mock_writer, mock_pricing):
        from kostrack.providers.anthropic_provider import Anthropic
        from kostrack.tracing import trace

        with patch("anthropic.Anthropic") as MockClient:
            mock_msgs = MockClient.return_value.messages
            mock_msgs.create.return_value = self._make_sdk_response(100, 200)

            client = Anthropic(_writer=mock_writer, _pricing=mock_pricing)

            with trace(tags={"project": "traced"}) as t:
                client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=512,
                    messages=[],
                )

        assert wait_for_rows(mock_writer, 1)
        row = mock_writer.written_rows[0]
        assert row["trace_id"] == str(t.trace_id)
        assert t.call_count == 1
        assert t.total_cost_usd > 0

    def test_cache_tokens_in_row(self, mock_writer, mock_pricing):
        from kostrack.providers.anthropic_provider import Anthropic

        with patch("anthropic.Anthropic") as MockClient:
            mock_msgs = MockClient.return_value.messages
            mock_msgs.create.return_value = self._make_sdk_response(
                input=200, output=100, cache_write=50, cache_read=150
            )
            client = Anthropic(_writer=mock_writer, _pricing=mock_pricing)
            client.messages.create(model="claude-sonnet-4-6", max_tokens=512, messages=[])

        assert wait_for_rows(mock_writer, 1)
        row = mock_writer.written_rows[0]
        assert row["cached_tokens"] == 150
        assert row["token_breakdown"]["cache_write"] == 50
        assert row["token_breakdown"]["cache_read"] == 150


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Provider — Full Call Flow
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIProvider:

    def _make_sdk_response(self, prompt=200, completion=400, cached=0):
        prompt_details = MagicMock(); prompt_details.cached_tokens = cached
        completion_details = MagicMock(); completion_details.reasoning_tokens = 0
        usage = MagicMock()
        usage.prompt_tokens = prompt
        usage.completion_tokens = completion
        usage.prompt_tokens_details = prompt_details
        usage.completion_tokens_details = completion_details
        resp = MagicMock()
        resp.usage = usage
        resp.model = "gpt-4o"
        return resp

    def test_basic_call_writes_row(self, mock_writer, mock_pricing):
        from kostrack.providers.openai_provider import OpenAI

        with patch("openai.OpenAI") as MockClient:
            mock_completions = MockClient.return_value.chat.completions
            mock_completions.create.return_value = self._make_sdk_response()

            client = OpenAI(
                tags={"project": "test", "feature": "classify"},
                _writer=mock_writer,
                _pricing=mock_pricing,
            )
            client.chat.completions.create(
                model="gpt-4o",
                max_tokens=512,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert wait_for_rows(mock_writer, 1)
        row = mock_writer.written_rows[0]
        assert row["provider"] == "openai"
        assert row["model"] == "gpt-4o"
        assert row["input_tokens"] == 200
        assert row["output_tokens"] == 400
        assert row["cost_usd"] > 0

    def test_cost_calculation_correct(self, mock_writer, mock_pricing):
        from kostrack.providers.openai_provider import OpenAI

        with patch("openai.OpenAI") as MockClient:
            mock_completions = MockClient.return_value.chat.completions
            mock_completions.create.return_value = self._make_sdk_response(1000, 500, cached=0)

            client = OpenAI(_writer=mock_writer, _pricing=mock_pricing)
            client.chat.completions.create(model="gpt-4o", max_tokens=512, messages=[])

        assert wait_for_rows(mock_writer, 1)
        row = mock_writer.written_rows[0]
        expected = (1000 * 2.5 + 500 * 10) / 1e6
        assert abs(row["cost_usd"] - expected) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Tracing — Full Agentic Workflow
# ─────────────────────────────────────────────────────────────────────────────

class TestTracing:

    def test_nested_spans_all_share_trace_id(self, mock_writer, mock_pricing):
        from kostrack.providers.anthropic_provider import Anthropic
        from kostrack.tracing import trace, span

        responses = []
        for _ in range(3):
            usage = MagicMock()
            usage.input_tokens = 100
            usage.output_tokens = 50
            usage.cache_creation_input_tokens = 0
            usage.cache_read_input_tokens = 0
            resp = MagicMock()
            resp.usage = usage
            resp.model = "claude-sonnet-4-6"
            resp.content = []
            responses.append(resp)

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = responses

            client = Anthropic(_writer=mock_writer, _pricing=mock_pricing)

            with trace(tags={"project": "openmanagr", "feature": "month-end"}) as root:
                with span("validate", parent=root):
                    client.messages.create(model="claude-sonnet-4-6", max_tokens=256, messages=[])
                with span("classify", parent=root):
                    client.messages.create(model="claude-sonnet-4-6", max_tokens=256, messages=[])
                with span("post", parent=root):
                    client.messages.create(model="claude-sonnet-4-6", max_tokens=256, messages=[])

        assert wait_for_rows(mock_writer, 3)
        trace_ids = {row["trace_id"] for row in mock_writer.written_rows}
        assert len(trace_ids) == 1  # all same trace
        assert list(trace_ids)[0] == str(root.trace_id)

        # Total cost rolled up correctly
        total = sum(row["cost_usd"] for row in mock_writer.written_rows)
        assert abs(root.total_cost_usd - total) < 1e-10

    def test_trace_outside_block_is_none(self):
        from kostrack.tracing import get_active_trace, trace
        assert get_active_trace() is None
        with trace() as t:
            assert get_active_trace() is t
        assert get_active_trace() is None

    def test_concurrent_traces_isolated(self):
        """Each thread has its own trace stack."""
        from kostrack.tracing import trace, get_active_trace
        import threading

        results = {}

        def run_trace(name):
            with trace(tags={"name": name}) as t:
                time.sleep(0.05)
                results[name] = get_active_trace().tags["name"]

        threads = [threading.Thread(target=run_trace, args=(n,)) for n in ["a", "b", "c"]]
        for t in threads: t.start()
        for t in threads: t.join()

        assert results == {"a": "a", "b": "b", "c": "c"}


# ─────────────────────────────────────────────────────────────────────────────
# Writer — Async Behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncWriter:

    def test_write_never_blocks(self, mock_writer):
        """write() should return near-instantly regardless of flush state."""
        from kostrack.models import CallRecord, TokenBreakdown
        record = CallRecord(provider="anthropic", model="test", tokens=TokenBreakdown(input=10, output=10))

        start = time.monotonic()
        for _ in range(1000):
            mock_writer.write(record.to_row())
        elapsed = time.monotonic() - start

        assert elapsed < 0.5  # 1000 enqueues in < 500ms

    def test_health_reports_stats(self, mock_writer):
        from kostrack.models import CallRecord, TokenBreakdown
        record = CallRecord(provider="openai", model="gpt-4o", tokens=TokenBreakdown(input=50, output=50))
        mock_writer.write(record.to_row())

        wait_for_rows(mock_writer, 1)
        h = mock_writer.health()
        assert "timescale_available" in h
        assert "sqlite_backlog" in h
        assert h["written_timescale"] >= 1

    def test_sqlite_fallback_on_tsdb_failure(self, tmp_sqlite):
        """When TimescaleDB is down, rows go to SQLite."""
        from kostrack.writers.batch_writer import AsyncBatchWriter
        from kostrack.writers.sqlite_queue import SQLiteQueue
        from kostrack.models import CallRecord, TokenBreakdown
        import queue

        writer = AsyncBatchWriter.__new__(AsyncBatchWriter)
        writer.dsn = "postgresql://bad/bad"
        writer.flush_interval = 0.1
        writer.max_batch_size = 100
        writer.service_id = "test-fallback"
        writer.fail_open = True
        writer._queue = queue.Queue()
        writer._sqlite = SQLiteQueue(tmp_sqlite)
        writer._tsdb_ok = False   # simulate DB down from start
        writer._conn = None
        writer._lock = threading.Lock()
        writer._stats = {"queued": 0, "written_timescale": 0, "written_sqlite": 0, "failed": 0, "last_flush": None}
        writer._last_retry = time.monotonic()  # suppress retry
        writer._last_backlog_flush = 0.0
        writer._stop_event = threading.Event()
        writer._update_health_table = lambda: None
        writer._try_connect = lambda: False

        writer._worker = threading.Thread(target=writer._run, daemon=True)
        writer._worker.start()

        record = CallRecord(provider="gemini", model="gemini-2.0-flash", tokens=TokenBreakdown(input=10, output=10))
        writer.write(record.to_row())

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if writer._sqlite.size() > 0:
                break
            time.sleep(0.05)

        assert writer._sqlite.size() > 0
        writer._stop_event.set()
        writer._sqlite.close()
