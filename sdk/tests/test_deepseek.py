"""
kostrack — DeepSeek Test Suite

Tests for:
- DeepSeek cost calculator (V3 and R1 response shapes)
- DeepSeek provider wrapper (basic call, per-call tag override, tag isolation)
- TraceContext.model_costs per-model breakdown
- Multi-model span rollup: cost_breakdown() across DeepSeek + Anthropic

Run:
    cd sdk
    python -m pytest tests/test_deepseek.py -v
"""

import sys
import queue as q
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_writer():
    from kostrack.writers.batch_writer import AsyncBatchWriter

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
    writer._stats = {
        "queued": 0, "written_timescale": 0,
        "written_sqlite": 0, "failed": 0, "last_flush": None,
    }
    writer._last_retry = 0.0
    writer._last_backlog_flush = 0.0
    writer._stop_event = threading.Event()
    writer._write_to_timescale = lambda batch: (written.extend(batch), True)[1]
    writer._update_health_table = lambda: None
    writer._worker = threading.Thread(target=writer._run, daemon=True)
    writer._worker.start()
    writer.written = written
    return writer


def _make_pricing():
    from kostrack.calculators.pricing_engine import PricingEngine

    pricing = PricingEngine.__new__(PricingEngine)
    pricing._lock = threading.RLock()
    pricing._cache = {}
    pricing._load_bundled()
    pricing._load_from_db = lambda: None
    return pricing


def _wait(writer, n, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(writer.written) >= n:
            return True
        time.sleep(0.05)
    return False


def _v3_response(model="deepseek-chat"):
    """Simulate a deepseek-chat (V3) response — no reasoning tokens."""
    usage = MagicMock()
    usage.prompt_tokens = 200
    usage.completion_tokens = 80
    usage.prompt_cache_hit_tokens = 50
    usage.prompt_cache_miss_tokens = 150
    # V3 has no completion_tokens_details
    usage.completion_tokens_details = None
    resp = MagicMock()
    resp.usage = usage
    resp.model = model
    return resp


def _r1_response(model="deepseek-reasoner"):
    """Simulate a deepseek-reasoner (R1) response — includes reasoning tokens."""
    usage = MagicMock()
    usage.prompt_tokens = 300
    usage.completion_tokens = 400   # includes reasoning tokens
    usage.prompt_cache_hit_tokens = 100
    usage.prompt_cache_miss_tokens = 200
    details = MagicMock()
    details.reasoning_tokens = 350
    usage.completion_tokens_details = details
    resp = MagicMock()
    resp.usage = usage
    resp.model = model
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Calculator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepSeekCalculator:

    def test_v3_basic_extraction(self):
        from kostrack.calculators.deepseek_calc import extract_tokens
        tokens = extract_tokens(_v3_response())
        assert tokens.input == 200
        assert tokens.output == 80
        assert tokens.cache_read == 50    # cache hit tokens
        assert tokens.cache_write == 150  # cache miss tokens (full price)
        assert tokens.thinking == 0

    def test_r1_reasoning_tokens(self):
        from kostrack.calculators.deepseek_calc import extract_tokens
        tokens = extract_tokens(_r1_response())
        assert tokens.input == 300
        assert tokens.output == 400
        assert tokens.thinking == 350     # reasoning tokens captured
        assert tokens.cache_read == 100

    def test_v3_no_cache_returns_zero(self):
        from kostrack.calculators.deepseek_calc import extract_tokens
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.prompt_cache_hit_tokens = 0
        usage.prompt_cache_miss_tokens = 0
        usage.completion_tokens_details = None
        resp = MagicMock()
        resp.usage = usage
        resp.model = "deepseek-chat"
        tokens = extract_tokens(resp)
        assert tokens.cache_read == 0
        assert tokens.cache_write == 0
        assert tokens.thinking == 0

    def test_missing_usage_returns_zero_breakdown(self):
        from kostrack.calculators.deepseek_calc import extract_tokens
        resp = MagicMock()
        resp.usage = None
        tokens = extract_tokens(resp)
        assert tokens.input == 0
        assert tokens.output == 0

    def test_extract_model(self):
        from kostrack.calculators.deepseek_calc import extract_model
        assert extract_model(_v3_response()) == "deepseek-chat"
        assert extract_model(_r1_response()) == "deepseek-reasoner"

    def test_dict_response_shape(self):
        """Calculator handles raw dicts (e.g. from httpx)."""
        from kostrack.calculators.deepseek_calc import extract_tokens
        raw = {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "prompt_cache_hit_tokens": 40,
                "prompt_cache_miss_tokens": 80,
            },
            "model": "deepseek-chat",
        }
        tokens = extract_tokens(raw)
        assert tokens.input == 120
        assert tokens.cache_read == 40


# ─────────────────────────────────────────────────────────────────────────────
# Pricing coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepSeekPricing:

    def test_deepseek_in_bundled_pricing(self):
        from kostrack.calculators.pricing_engine import _BUNDLED_PRICING
        providers = {e["provider"] for e in _BUNDLED_PRICING}
        assert "deepseek" in providers

    def test_both_models_in_bundled(self):
        from kostrack.calculators.pricing_engine import _BUNDLED_PRICING
        models = {e["model"] for e in _BUNDLED_PRICING if e["provider"] == "deepseek"}
        assert "deepseek-chat" in models
        assert "deepseek-reasoner" in models

    def test_deepseek_in_current_pricing(self):
        from kostrack.sync.pricing_sync import CURRENT_PRICING
        providers = {e["provider"] for e in CURRENT_PRICING}
        assert "deepseek" in providers

    def test_v3_cost_calculation(self):
        pricing = _make_pricing()
        from kostrack.calculators.deepseek_calc import extract_tokens
        tokens = extract_tokens(_v3_response())
        cost = pricing.get_cost("deepseek", "deepseek-chat", tokens)
        assert cost > 0
        # 200 input @ 0.27/M + 80 output @ 1.10/M + 50 cache_read @ 0.07/M
        expected = 200 * 0.27e-6 + 80 * 1.10e-6 + 50 * 0.07e-6
        assert abs(cost - expected) < 1e-10

    def test_reasoner_costs_more_than_chat(self):
        """R1 is priced higher than V3 at same token counts."""
        pricing = _make_pricing()
        from kostrack.calculators.deepseek_calc import extract_tokens
        from kostrack.models import TokenBreakdown
        tokens = TokenBreakdown(input=1000, output=500)
        chat_cost = pricing.get_cost("deepseek", "deepseek-chat", tokens)
        r1_cost = pricing.get_cost("deepseek", "deepseek-reasoner", tokens)
        assert r1_cost > chat_cost

    def test_reasoner_cheaper_than_claude_sonnet(self):
        """DeepSeek R1 is significantly cheaper than claude-sonnet-4-6."""
        pricing = _make_pricing()
        from kostrack.models import TokenBreakdown
        tokens = TokenBreakdown(input=1000, output=500)
        r1_cost = pricing.get_cost("deepseek", "deepseek-reasoner", tokens)
        sonnet_cost = pricing.get_cost("anthropic", "claude-sonnet-4-6", tokens)
        assert r1_cost < sonnet_cost


# ─────────────────────────────────────────────────────────────────────────────
# Provider tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepSeekProvider:

    def test_basic_call_writes_row(self):
        from kostrack.providers.deepseek_provider import DeepSeek

        writer = _make_writer()
        pricing = _make_pricing()

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = _v3_response()
            client = DeepSeek(
                api_key="sk-test",
                tags={"project": "openmanagr", "feature": "classifier"},
                _writer=writer, _pricing=pricing,
            )
            client.chat.completions.create(
                model="deepseek-chat", max_tokens=256, messages=[],
            )

        assert _wait(writer, 1)
        row = writer.written[0]
        assert row["provider"] == "deepseek"
        assert row["model"] == "deepseek-chat"
        assert row["cost_usd"] > 0
        assert row["tags"]["project"] == "openmanagr"

    def test_r1_call_captures_reasoning_tokens(self):
        from kostrack.providers.deepseek_provider import DeepSeek

        writer = _make_writer()
        pricing = _make_pricing()

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = _r1_response()
            client = DeepSeek(
                api_key="sk-test",
                tags={"project": "openmanagr"},
                _writer=writer, _pricing=pricing,
            )
            client.chat.completions.create(
                model="deepseek-reasoner", max_tokens=1024, messages=[],
            )

        assert _wait(writer, 1)
        row = writer.written[0]
        assert row["provider"] == "deepseek"
        assert row["model"] == "deepseek-reasoner"
        assert row["token_breakdown"]["thinking"] == 350

    def test_per_call_tag_override(self):
        from kostrack.providers.deepseek_provider import DeepSeek

        writer = _make_writer()
        pricing = _make_pricing()

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = _v3_response()
            client = DeepSeek(
                api_key="sk-test",
                tags={"project": "openmanagr", "feature": "base"},
                _writer=writer, _pricing=pricing,
            )
            client.chat.completions.create(
                model="deepseek-chat", max_tokens=128, messages=[],
                kostrack_tags={"feature": "override", "ab_test": "v2"},
            )

        assert _wait(writer, 1)
        row = writer.written[0]
        assert row["tags"]["project"] == "openmanagr"
        assert row["tags"]["feature"] == "override"
        assert row["tags"]["ab_test"] == "v2"

    def test_tags_not_leaked_between_calls(self):
        from kostrack.providers.deepseek_provider import DeepSeek

        writer = _make_writer()
        pricing = _make_pricing()

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = [
                _v3_response(), _v3_response(),
            ]
            client = DeepSeek(
                api_key="sk-test",
                tags={"project": "openmanagr", "feature": "base"},
                _writer=writer, _pricing=pricing,
            )
            client.chat.completions.create(
                model="deepseek-chat", max_tokens=128, messages=[],
                kostrack_tags={"feature": "override"},
            )
            client.chat.completions.create(
                model="deepseek-chat", max_tokens=128, messages=[],
            )

        assert _wait(writer, 2)
        assert writer.written[0]["tags"]["feature"] == "override"
        assert writer.written[1]["tags"]["feature"] == "base"

    def test_missing_api_key_raises(self):
        from kostrack.providers.deepseek_provider import DeepSeek
        import os
        env_backup = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key"):
                DeepSeek(tags={})
        finally:
            if env_backup:
                os.environ["DEEPSEEK_API_KEY"] = env_backup

    def test_base_url_points_to_deepseek(self):
        """Verify the openai client is instantiated with DeepSeek's base URL."""
        from kostrack.providers.deepseek_provider import DeepSeek, DEEPSEEK_BASE_URL

        writer = _make_writer()
        pricing = _make_pricing()

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = _v3_response()
            DeepSeek(api_key="sk-test", _writer=writer, _pricing=pricing)
            call_kwargs = MockClient.call_args[1]
            assert call_kwargs["base_url"] == DEEPSEEK_BASE_URL


# ─────────────────────────────────────────────────────────────────────────────
# TraceContext.model_costs
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCostsTracking:

    def test_single_model_cost_recorded(self):
        from kostrack.models import TraceContext
        ctx = TraceContext()
        ctx.record_call(0.001, model="deepseek-chat")
        assert ctx.model_costs["deepseek-chat"] == pytest.approx(0.001)
        assert ctx.total_cost_usd == pytest.approx(0.001)
        assert ctx.call_count == 1

    def test_multi_model_costs_accumulate(self):
        from kostrack.models import TraceContext
        ctx = TraceContext()
        ctx.record_call(0.002, model="deepseek-reasoner")
        ctx.record_call(0.003, model="claude-sonnet-4-6")
        ctx.record_call(0.001, model="deepseek-reasoner")
        assert ctx.model_costs["deepseek-reasoner"] == pytest.approx(0.003)
        assert ctx.model_costs["claude-sonnet-4-6"] == pytest.approx(0.003)
        assert ctx.total_cost_usd == pytest.approx(0.006)
        assert ctx.call_count == 3

    def test_record_call_without_model_still_works(self):
        """Backward-compat — model is optional."""
        from kostrack.models import TraceContext
        ctx = TraceContext()
        ctx.record_call(0.005)
        assert ctx.total_cost_usd == pytest.approx(0.005)
        assert ctx.model_costs == {}

    def test_cost_breakdown_sorted_by_spend(self):
        from kostrack.models import TraceContext
        ctx = TraceContext()
        ctx.record_call(0.001, model="gemini-2.0-flash")
        ctx.record_call(0.010, model="claude-sonnet-4-6")
        ctx.record_call(0.004, model="deepseek-reasoner")
        breakdown = ctx.cost_breakdown()
        assert breakdown[0]["model"] == "claude-sonnet-4-6"
        assert breakdown[1]["model"] == "deepseek-reasoner"
        assert breakdown[2]["model"] == "gemini-2.0-flash"

    def test_cost_breakdown_percentages_sum_to_100(self):
        from kostrack.models import TraceContext
        ctx = TraceContext()
        ctx.record_call(0.006, model="deepseek-reasoner")
        ctx.record_call(0.004, model="claude-haiku-4-5-20251001")
        breakdown = ctx.cost_breakdown()
        total_pct = sum(b["pct"] for b in breakdown)
        assert abs(total_pct - 100.0) < 0.2  # rounding tolerance

    def test_cost_breakdown_empty_when_no_calls(self):
        from kostrack.models import TraceContext
        ctx = TraceContext()
        assert ctx.cost_breakdown() == []

    def test_child_span_inherits_model_costs(self):
        """child_span() starts with empty model_costs, not a copy of parent's."""
        from kostrack.models import TraceContext
        parent = TraceContext()
        parent.record_call(0.005, model="claude-sonnet-4-6")
        child = parent.child_span()
        # Child starts fresh — no inherited model_costs
        assert child.model_costs == {}
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id


# ─────────────────────────────────────────────────────────────────────────────
# Span rollup — multi-model cost_breakdown()
# ─────────────────────────────────────────────────────────────────────────────

class TestSpanRollup:

    def test_model_costs_roll_up_from_span_to_trace(self):
        """
        Span's model_costs must merge into the parent trace after the span exits.
        """
        import kostrack
        from kostrack.tracing import trace, span

        with trace(tags={"project": "openmanagr"}) as root:
            with span("classify", parent=root) as s:
                s.record_call(0.003, model="deepseek-reasoner")
            with span("extract", parent=root) as s2:
                s2.record_call(0.002, model="claude-haiku-4-5-20251001")

        assert root.model_costs["deepseek-reasoner"] == pytest.approx(0.003)
        assert root.model_costs["claude-haiku-4-5-20251001"] == pytest.approx(0.002)
        assert root.total_cost_usd == pytest.approx(0.005)

    def test_call_count_rolls_up_correctly(self):
        from kostrack.tracing import trace, span

        with trace(tags={"feature": "test"}) as root:
            with span("step-a", parent=root) as s:
                s.record_call(0.001, model="deepseek-chat")
                s.record_call(0.001, model="deepseek-chat")
            with span("step-b", parent=root) as s2:
                s2.record_call(0.001, model="gpt-4o")

        assert root.call_count == 3
        assert root.model_costs["deepseek-chat"] == pytest.approx(0.002)
        assert root.model_costs["gpt-4o"] == pytest.approx(0.001)

    def test_same_model_across_spans_accumulates(self):
        """Two spans both call deepseek-chat — costs should sum on the root."""
        from kostrack.tracing import trace, span

        with trace(tags={}) as root:
            with span("a", parent=root) as s1:
                s1.record_call(0.004, model="deepseek-chat")
            with span("b", parent=root) as s2:
                s2.record_call(0.006, model="deepseek-chat")

        assert root.model_costs["deepseek-chat"] == pytest.approx(0.010)
        assert root.total_cost_usd == pytest.approx(0.010)

    def test_cost_breakdown_on_multi_model_trace(self):
        from kostrack.tracing import trace, span

        with trace(tags={"project": "openmanagr", "feature": "month-end"}) as root:
            with span("classify", parent=root) as s:
                s.record_call(0.0142, model="deepseek-reasoner")
            with span("extract", parent=root) as s2:
                s2.record_call(0.0038, model="claude-haiku-4-5-20251001")
            with span("post", parent=root) as s3:
                s3.record_call(0.0020, model="claude-sonnet-4-6")

        breakdown = root.cost_breakdown()
        assert breakdown[0]["model"] == "deepseek-reasoner"
        assert breakdown[0]["pct"] == pytest.approx(71.0, abs=0.5)
        assert len(breakdown) == 3
        assert root.call_count == 3

    def test_nested_spans_roll_up_transitively(self):
        """Grandchild span costs propagate through child to root."""
        from kostrack.tracing import trace, span

        with trace(tags={}) as root:
            with span("parent-span", parent=root) as mid:
                with span("child-span", parent=mid) as leaf:
                    leaf.record_call(0.005, model="gemini-2.0-flash")
                # After leaf exits, mid should have gemini cost
                assert mid.model_costs.get("gemini-2.0-flash", 0) == pytest.approx(0.005)

        # After mid exits, root should also have it
        assert root.model_costs.get("gemini-2.0-flash", 0) == pytest.approx(0.005)
        assert root.total_cost_usd == pytest.approx(0.005)
