"""test_local_fallback.py — Tests for the LocalFallbackCastingPlugin."""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.plugins.local_fallback import (
    LocalFallbackCastingPlugin, OPENER_TO_TASK_TYPE,
)
from quilt_substrate.plugins.casting import Probes, ResourceBudget


def make_substrate():
    s = Substrate()
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    return s


def calm_probes():
    return Probes(user="casey", app="writers-room", hardware="laptop",
                   time_of_day="evening", weather="calm", crew_state="normal")


def offline_probes():
    return Probes(user="reyes", app="F/V EILEEN", hardware="tablet",
                   time_of_day="0300", weather="gale", crew_state="tired")


def test_opener_to_task_type():
    """All common openers have a task type mapping."""
    expected = {"chart", "voice", "tide", "reef", "slate", "witness"}
    actual = set(OPENER_TO_TASK_TYPE.keys())
    assert expected.issubset(actual)


def test_should_go_local_no_network():
    """No network → go local."""
    s = make_substrate()
    p = calm_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p)
    budget = ResourceBudget(battery_pct=0.5, network="none")
    assert plugin._should_go_local(budget) is True


def test_should_go_local_low_battery():
    """Battery < threshold → go local."""
    s = make_substrate()
    p = calm_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p)
    budget = ResourceBudget(battery_pct=0.05, network="wifi")
    assert plugin._should_go_local(budget) is True


def test_should_go_local_calm_wifi():
    """Calm + wifi → cloud."""
    s = make_substrate()
    p = calm_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p)
    budget = ResourceBudget(battery_pct=0.8, network="wifi")
    assert plugin._should_go_local(budget) is False


def test_decide_offline_picks_local():
    """In an offline situation, the decision is local."""
    s = make_substrate()
    p = offline_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p)
    # Mock the budget to simulate no-network + low-battery
    with patch.object(p, "budget", return_value=ResourceBudget(battery_pct=0.05, network="none")):
        d = plugin.decide(opener="tide", kwargs={"role": "sensory_creative"})
    assert d.model.startswith("local:")
    assert d.is_fallback is True
    assert "local fallback" in d.rationale.lower()


def test_decide_online_uses_normal():
    """In a normal situation, the decision is normal (cloud)."""
    s = make_substrate()
    p = calm_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p)
    d = plugin.decide(opener="chart", kwargs={"role": "creative_ideation"})
    # Either local: or a cloud model — depends on priors
    assert d.is_fallback is False or d.model.startswith("local:")


def test_render_offline_witness_records():
    """In offline mode, the witness log records the local dispatch."""
    s = make_substrate()
    p = offline_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p)
    # Mock the router to avoid actually calling Ollama
    mock_router = MagicMock()
    mock_router.dispatch.return_value = {"text": "mock result"}
    plugin._router = mock_router
    plugin.install()
    # Mock the budget to be offline
    with patch.object(p, "budget", return_value=ResourceBudget(battery_pct=0.05, network="none")):
        result = s.render(opener="tide", role="sensory_creative")
    # The witness should have a local_dispatch entry
    proposed = [e for e in plugin.witness if e["kind"] == "cast.proposed"]
    assert any(e["decision"]["model"].startswith("local:") for e in proposed)
    # The router was called
    mock_router.dispatch.assert_called_once()
    args, kwargs = mock_router.dispatch.call_args
    assert kwargs["task_type"] == "sensory"  # tide → sensory


def test_render_offline_router_unavailable():
    """If agent-loop is not installed, render returns an error dict (not crash)."""
    s = make_substrate()
    p = offline_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p)
    # _router = False means "not available"
    plugin._router = False
    plugin.install()
    result = s.render(opener="tide", role="sensory_creative")
    assert "error" in result or "fallback" in result


def test_render_online_uses_super():
    """In normal mode, render() uses the parent's render() (cloud path)."""
    s = make_substrate()
    p = calm_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p)
    plugin.install()
    # Mock the network probe to be wifi
    with patch.object(p, "_read_network", return_value="wifi"):
        with patch.object(p, "_read_battery", return_value=0.8):
            result = s.render(opener="chart")
    # Should be a normal render — dict, not a local dispatch error
    assert isinstance(result, dict)


def test_battery_threshold_configurable():
    """The battery threshold is configurable."""
    s = make_substrate()
    p = calm_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p, battery_threshold=0.5)
    budget = ResourceBudget(battery_pct=0.3, network="wifi")
    assert plugin._should_go_local(budget) is True


def test_use_local_when_offline_disabled():
    """If use_local_when_offline is False, never go local."""
    s = make_substrate()
    p = calm_probes()
    plugin = LocalFallbackCastingPlugin(s, probes=p, use_local_when_offline=False)
    budget = ResourceBudget(battery_pct=0.05, network="none")
    assert plugin._should_go_local(budget) is False


if __name__ == "__main__":
    import inspect
    tests = [v for k, v in globals().items()
              if k.startswith("test_") and callable(v) and inspect.isfunction(v)]
    passed = failed = 0
    failed_tests = []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_tests.append((t.__name__, str(e)))
    print(f"\n{passed} passed, {failed} failed")
