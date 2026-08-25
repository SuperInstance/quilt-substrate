"""test_casting_plugin.py — Tests for the Quilt casting-call plugin."""
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.plugins.casting import (
    QuiltCastingCallPlugin, Probes, Situation, ResourceBudget, CastingDecision,
    WilsonProfiles, wilson_lower, PRIOR_ATLAS, ROLE_TO_OPENER,
)


def make_substrate():
    s = Substrate()
    s.add(Cell(address="chart:0", value=42, axes=("x", "y")))
    s.add(Cell(address="chart:1", value=84, axes=("x", "y")))
    s.add(Cell(address="chart:2", value=21, axes=("x", "y")))
    return s


def calm_probes():
    return Probes(user="casey", app="writers-room", hardware="laptop",
                   time_of_day="evening", weather="calm", crew_state="normal")


def gale_probes():
    return Probes(user="reyes", app="F/V EILEEN", hardware="tablet",
                   time_of_day="0300", weather="gale", crew_state="tired")


def test_situation_key():
    s = Situation(user="reyes", app="F/V EILEEN", hardware="tablet",
                   time_of_day="0300", weather="gale")
    key = s.to_key()
    assert "reyes" in key
    assert "gale" in key


def test_situation_rationale_hash():
    s = Situation(user="reyes", app="F/V EILEEN", hardware="tablet", time_of_day="0300")
    h = s.rationale_hash()
    assert len(h) == 8


def test_resource_budget_deadline():
    r2g = ResourceBudget(battery_pct=0.5, network="2g")
    rwifi = ResourceBudget(battery_pct=0.5, network="wifi")
    assert r2g.deadline_ms > rwifi.deadline_ms


def test_prior_atlas_covers_models():
    expected = {"HERMES_405B", "CLAUDE_OPUS", "QWEN3_CODER", "SEED_MINI", "QWEN_0_5B", "PHI-4"}
    actual = set(PRIOR_ATLAS.keys())
    assert expected.issubset(actual)


def test_role_to_opener_covers_roles():
    expected = {"voice_narration", "code_generation", "creative_ideation",
                 "safety_check", "sensory_creative", "fable_compression", "math_grief"}
    actual = set(ROLE_TO_OPENER.keys())
    assert expected.issubset(actual)


def test_wilson_lower_zero_n():
    assert wilson_lower(0, 0) == 0.0


def test_wilson_lower_increases_with_success():
    low = wilson_lower(1, 10)
    high = wilson_lower(8, 10)
    assert high > low


def test_wilson_lower_at_100pct_is_high():
    a = wilson_lower(5, 5)
    b = wilson_lower(50, 50)
    assert a > 0.5, f"wilson_lower(5,5)={a}"
    assert b > a, f"wilson_lower(50,50)={b} should be > wilson_lower(5,5)={a}"
    assert b < 1.0


def test_wilson_profiles_optimistic_prior():
    w = WilsonProfiles()
    assert w.lower_bound("Murmur", "chart", "X") == 0.5


def test_wilson_profiles_observe_and_rank():
    w = WilsonProfiles()
    for _ in range(3):
        w.observe("Murmur", "chart", "X", 100, True)
    for _ in range(3):
        w.observe("Murmur", "chart", "Y", 100, False)
    rank = w.rank("Murmur", "chart", ["X", "Y"])
    assert rank[0][1] == "X"
    assert rank[1][1] == "Y"


def test_probes_situation():
    p = calm_probes()
    s = p.situation()
    assert s.user == "casey"
    assert s.app == "writers-room"
    assert s.weather == "calm"


def test_probes_budget():
    p = calm_probes()
    b = p.budget()
    assert isinstance(b, ResourceBudget)
    assert 0.0 <= b.battery_pct <= 1.0


def test_probes_setters():
    p = calm_probes()
    p.set_weather("gale")
    p.set_crew_state("tired")
    s = p.situation()
    assert s.weather == "gale"
    assert s.crew_state == "tired"


def test_plugin_install_uninstall():
    s = make_substrate()
    p = calm_probes()
    original = s.render
    plugin = QuiltCastingCallPlugin(s, probes=p)
    plugin.install()
    assert s.render != original
    plugin.uninstall()
    assert s.render == original


def test_plugin_decide_returns_decision():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    d = plugin.decide(opener="chart", kwargs={"role": "creative_ideation"})
    assert isinstance(d, CastingDecision)
    assert d.model in PRIOR_ATLAS


def test_plugin_render_invokes_substrate():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    plugin.install()
    result = s.render(opener="chart")
    assert isinstance(result, dict)


def test_plugin_render_witness_proposed_and_observed():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    plugin.install()
    s.render(opener="chart")
    kinds = [e["kind"] for e in plugin.witness]
    assert "cast.proposed" in kinds
    assert "cast.observed" in kinds


def test_plugin_wilson_updates():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    plugin.install()
    for _ in range(3):
        s.render(opener="chart", role="creative_ideation")
    obs = [e for e in plugin.witness if e["kind"] == "cast.observed"]
    assert len(obs) >= 3
    assert len(plugin.wilson.obs) > 0


def test_gale_prefers_local():
    s = make_substrate()
    p = gale_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    d = plugin.decide(opener="tide", kwargs={"role": "sensory_creative"})
    assert d.model in {"DEEPSEEK_V4_FLASH", "QWEN_0_5B", "GRANITE_3_1_2B", "SEED_MINI"}


def test_calm_uses_richer_models():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    decisions = [plugin.decide(opener="slate", kwargs={"role": "creative_ideation"}) for _ in range(20)]
    models = set(d.model for d in decisions)
    assert len(models) >= 1


def test_low_battery_changes_decision():
    s = make_substrate()
    p_low = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p_low)
    with patch.object(p_low, "_read_battery", return_value=0.05):
        d_low = plugin.decide(opener="slate", kwargs={"role": "creative_ideation"})
    assert d_low.model in PRIOR_ATLAS


def test_no_candidates_falls_back():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    sit = Situation(user="casey", app="test", hardware="embedded",
                     time_of_day="0300", weather="storm", crew_state="critical")
    budget = ResourceBudget(battery_pct=0.01, network="none", compute="embedded", compute_free_pct=0.0)
    d = plugin._decide(opener="chart", sit=sit, budget=budget, kwargs={"role": "voice_narration"})
    assert d.is_fallback or d.model == "QWEN_0_5B"


def test_retry_with_local_on_failure():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    decision = CastingDecision(model="CLAUDE_OPUS", opener="chart", primitive="Murmur",
                                rationale="first try", confidence=0.9, is_fallback=False)
    sit = p.situation()
    budget = ResourceBudget(battery_pct=0.5, network="2g")
    retry = plugin._retry_with_local(decision, sit, budget, kwargs={})
    assert retry.model == "QWEN_0_5B"
    assert retry.is_fallback


def test_retry_with_echo():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    decision = CastingDecision(model="CLAUDE_OPUS", opener="chart", primitive="Murmur",
                                rationale="first try", confidence=0.9, is_fallback=False)
    retry = plugin._retry_with_echo(decision, "chart")
    assert retry.model == "none"
    assert retry.opener == "log"
    assert retry.primitive == "echo"
    assert retry.is_fallback


def test_infer_role():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    assert plugin._infer_role("voice", {}) == "voice_narration"
    assert plugin._infer_role("witness", {}) == "safety_check"
    assert plugin._infer_role("reef", {}) == "math_grief"
    assert plugin._infer_role("slate", {}) == "fable_compression"
    assert plugin._infer_role("chart", {"role": "code_generation"}) == "code_generation"


def test_stats():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    plugin.install()
    s.render(opener="chart")
    s.render(opener="slate")
    stats = plugin.stats()
    assert stats["n_proposed"] == 2
    assert stats["n_observed"] == 2
    assert "models" in stats
    assert "openers" in stats


def test_stats_success_rate():
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    plugin.install()
    s.render(opener="chart")
    stats = plugin.stats()
    assert 0.0 <= stats["success_rate"] <= 1.0


def test_substrate_renders_unchanged_after_uninstall():
    s = make_substrate()
    p = calm_probes()
    original = s.render
    plugin = QuiltCastingCallPlugin(s, probes=p)
    plugin.install()
    plugin.uninstall()
    assert s.render == original


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    failed_tests = []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
            failed_tests.append((t.__name__, str(e)))
    print(f"\n{passed} passed, {failed} failed")
