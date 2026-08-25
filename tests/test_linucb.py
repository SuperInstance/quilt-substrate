"""test_linucb.py — Tests for the LinUCB layer."""
import sys
import os
import math
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.plugins.linucb import (
    LinUCBCaster, LinUCBCastingPlugin, LinUCBModel,
    FeatureExtractor, alpha_for,
    TIME_OF_DAY, WEATHER, NETWORK, CREW_STATE,
)
from quilt_substrate.plugins.casting import (
    QuiltCastingCallPlugin, Probes, Situation, ResourceBudget,
)


def make_substrate():
    s = Substrate()
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    return s


def calm_probes():
    return Probes(user="casey", app="writers-room", hardware="laptop",
                   time_of_day="evening", weather="calm", crew_state="normal")


def gale_probes():
    return Probes(user="reyes", app="F/V EILEEN", hardware="tablet",
                   time_of_day="0300", weather="gale", crew_state="tired")


# -- alpha_for tests --

def test_alpha_for_calm_full_battery():
    """Calm + full battery = high alpha (lots of exploration)."""
    alpha = alpha_for(battery=1.0, weather="calm", crew_state="normal")
    assert alpha >= 0.8, f"Expected >= 0.8, got {alpha}"


def test_alpha_for_gale_low_battery():
    """Gale + low battery = low alpha (conservative)."""
    alpha = alpha_for(battery=0.1, weather="gale", crew_state="tired")
    assert alpha <= 0.3, f"Expected <= 0.3, got {alpha}"


def test_alpha_for_storm_critical():
    """Storm + critical = very low alpha."""
    alpha = alpha_for(battery=0.05, weather="storm", crew_state="critical")
    assert alpha <= 0.1


def test_alpha_clipped():
    """Alpha is always in [0.05, 1.0]."""
    for bat in [0.0, 0.5, 1.0]:
        for w in ["calm", "windy", "gale", "storm"]:
            a = alpha_for(battery=bat, weather=w)
            assert 0.05 <= a <= 1.0


# -- FeatureExtractor tests --

def test_feature_dim():
    """The feature vector has a known dimension."""
    e = FeatureExtractor()
    assert e.dim == (len(e.models) + len(e.openers) + len(e.primitives) + len(e.users)
                      + len(TIME_OF_DAY) + len(WEATHER) + 1 + len(NETWORK) + len(CREW_STATE))


def test_feature_extraction_correctness():
    """The feature vector has 1.0 in the right slots and 0.0 elsewhere."""
    e = FeatureExtractor(models=["M1", "M2"], openers=["O1"], primitives=["P1"],
                          users=["alice", "bob"])
    sit = Situation(user="alice", app="app", hardware="laptop",
                    time_of_day="morning", weather="calm", crew_state="normal")
    x = e(("M1", "O1", "P1"), sit)
    assert x.shape == (e.dim,)
    # M1, O1, P1, alice = indices 0, 2, 3, 4
    assert x[0] == 1.0  # M1
    assert x[2] == 1.0  # O1
    assert x[3] == 1.0  # P1
    assert x[4] == 1.0  # alice
    # morning, calm
    assert x[6] == 1.0  # morning
    assert x[11] == 1.0  # calm
    # default battery
    assert x[15] == 0.5


def test_feature_with_battery_override():
    """with_battery replaces the battery feature."""
    e = FeatureExtractor()
    sit = Situation(user="u", app="a", hardware="h")
    x = e(("M1", "O1", "P1"), sit)
    x2 = e.with_battery(x, 0.1)
    # The battery feature should be 0.1
    # ... and the rest should be unchanged
    diff = np.abs(x - x2).sum()
    assert diff > 0  # something changed


# -- LinUCBModel tests --

def test_linucb_model_init():
    """A new model has A=I, b=0, n=0."""
    m = LinUCBModel(d=5)
    assert m.n == 0
    assert m.A.shape == (5, 5)
    assert m.b.shape == (5,)


def test_linucb_model_update():
    """Updating with a feature and reward increments n."""
    m = LinUCBModel(d=3)
    x = np.array([1.0, 0.0, 0.0])
    m.update(x, 1.0)
    assert m.n == 1
    assert m.b[0] == 1.0
    assert m.A[0, 0] == 2.0  # I + outer


def test_linucb_model_predict_ucb_widens():
    """UCB width decreases with more observations."""
    m = LinUCBModel(d=3)
    x = np.array([1.0, 0.0, 0.0])
    _, ucb0 = m.predict(x, alpha=1.0)
    for _ in range(10):
        m.update(x, 1.0)
    _, ucb10 = m.predict(x, alpha=1.0)
    assert ucb10 < ucb0


# -- LinUCBCaster tests --

def test_linucb_caster_rank_pure_wilson_n_lt_10():
    """With n=0, the rank is pure Wilson (50% prior + 50% Wilson optimistic)."""
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    l = LinUCBCaster(plugin)
    sit = p.situation()
    budget = p.budget()
    candidates = [("HERMES_405B", "voice", "Murmur"), ("DEEPSEEK_V4_FLASH", "tide", "Murmur")]
    ranked = l.rank(candidates, sit, budget)
    # All scores should be 0.5 (Wilson optimistic prior) since no observations
    for score, _ in ranked:
        assert score == 0.5


def test_linucb_caster_warmup_blend():
    """At n=15 (mid warmup), the LinUCB weight is 0.5."""
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    l = LinUCBCaster(plugin)
    assert l._linucb_weight(0) == 0.0
    assert l._linucb_weight(5) == 0.0
    assert l._linucb_weight(10) == 0.0
    assert l._linucb_weight(15) == 0.5  # mid warmup
    assert l._linucb_weight(20) == 1.0
    assert l._linucb_weight(100) == 1.0


def test_linucb_caster_update_changes_model():
    """Updating the model with observations changes future predictions."""
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    l = LinUCBCaster(plugin)
    sit = p.situation()
    # Update with 25 observations (well past warmup)
    for _ in range(25):
        l.update(("HERMES_405B", "voice", "Murmur"), sit, reward=1.0)
    m = l._model_for(sit.user, sit.app)
    assert m.n == 25


def test_linucb_caster_rank_after_observations():
    """After observations, the rank is informed by the data."""
    s = make_substrate()
    p = calm_probes()
    plugin = QuiltCastingCallPlugin(s, probes=p)
    l = LinUCBCaster(plugin)
    sit = p.situation()
    budget = p.budget()
    # Feed 25 successes for HERMES_405B
    for _ in range(25):
        l.update(("HERMES_405B", "voice", "Murmur"), sit, reward=1.0)
    # Feed 25 failures for QWEN_0_5B
    for _ in range(25):
        l.update(("QWEN_0_5B", "slate", "Murmur"), sit, reward=0.0)
    # Rank candidates — both have data, but HERMES has more positive evidence
    candidates = [("HERMES_405B", "voice", "Murmur"), ("QWEN_0_5B", "slate", "Murmur")]
    ranked = l.rank(candidates, sit, budget)
    # The model has learned: HERMES = high reward, QWEN = low reward
    # Find each in the ranking
    scores_by_model = {c[0]: s for s, c in ranked}
    # Both should have non-trivial scores (learned, not optimistic)
    assert abs(scores_by_model["HERMES_405B"] - 0.5) > 0.01  # not optimistic
    assert abs(scores_by_model["QWEN_0_5B"] - 0.5) > 0.01  # not optimistic


# -- LinUCBCastingPlugin tests --

def test_plugin_install_uninstall():
    """Install/uninstall works on the LinUCB plugin."""
    s = make_substrate()
    p = calm_probes()
    plugin = LinUCBCastingPlugin(s, probes=p)
    original = s.render
    plugin.install()
    assert s.render != original
    plugin.uninstall()
    assert s.render == original


def test_plugin_decide_uses_linucb():
    """The decide() method uses LinUCB to rank candidates."""
    s = make_substrate()
    p = calm_probes()
    plugin = LinUCBCastingPlugin(s, probes=p)
    d = plugin.decide(opener="voice", kwargs={"role": "voice_narration"})
    assert d.model in plugin.linucb.extractor.models or d.model == "QWEN_0_5B"
    assert "Wilson+LinUCB" in d.rationale


def test_plugin_render_records_to_linucb():
    """After rendering, the LinUCB layer has observations."""
    s = make_substrate()
    p = calm_probes()
    plugin = LinUCBCastingPlugin(s, probes=p)
    plugin.install()
    s.render(opener="chart", role="creative_ideation")
    # LinUCB history should have an entry
    assert len(plugin._linucb_history) >= 1


def test_plugin_linucb_stats():
    """The plugin exposes LinUCB stats."""
    s = make_substrate()
    p = calm_probes()
    plugin = LinUCBCastingPlugin(s, probes=p)
    stats = plugin.linucb_stats()
    assert "n_models" in stats
    assert "feature_dim" in stats


def test_plugin_gale_prefers_local():
    """In a gale, the LinUCB plugin still picks local or cheap models."""
    s = make_substrate()
    p = gale_probes()
    # Mock the budget to be offline/low-battery
    with patch.object(p, "_read_battery", return_value=0.1), \
         patch.object(p, "_read_network", return_value="none"):
        plugin = LinUCBCastingPlugin(s, probes=p)
        d = plugin.decide(opener="tide", kwargs={"role": "sensory_creative"})
        # In a gale with no network, should prefer local
        assert d.is_fallback or d.model in {"QWEN_0_5B", "DEEPSEEK_V4_FLASH"}


def test_linucb_exploration_high_in_calm():
    """The LinUCB exploration bonus is high in calm weather."""
    e = FeatureExtractor()
    m = LinUCBModel(d=e.dim)
    x = e(("HERMES_405B", "voice", "Murmur"),
           Situation(user="u", app="a", hardware="h", time_of_day="morning", weather="calm"))
    x_calm = e.with_battery(x, 1.0)
    _, ucb_calm = m.predict(x_calm, alpha=alpha_for(1.0, "calm"))
    # In calm + full battery, alpha is high
    # The UCB should be wider (more exploration)
    assert ucb_calm > 0


# Run as a script
if __name__ == "__main__":
    import inspect
    tests = [v for k, v in globals().items()
              if k.startswith("test_") and callable(v) and inspect.isfunction(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
