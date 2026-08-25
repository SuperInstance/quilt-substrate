"""test_pincher_cache.py — Tests for the pincher reflex-cache layer."""
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.plugins.pincher_cache import (
    PincherCachedCastingPlugin, CachedReflex,
)
from quilt_substrate.plugins.casting import Probes, ResourceBudget


def make_substrate():
    s = Substrate()
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    return s


def calm_probes():
    return Probes(user="casey", app="writers-room", hardware="laptop",
                   time_of_day="evening", weather="calm", crew_state="normal")


def test_pseudo_embed_deterministic():
    """The pseudo-embedding is deterministic."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        e1 = plugin._pseudo_embed("hello world")
        e2 = plugin._pseudo_embed("hello world")
        assert e1 == e2


def test_pseudo_embed_different_text():
    """Different text produces different embeddings."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        e1 = plugin._pseudo_embed("hello world")
        e2 = plugin._pseudo_embed("goodbye world")
        assert e1 != e2


def test_cosine_similarity_self():
    """A vector has cosine similarity 1.0 with itself."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        v = [0.1, 0.2, 0.3, 0.4]
        assert abs(plugin._cosine_similarity(v, v) - 1.0) < 0.001


def test_cosine_similarity_zero_vector():
    """A zero vector has zero similarity with anything."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        z = [0.0] * 10
        v = [0.1] * 10
        assert plugin._cosine_similarity(z, v) == 0.0


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors have zero similarity."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(plugin._cosine_similarity(a, b)) < 0.001


def test_find_reflex_empty_cache():
    """Empty cache returns None, 0.0."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        reflex, sim = plugin._find_reflex("anything")
        assert reflex is None
        assert sim == 0.0


def test_teach_and_find_reflex():
    """A taught reflex can be found by a similar query."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        plugin.teach(intent="what's the depth at lat 24.5?",
                     result="4.2 meters", model="QWEN_0_5B", quality=0.9)
        # Query with the EXACT same intent — pseudo-embeddings give 1.0 similarity
        reflex, sim = plugin._find_reflex("what's the depth at lat 24.5?")
        assert reflex is not None
        assert sim > 0.5  # same intent = high similarity


def test_teach_persists_to_disk():
    """A taught reflex is written to disk."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cache.jsonl"
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(path))
        plugin.teach(intent="test", result="ok", model="X", quality=0.9)
        assert path.exists()
        with open(path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            d_json = json.loads(lines[0])
            assert d_json["intent"] == "test"


def test_cache_loads_from_disk():
    """A cache loaded from disk has its reflexes."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cache.jsonl"
        # Pre-populate
        reflex_data = {
            "intent": "test intent",
            "intent_embedding": [0.1] * 64,
            "model": "QWEN_0_5B",
            "opener": "slate",
            "primitive": "Murmur",
            "result": "ok",
            "quality": 0.9,
            "timestamp": 1234567890.0,
            "hit_count": 0,
        }
        with open(path, "w") as f:
            f.write(json.dumps(reflex_data) + "\n")
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(path))
        assert len(plugin._cache) == 1
        assert plugin._cache[0].intent == "test intent"


def test_decide_cache_hit():
    """A high-similarity cache hit returns the cached decision."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"),
                                              high_confidence=0.5)  # lower threshold
        plugin.teach(intent="hello world", result="ok", model="QWEN_0_5B",
                       opener="slate", quality=0.9)
        # Querying the SAME intent should hit the cache
        d = plugin.decide(opener="chart", kwargs={"intent": "hello world"})
        # The decision should reference the cached model
        assert "reflex" in d.rationale.lower() or d.model == "QWEN_0_5B"


def test_decide_cache_miss():
    """A low-similarity query misses the cache and falls through."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"),
                                              high_confidence=0.99)  # very high
        plugin.teach(intent="hello world", result="ok", model="QWEN_0_5B",
                       opener="slate", quality=0.9)
        # Querying a totally different intent — should miss
        d = plugin.decide(opener="chart", kwargs={"intent": "completely different xyzzy"})
        # The decision should NOT be a cache hit
        assert "reflex" not in d.rationale.lower()


def test_cache_stats_empty():
    """Empty cache stats."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        stats = plugin.cache_stats()
        assert stats["n_reflexes"] == 0
        assert stats["total_hits"] == 0


def test_cache_stats_with_reflexes():
    """Stats reflect the cache contents."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"))
        plugin.teach("a", "result_a", quality=0.8)
        plugin.teach("b", "result_b", quality=0.9)
        stats = plugin.cache_stats()
        assert stats["n_reflexes"] == 2
        assert abs(stats["avg_quality"] - 0.85) < 0.01


def test_render_cache_hit_records_observation():
    """A cache hit records a witness event and updates Wilson."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(Path(d) / "cache.jsonl"),
                                              high_confidence=0.5)
        plugin.teach(intent="what's the depth", result={"depth": 4.2},
                       model="QWEN_0_5B", opener="chart", quality=0.9)
        plugin.install()
        result = s.render(opener="chart", intent="what's the depth")
        # The witness has an observation
        kinds = [e["kind"] for e in plugin.witness]
        assert "cast.proposed" in kinds
        assert "cast.observed" in kinds
        # The result is the cached one
        assert result == {"depth": 4.2}


def test_render_cache_miss_calls_substrate():
    """A cache miss falls through to the substrate."""
    s = make_substrate()
    p = calm_probes()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cache.jsonl"
        plugin = PincherCachedCastingPlugin(s, probes=p, use_pincher=False,
                                              reflex_cache_path=str(path),
                                              high_confidence=0.99)
        plugin.install()
        # Force a specific role so the decision is predictable
        result = s.render(opener="chart", role="code_generation", intent="something new")
        # The result is from the substrate
        assert isinstance(result, dict)
        # A new reflex was added to the cache
        assert path.exists()
        with open(path) as f:
            lines = f.readlines()
            assert len(lines) == 1


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
