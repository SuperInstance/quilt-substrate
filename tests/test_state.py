"""test_state.py — Tests for the persistent state layer."""
import json
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/workspace/quilt-substrate/src")

from quilt_substrate.state import (
    SCHEMA_VERSION, StateManager,
    save_wilson, load_wilson, save_linucb, load_linucb_models,
    atomic_write_json, load_json, atomic_write_jsonl, load_jsonl,
    wilson_to_dict, wilson_from_dict,
)


# --- Pure functions -------------------------------------------------

def test_atomic_write_and_load_json():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.json")
        atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
        assert os.path.exists(path)
        d_loaded = load_json(path)
        assert d_loaded["a"] == 1
        assert d_loaded["b"] == [1, 2, 3]


def test_load_json_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        result = load_json(os.path.join(d, "missing.json"))
        assert result is None


def test_atomic_write_jsonl():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.jsonl")
        records = [{"i": i, "v": f"row{i}"} for i in range(5)]
        atomic_write_jsonl(path, records)
        loaded = load_jsonl(path)
        assert len(loaded) == 5
        assert loaded[0]["i"] == 0
        assert loaded[4]["v"] == "row4"


def test_load_jsonl_missing_returns_empty():
    with tempfile.TemporaryDirectory() as d:
        result = load_jsonl(os.path.join(d, "missing.jsonl"))
        assert result == []


def test_atomic_write_uses_tmp_and_rename():
    """No .tmp file should be left over after a successful write."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.json")
        atomic_write_json(path, {"k": "v"})
        # No .tmp file
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path)


# --- Wilson round-trip ----------------------------------------------

def test_wilson_roundtrip_via_dict():
    obs = {
        ("Murmur", "tide", "PHI-4"): [
            (0.9, 100.0, 200, True),
            (0.85, 110.0, 250, True),
            (0.1, 200.0, 500, False),
        ],
        ("JEPA", "reef", "QWEN_0_5B"): [
            (0.5, 100.0, 50, True),
        ],
    }
    d = wilson_to_dict(obs, threshold=0.6, window=200, half_life=3600.0)
    assert d["schema"] == SCHEMA_VERSION
    assert d["kind"] == "wilson"
    assert "Murmur|tide|PHI-4" in d["obs"]
    threshold, window, half_life, obs2 = wilson_from_dict(d)
    assert threshold == 0.6
    assert window == 200
    assert half_life == 3600.0
    assert len(obs2) == 2
    assert obs2[("Murmur", "tide", "PHI-4")][0] == (0.9, 100.0, 200, True)
    assert obs2[("JEPA", "reef", "QWEN_0_5B")][0] == (0.5, 100.0, 50, True)


def test_wilson_roundtrip_via_disk():
    from quilt_substrate.plugins.casting import WilsonProfiles
    w = WilsonProfiles(threshold=0.7, window=100, half_life_s=1800.0)
    w.observe("Murmur", "tide", "PHI-4", 200, True, 0.9)
    w.observe("Murmur", "tide", "PHI-4", 250, True, 0.85)
    w.observe("JEPA", "reef", "QWEN_0_5B", 50, True, 0.5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "wilson.json")
        save_wilson(w, path)
        w2 = load_wilson(path)
        assert w2 is not None
        assert w2.threshold == 0.7
        assert w2.window == 100
        assert w2.half_life == 1800.0
        # The lower_bound should be identical
        assert (w.lower_bound("Murmur", "tide", "PHI-4")
                  == w2.lower_bound("Murmur", "tide", "PHI-4"))


def test_load_wilson_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        result = load_wilson(os.path.join(d, "missing.json"))
        assert result is None


def test_wilson_unknown_schema_raises():
    d = {
        "schema": 999, "kind": "wilson",
        "threshold": 0.6, "window": 200, "half_life": 3600.0,
        "obs": {},
    }
    try:
        wilson_from_dict(d)
        assert False, "should have raised"
    except ValueError as e:
        assert "Unknown schema" in str(e)


# --- LinUCB round-trip ----------------------------------------------

def test_linucb_roundtrip_via_dict():
    """Build a fake LinUCB caster, serialize, deserialize."""
    import numpy as np
    from quilt_substrate.plugins.linucb import LinUCBCaster, LinUCBModel, FeatureExtractor
    # LinUCBCaster takes a plugin; we mock it
    class FakePlugin:
        pass
    extractor = FeatureExtractor()  # dim is on the extractor
    caster = LinUCBCaster(plugin=FakePlugin(), extractor=extractor)
    m = LinUCBModel(d=extractor.dim)
    m.A = np.eye(extractor.dim) * 2
    m.b = np.ones(extractor.dim) * 0.3
    m.n = 5
    caster.models[("casey", "writers-room")] = m

    d_dict = {
        "schema": SCHEMA_VERSION,
        "kind": "linucb",
        "feature_dim": extractor.dim,
        "models": {
            "casey|writers-room": {
                "A": m.A.tolist(),
                "b": m.b.tolist(),
                "n": 5,
                "feature_dim": extractor.dim,
            }
        }
    }
    d_json = json.dumps(d_dict)
    loaded = json.loads(d_json)
    models = linucb_from_dict(loaded)
    assert ("casey", "writers-room") in models
    m2 = models[("casey", "writers-room")]
    assert m2.n == 5
    assert np.allclose(m2.A, m.A)
    assert np.allclose(m2.b, m.b)


def linucb_from_dict(d):
    """Helper that delegates to the state module."""
    from quilt_substrate.state import linucb_from_dict as _real
    return _real(d)


def test_linucb_via_disk():
    import numpy as np
    from quilt_substrate.plugins.linucb import LinUCBCaster, LinUCBModel, FeatureExtractor
    class FakePlugin:
        pass
    extractor = FeatureExtractor()
    caster = LinUCBCaster(plugin=FakePlugin(), extractor=extractor)
    m = LinUCBModel(d=extractor.dim)
    m.A = np.eye(extractor.dim) * 3
    m.b = np.ones(extractor.dim) * 0.1
    m.n = 7
    caster.models[("reyes", "F/V EILEEN")] = m
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "linucb.json")
        save_linucb(caster, path)
        models = load_linucb_models(path)
        assert models is not None
        assert ("reyes", "F/V EILEEN") in models
        m2 = models[("reyes", "F/V EILEEN")]
        assert m2.n == 7
        assert np.allclose(m2.b, m.b)


# --- StateManager ---------------------------------------------------

def test_state_manager_paths():
    with tempfile.TemporaryDirectory() as d:
        sm = StateManager(d)
        assert sm.wilson_path == os.path.join(d, "wilson.json")
        assert sm.linucb_path == os.path.join(d, "linucb.json")
        assert sm.witness_path == os.path.join(d, "witness.jsonl")
        assert sm.cowboy_path == os.path.join(d, "cowboy.jsonl")


def test_state_manager_save_load_wilson():
    from quilt_substrate.plugins.casting import WilsonProfiles
    with tempfile.TemporaryDirectory() as d:
        sm = StateManager(d)
        w = WilsonProfiles()
        w.observe("Murmur", "tide", "PHI-4", 200, True, 0.9)
        sm.save_wilson(w)
        assert os.path.exists(sm.wilson_path)
        w2 = sm.load_wilson()
        assert w2 is not None
        assert w2.lower_bound("Murmur", "tide", "PHI-4") == \
            w.lower_bound("Murmur", "tide", "PHI-4")


def test_state_manager_exists_empty():
    with tempfile.TemporaryDirectory() as d:
        sm = StateManager(d)
        assert sm.exists() is False


def test_state_manager_exists_with_files():
    with tempfile.TemporaryDirectory() as d:
        sm = StateManager(d)
        atomic_write_json(sm.wilson_path, {"x": 1})
        assert sm.exists() is True
        files = sm.list_files()
        assert "wilson.json" in files


def test_state_manager_round_trip_full():
    """Save all 3, load all 3, verify."""
    from quilt_substrate.plugins.casting import WilsonProfiles
    import numpy as np
    from quilt_substrate.plugins.linucb import LinUCBCaster, LinUCBModel, FeatureExtractor
    with tempfile.TemporaryDirectory() as d:
        sm = StateManager(d)
        # Wilson
        w = WilsonProfiles()
        w.observe("Murmur", "tide", "PHI-4", 200, True, 0.9)
        sm.save_wilson(w)
        # LinUCB
        class FakePlugin:
            pass
        extractor = FeatureExtractor()
        caster = LinUCBCaster(plugin=FakePlugin(), extractor=extractor)
        m = LinUCBModel(d=extractor.dim)
        m.A = np.eye(extractor.dim)
        m.b = np.ones(extractor.dim) * 0.1
        m.n = 3
        caster.models[("u", "a")] = m
        sm.save_linucb(caster)
        # Witness (use the deckhand witness)
        from quilt_substrate.plugins.deckhand_witness import DeckhandWitness
        dw = DeckhandWitness(witness_path=sm.witness_path)
        dw.remember({"text": "hello world", "kind": "test", "model": "PHI-4"})
        # Reload
        w2 = sm.load_wilson()
        models = sm.load_linucb_models()
        assert w2 is not None
        assert models is not None
        assert ("u", "a") in models
        dw2 = DeckhandWitness(witness_path=sm.witness_path)
        assert len(dw2.events) == 1
