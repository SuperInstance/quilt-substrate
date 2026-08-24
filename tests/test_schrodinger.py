"""Tests for the Schrödinger pattern and the substrate-level operations (paper 107)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Cell, Substrate


def test_cell_initially_not_canonical():
    c = Cell(address="c1", value=42.0)
    assert c.canonical is False


def test_infer_sets_inference_not_canonical():
    c = Cell(address="c1", value=42.0)
    c.infer(50.0)
    assert c.inference == 50.0
    assert c.canonical is False
    # The actual value is still 42.0
    assert c.value == 42.0


def test_observe_canonical_marks_canonical():
    c = Cell(address="c1", value=42.0)
    c.infer(50.0)
    val = c.observe_canonical()
    assert c.canonical is True
    # Observe returns the actual value, not the inference
    assert val == 42.0


def test_substrate_observe_witnesses():
    s = Substrate()
    c = Cell(address="c1", value=42.0)
    s.add(c)
    val = s.observe("c1", agent_id="captain-reyes")
    assert val == 42.0
    assert c.canonical is True
    assert len(c.witness_log) == 1
    assert c.witness_log[0].agent_id == "captain-reyes"


def test_substrate_infer_witnesses():
    s = Substrate()
    c = Cell(address="c1", value=42.0)
    s.add(c)
    s.infer("c1", 50.0, agent_id="drone-skate")
    assert c.inference == 50.0
    assert c.canonical is False
    assert len(c.witness_log) == 1


def test_substrate_refresh_resets_decay():
    s = Substrate()
    c = Cell(address="c1", value=42.0)
    s.add(c)
    c.decay.t0 -= 3600
    s.refresh("c1")
    assert c.confidence > 0.99


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
