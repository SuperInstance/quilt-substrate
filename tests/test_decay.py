"""Tests for the Decay primitive (paper 109)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from substrate import Cell, Substrate


def test_initial_confidence_is_one():
    c = Cell(address="c1", value=1.0)
    assert abs(c.confidence - 1.0) < 0.01


def test_confidence_decays_with_time():
    c = Cell(address="c1", value=1.0)
    # Manually push t0 back in time to simulate aging
    c.decay.t0 -= 1000  # 1000 seconds ago
    conf = c.confidence
    assert conf < 1.0
    assert conf > 0.0


def test_refresh_resets_confidence():
    c = Cell(address="c1", value=1.0)
    c.decay.t0 -= 1000
    c.refresh()
    assert abs(c.confidence - 1.0) < 0.01


def test_fresh_cell_higher_confidence_than_stale():
    fresh = Cell(address="fresh", value=1.0)
    stale = Cell(address="stale", value=1.0)
    stale.decay.t0 -= 3600  # 1 hour ago
    assert fresh.confidence > stale.confidence


def test_decay_function_with_higher_lambda():
    slow = Cell(address="slow", value=1.0)
    slow.decay.lam = 0.0001
    slow.decay.t0 -= 1000
    fast = Cell(address="fast", value=1.0)
    fast.decay.lam = 0.001
    fast.decay.t0 -= 1000
    assert slow.confidence > fast.confidence


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
