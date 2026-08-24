"""Tests for the Convoy primitive (paper 108)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Cell, Substrate


def test_convoy_initially_empty():
    c = Cell(address="c1", value=1.0)
    assert c.convoy == []


def test_convoy_adds_on_connect():
    a = Cell(address="a", value=1.0)
    b = Cell(address="b", value=2.0)
    # b receives from a → b's convoy records a
    b.connect(a, weight=0.7)
    assert len(b.convoy) == 1
    assert b.convoy[0].agent_id == "a"
    assert b.convoy[0].weight == 0.7


def test_convoy_replaces_existing_entry():
    a = Cell(address="a", value=1.0)
    b = Cell(address="b", value=2.0)
    b.connect(a, weight=0.5)
    b.connect(a, weight=0.9)
    assert len(b.convoy) == 1
    assert b.convoy[0].weight == 0.9


def test_convoy_value_fallback():
    c = Cell(address="c", value=42.0)
    val = c.convoy_value()
    assert val == 42.0


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
