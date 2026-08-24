"""Tests for the Opener Layer (paper 111)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Cell, Substrate


def test_chart_opener():
    s = Substrate()
    c = Cell(address="bay/A17", value=12.5)
    s.add(c)
    out = s.render("chart")
    assert "cells" in out
    assert out["cells"][0]["address"] == "bay/A17"


def test_list_opener():
    s = Substrate()
    s.add(Cell(address="a", value=1))
    s.add(Cell(address="b", value=2))
    out = s.render("list")
    assert len(out) == 2


def test_tensor_opener():
    s = Substrate()
    c = Cell(address="t", value=None, tensor=[[[1.0, 2.0]]], axes=("d", "x", "y"))
    s.add(c)
    out = s.render("tensor", address="t")
    assert out == [[[1.0, 2.0]]]


def test_witness_opener():
    s = Substrate()
    c = Cell(address="w", value=42.0)
    s.add(c)
    s.witness(c, "agent-001", "read", 42.0)
    out = s.render("witness", address="w")
    assert len(out) == 1


def test_convoy_opener():
    s = Substrate()
    a = Cell(address="a", value=1.0)
    b = Cell(address="b", value=2.0)
    # b receives from a → b's convoy records a
    b.connect(a, weight=0.7)
    s.add(a)
    s.add(b)
    out = s.render("convoy", address="b")
    assert len(out) == 1
    assert out[0]["weight"] == 0.7


def test_graph_opener():
    s = Substrate()
    a = Cell(address="a", value=1)
    b = Cell(address="b", value=2)
    b.connect(a)
    s.add(a)
    s.add(b)
    out = s.render("graph")
    assert len(out["nodes"]) == 2
    assert len(out["edges"]) == 1


def test_unknown_opener_returns_error():
    s = Substrate()
    s.add(Cell(address="a", value=1))
    out = s.render("nonexistent")
    assert "error" in out


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
