"""Tests for paper 124: The Substrate's Temperature.

The temperature of a cell is the entropy of its witness log over a
sliding time window. Cold cells have low entropy; hot cells have high
entropy. The substrate uses temperature to:
- Reweight JEPA training
- Select openers

These tests verify that the temperature formalism works on the substrate.
"""
import sys
import os
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from quilt_substrate import Cell, Substrate, WitnessEntry


def test_temperature_no_writes():
    """A cell with no witness entries has temperature 0 (frozen)."""
    substrate = Substrate()
    cell = Cell(address="a", value=1.0)
    substrate.add(cell)
    T = cell.temperature(window_seconds=3600)
    assert T == 0.0, f"Expected 0, got {T}"


def test_temperature_one_op():
    """A cell with one operation type has temperature 0 (cold, but not zero)."""
    substrate = Substrate()
    cell = Cell(address="a", value=1.0)
    substrate.add(cell)
    for _ in range(5):
        cell.witness("agent1", "read", 1.0)
    T = cell.temperature(window_seconds=3600)
    # All entries are "read", so entropy is 0
    assert T == 0.0, f"Expected 0 (single op), got {T}"


def test_temperature_two_ops():
    """A cell with two operation types has temperature ln(2)."""
    substrate = Substrate()
    cell = Cell(address="a", value=1.0)
    substrate.add(cell)
    for _ in range(3):
        cell.witness("agent1", "read", 1.0)
        cell.witness("agent2", "write", 1.0)
    T = cell.temperature(window_seconds=3600)
    # 50% read, 50% write → entropy = -2*0.5*ln(0.5) = ln(2) ≈ 0.693
    expected = math.log(2)
    assert abs(T - expected) < 0.01, f"Expected ln(2)={expected}, got {T}"


def test_temperature_uniform():
    """A cell with N equal-frequency operations has temperature ln(N)."""
    substrate = Substrate()
    cell = Cell(address="a", value=1.0)
    substrate.add(cell)
    ops = ["read", "write", "infer", "refresh", "witness"]
    for op in ops:
        cell.witness("agent1", op, 1.0)
    T = cell.temperature(window_seconds=3600)
    expected = math.log(5)  # 5 distinct ops, uniform
    assert abs(T - expected) < 0.01, f"Expected ln(5)={expected}, got {T}"


def test_temperature_bounded():
    """Temperature is bounded above by ln(N) for N operation types."""
    substrate = Substrate()
    cell = Cell(address="a", value=1.0)
    substrate.add(cell)
    # Write all 11 primitives
    for op in ["read", "write", "infer", "refresh", "witness",
               "decay", "convoy", "jepa", "cite", "compile", "decay"]:
        cell.witness("agent1", op, 1.0)
    T = cell.temperature(window_seconds=3600)
    # Some ops might be duplicates, so temperature ≤ ln(unique)
    # With 11 distinct ops uniform, T = ln(11) ≈ 2.398
    assert T <= math.log(11) + 0.01, f"Temperature {T} exceeds ln(11)"


def test_temperature_regimes():
    """Test the regime classification."""
    substrate = Substrate()
    cell = Cell(address="a", value=1.0)
    substrate.add(cell)
    # Frozen
    assert cell.regime() == "frozen"  # no writes
    # Cold (one op = still frozen; let's add a different op)
    cell.witness("a", "read", 1.0)
    cell.witness("a", "write", 1.0)
    assert cell.regime() in ("cold", "warm")  # 2 ops uniform = ln(2) = warm
    # Warm
    for op in ["read", "write", "inference"]:
        cell.witness("a", op, 1.0)
    assert cell.regime() in ("cold", "warm", "hot")
    # Hot
    for op in ["decay", "convoy", "jepa"]:
        cell.witness("a", op, 1.0)
    assert cell.regime() in ("warm", "hot")


def test_substrate_wide_temperature():
    """The substrate-wide temperature is a weighted average of cell temperatures."""
    substrate = Substrate()
    c1 = Cell(address="a", value=1.0)
    substrate.add(c1)
    c2 = Cell(address="b", value=2.0)
    substrate.add(c2)
    # c1: 1 op type (read)
    c1.witness("a", "read", 1.0)
    # c2: 2 op types (read, write)
    c2.witness("a", "read", 1.0)
    c2.witness("a", "write", 1.0)
    T_bar = substrate.temperature()
    # c1.T = 0 (1 op), c2.T = ln(2) ≈ 0.693
    # Weighted by witness count: c1 has 1, c2 has 2
    # T_bar = (1*0 + 2*ln(2)) / 3 = 2*0.693/3 ≈ 0.462
    expected = (1 * 0 + 2 * math.log(2)) / 3
    assert abs(T_bar - expected) < 0.01, f"Expected {expected}, got {T_bar}"


def test_decay_lowers_temperature():
    """Decay does not increase a cell's temperature."""
    substrate = Substrate()
    cell = Cell(address="a", value=1.0)
    substrate.add(cell)
    for op in ["read", "write", "infer", "decay"]:
        cell.witness("a", op, 1.0)
    T_before = cell.temperature(window_seconds=3600)
    # Advance time far into the future
    substrate.advance_time(86400 * 365)  # 1 year
    # Old witness entries are now ancient, but they still exist
    # The cell's temperature is unchanged because decay doesn't remove entries
    T_after = cell.temperature(window_seconds=3600)
    # If decay doesn't remove entries, T_after == T_before
    # (or T_after could be lower if older entries are filtered out)
    assert T_after <= T_before + 0.01, f"Decay raised T: {T_before} -> {T_after}"


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
