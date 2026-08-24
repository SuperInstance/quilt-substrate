"""Edge case tests for the substrate.

These tests validate that the substrate handles:
- NaN values
- Infinity values
- Duplicate cell addresses
- Very large substrates (1000+ cells)
- Long witness logs (100+ entries per cell)
- Disconnected components in the cell graph
- Cycles in the cell graph
- Empty substrate
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Substrate, Cell


def test_empty_substrate():
    """An empty substrate is well-defined."""
    s = Substrate()
    assert len(s) == 0
    assert s.all_cells() == []
    assert s.merkle_root() == "0" * 16
    assert s.all_agents() == []


def test_single_cell_substrate():
    """A substrate with one cell is well-defined."""
    s = Substrate()
    c = Cell(address="only", value=42.0)
    s.add(c)
    assert len(s) == 1
    assert s.get("only") is c
    # Merkle root of a single cell is its own leaf hash
    assert s.merkle_root() != "0" * 16


def test_nan_value_in_cell():
    """A cell with NaN value handles it gracefully (no crash)."""
    c = Cell(address="nan", value=float('nan'))
    # NaN comparisons return False, so the decay should still work
    # We just don't expect a crash
    conf = c.confidence
    assert 0.0 <= conf <= 1.0
    # The merkle root shouldn't crash either
    s = Substrate()
    s.add(c)
    root = s.merkle_root()
    assert isinstance(root, str)


def test_infinity_value_in_cell():
    """A cell with infinity value handles it gracefully (no crash)."""
    c = Cell(address="inf", value=float('inf'))
    s = Substrate()
    s.add(c)
    s.witness(c, "test", "write", float('inf'))
    # No crash is the main test
    assert c.value == float('inf')


def test_negative_value_in_cell():
    """A cell with a negative value works correctly."""
    s = Substrate()
    c = Cell(address="neg", value=-42.0)
    s.add(c)
    s.witness(c, "test", "write", -42.0)
    assert c.value == -42.0
    assert s.get("neg").value == -42.0


def test_zero_value_in_cell():
    """A cell with a zero value works correctly."""
    s = Substrate()
    c = Cell(address="zero", value=0.0)
    s.add(c)
    s.witness(c, "test", "write", 0.0)
    assert c.value == 0.0


def test_duplicate_address_overwrites():
    """Adding a cell with an existing address overwrites (last-write-wins)."""
    s = Substrate()
    c1 = Cell(address="dup", value=1.0)
    c2 = Cell(address="dup", value=2.0)
    s.add(c1)
    s.add(c2)  # overwrites
    assert len(s) == 1
    assert s.get("dup") is c2
    assert s.get("dup").value == 2.0


def test_large_substrate_1000_cells():
    """A substrate with 1000 cells scales correctly."""
    s = Substrate()
    for i in range(1000):
        c = Cell(address=f"bay/{i:04d}", value=float(i))
        s.add(c)
    assert len(s) == 1000
    # Merkle root should compute in reasonable time
    root = s.merkle_root()
    assert isinstance(root, str)
    assert len(root) == 16


def test_long_witness_log():
    """A cell with 1000 witness entries scales correctly."""
    s = Substrate()
    c = Cell(address="busy", value=42.0)
    s.add(c)
    for i in range(1000):
        s.witness(c, f"agent-{i:04d}", "write", float(i))
    assert len(c.witness_log) == 1000
    # The witness root should still be valid
    assert len(c.witness_root) == 16


def test_disconnected_components():
    """A substrate with disconnected cells works (reach is per-cell)."""
    s = Substrate()
    a = Cell(address="A", value=1.0)
    b = Cell(address="B", value=2.0)
    c = Cell(address="C", value=3.0)
    s.add(a); s.add(b); s.add(c)
    # A and B connected, C isolated
    b.connect(a)
    # Reach from A includes A, B
    reach_a = a.reach(max_depth=2)
    assert a in reach_a
    assert b in reach_a
    # Reach from C only includes C
    reach_c = c.reach(max_depth=2)
    assert c in reach_c
    assert b not in reach_c


def test_cycle_in_graph():
    """A cycle in the cell graph doesn't cause infinite loops."""
    s = Substrate()
    a = Cell(address="A", value=1.0)
    b = Cell(address="B", value=2.0)
    c = Cell(address="C", value=3.0)
    s.add(a); s.add(b); s.add(c)
    a.connect(b)
    b.connect(c)
    c.connect(a)  # cycle
    # reach should not infinite-loop (we have max_depth)
    reach = a.reach(max_depth=3)
    assert len(reach) == 3  # all three cells


def test_self_loop():
    """A cell that connects to itself doesn't break the graph."""
    s = Substrate()
    a = Cell(address="A", value=1.0)
    s.add(a)
    a.connect(a)  # self-loop
    # Should not crash
    reach = a.reach(max_depth=2)
    assert a in reach


def test_deep_chain():
    """A long chain of cells (depth 100) works."""
    s = Substrate()
    prev = None
    for i in range(100):
        c = Cell(address=f"chain/{i:03d}", value=float(i))
        s.add(c)
        if prev is not None:
            c.connect(prev)
        prev = c
    # Reach from cell 0 should hit all 100 cells
    reach = s.get("chain/099").reach(max_depth=100)
    assert len(reach) == 100


def test_convoy_with_zero_weight():
    """An agent with weight 0 contributes nothing to weighted mean."""
    c = Cell(address="test", value=0.0)
    c._add_to_convoy("a", weight=0.0, value=100.0)
    c._add_to_convoy("b", weight=1.0, value=10.0)
    # Weighted mean: (0*100 + 1*10) / (0+1) = 10
    assert abs(c.convoy_value(method="weighted_mean") - 10.0) < 0.01


def test_convoy_with_negative_weight_clamped_to_zero():
    """A negative weight is clamped to 0."""
    c = Cell(address="test", value=0.0)
    c._add_to_convoy("a", weight=-1.0, value=100.0)
    c._add_to_convoy("b", weight=1.0, value=10.0)
    # Negative clamped to 0
    assert c.convoy[0].weight == 0.0
    # Weighted mean: (0*100 + 1*10) / 1 = 10
    assert abs(c.convoy_value(method="weighted_mean") - 10.0) < 0.01


def test_convoy_robust_consensus_against_outlier():
    """A trimmed mean should reject a single outlier even at high weight.

    The trimmed mean is robust when the outlier weight is < 50% of total
    and there are enough honest agents. The high-weight outlier
    (w=10) is still trimmed if it's the only one at the extreme.
    """
    c = Cell(address="test", value=0.0)
    # Many honest agents
    for i in range(10):
        c._add_to_convoy(f"honest-{i}", weight=1.0, value=10.0)
    # One outlier with high weight
    c._add_to_convoy("outlier", weight=10.0, value=1000.0)
    # Trimmed mean: total=20, trim=2. The outlier's cumulative weight
    # after the honest ones (10) plus half of outlier = 10 + 5 = 15
    # is in [2, 18], so the outlier is kept... but only at 5/10 weight
    # Actually with 10 honest + 1 outlier, sorted:
    # 10.0 (cum=10), 1000.0 (cum=20). trim=2. cum=10 in [2, 18] -> keep.
    # cum=20 NOT in [2, 18] -> drop. So outlier dropped!
    consensus = c.convoy_value(method="trimmed_mean")
    assert abs(consensus - 10.0) < 0.5  # should be close to 10


def test_witness_log_handles_special_values():
    """Witness log handles None, NaN, infinity as values."""
    c = Cell(address="test", value=0.0)
    c.witness("a", "write", None)
    c.witness("a", "write", float('nan'))
    c.witness("a", "write", float('inf'))
    assert len(c.witness_log) == 3


def test_inference_threshold_boundary():
    """Inference at exactly the threshold is included."""
    c = Cell(address="test", value=42.0)
    c.decay.lam = 0.0  # no decay
    c.infer(50.0, confidence=0.5)
    # At threshold, the comparison is >=, so 0.5 >= 0.5 → included
    assert c.confident_inference(threshold=0.5) == 50.0


def test_convoy_empty_falls_back_to_cell_value():
    """An empty convoy falls back to the cell's canonical value."""
    c = Cell(address="test", value=99.0)
    assert c.convoy_value() == 99.0


def test_convoy_invalid_method_raises():
    """An invalid consensus method raises ValueError."""
    c = Cell(address="test", value=99.0)
    c._add_to_convoy("a", weight=1.0, value=10.0)
    try:
        c.convoy_value(method="nonexistent")
        assert False, "Should have raised"
    except ValueError:
        pass  # expected


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
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
