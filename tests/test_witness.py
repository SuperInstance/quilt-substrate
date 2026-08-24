"""Tests for the Witness primitive (paper 110)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from substrate import Cell, Substrate


def test_witness_log_initially_empty():
    c = Cell(address="c1", value=1.0)
    assert c.witness_log == []
    assert c.witness_root == "0" * 16


def test_witness_appends():
    c = Cell(address="c1", value=1.0)
    e = c.witness("agent-001", "read", 1.0)
    assert len(c.witness_log) == 1
    assert e.agent_id == "agent-001"
    assert e.action == "read"
    assert c.witness_root != "0" * 16


def test_witness_merkle_chain():
    c = Cell(address="c1", value=1.0)
    e1 = c.witness("agent-001", "read", 1.0)
    root_after_e1 = c._witness_root
    e2 = c.witness("agent-002", "write", 2.0)
    # e2's prev_hash is the root after e1 (which is a hash of e1's dict)
    assert e2.prev_hash == root_after_e1
    root_after_e2 = c._witness_root
    e3 = c.witness("agent-003", "inference", 3.0)
    # e3's prev_hash is the root after e2
    assert e3.prev_hash == root_after_e2
    # All entries should be in the log
    assert len(c.witness_log) == 3
    # The witness root should be different from the initial all-zeros
    assert c.witness_root != "0" * 16


def test_substrate_witness():
    s = Substrate()
    c = Cell(address="c1", value=42.0)
    s.add(c)
    s.witness(c, "agent-001", "read", 42.0)
    assert len(c.witness_log) == 1
    assert c.witness_log[0].action == "read"


def test_archeological_read():
    """The deep-time archeologist (scenario 09) can read the witness log."""
    s = Substrate()
    c = Cell(address="bay/A17", value=12.5)
    s.add(c)
    s.witness(c, "captain-reyes", "read", 12.5)
    s.witness(c, "drone-skate", "write", 12.5)
    log = s.render("witness", address="bay/A17")
    assert len(log) == 2
    assert log[0]["agent_id"] == "captain-reyes"
    assert log[1]["agent_id"] == "drone-skate"


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
