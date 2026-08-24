"""Tests for Open Question 8: per-agent decay rate selection.

Different agents have different freshness requirements. The substrate
should let agents declare their decay rate per write, not per cell.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Substrate, Cell


def test_set_agent_decay_basic():
    """An agent can set its decay rate."""
    s = Substrate()
    s.set_agent_decay("chat", 0.1)
    assert s.get_agent_decay("chat") == 0.1


def test_get_agent_decay_default():
    """An agent with no decay set gets the default."""
    s = Substrate()
    assert s.get_agent_decay("unknown") == 1e-4


def test_set_agent_decay_negative_raises():
    """A negative decay rate raises ValueError."""
    s = Substrate()
    try:
        s.set_agent_decay("bad", -1.0)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_agent_decay_applies_on_write():
    """When an agent writes, the cell's decay rate is set to the agent's rate."""
    s = Substrate()
    s.set_agent_decay("chat", 0.1)
    c = Cell(address="test", value=42.0)
    s.add(c)
    s.witness(c, "chat", "write", 42.0)
    # The cell's decay rate should now be 0.1
    assert c.decay.lam == 0.1


def test_different_agents_different_decay_rates():
    """Different agents can have different decay rates on the same cell."""
    s = Substrate()
    s.set_agent_decay("chat", 0.1)
    s.set_agent_decay("sensor", 0.001)
    c = Cell(address="test", value=42.0)
    s.add(c)
    s.witness(c, "chat", "write", 42.0)
    assert c.decay.lam == 0.1
    # Sensor writes override
    s.witness(c, "sensor", "write", 43.0)
    assert c.decay.lam == 0.001


def test_chat_decays_fast():
    """A chat agent's data decays quickly (high λ)."""
    s = Substrate()
    s.set_agent_decay("chat", 0.1)
    c = Cell(address="test", value=42.0)
    s.add(c)
    s.witness(c, "chat", "write", 42.0)
    initial_conf = c.confidence
    time.sleep(0.5)
    new_conf = c.confidence
    # Should decay significantly in 0.5s with λ=0.1 (e^(-0.05) ≈ 0.95)
    # But we use time.time() which is more precise
    assert new_conf < initial_conf


def test_chart_decays_slow():
    """A chart agent's data decays slowly (low λ)."""
    s = Substrate()
    s.set_agent_decay("chart", 1e-6)
    c = Cell(address="test", value=42.0)
    s.add(c)
    s.witness(c, "chart", "write", 42.0)
    initial_conf = c.confidence
    time.sleep(0.1)
    new_conf = c.confidence
    # Should barely decay in 0.1s with λ=1e-6 (e^(-1e-7) ≈ 0.9999999)
    # Confidence should still be near 1.0
    assert new_conf > 0.999


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
