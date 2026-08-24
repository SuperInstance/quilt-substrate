"""Tests for the 8 base primitives (paper 107 + cell-runtime)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Cell, Substrate, Vibe


def test_basic_value():
    c = Cell(address="test", value=42)
    assert c.value == 42
    assert c.address == "test"
    assert c.ticks == 0


def test_tick_holds():
    c = Cell(address="t", value=10)
    c.tick()
    assert c.value == 10
    assert c.ticks == 1


def test_wired_sum():
    a = Cell(address="a", value=1.0)
    b = Cell(address="b", value=2.0)
    c = Cell(address="c", value=0.0, jepa=lambda inputs: sum(v for v in inputs.values() if isinstance(v, (int, float))))
    c.connect(a)
    c.connect(b)
    s = Substrate()
    s.add(a).add(b).add(c)
    for _ in range(3):
        s.tick()
    assert c.value == 3.0


def test_murmur():
    c = Cell(address="m", value=1)
    assert c.murmur() is True
    assert c._murmur_count == 1


def test_double_entry():
    c = Cell(address="de", value=10)
    c.value = 20
    assert c.debit == 10
    assert c.credit == 20


def test_vibe_step():
    v = Vibe(pos=(0.0,), vel=(1.0,), acc=(0.5,))
    v2 = v.step(dt=2.0)
    # Math (paper 117, §2.5):
    # p_{t+1} = p_t + v_t dt + 0.5 a_t dt^2 = 0 + 1*2 + 0.5*0.5*4 = 3.0
    # v_{t+1} = (v_t + a_t dt) * (1 - damping) = (1 + 0.5*2) * (1 - 0.1) = 2.0 * 0.9 = 1.8
    assert v2.pos[0] == 3.0
    assert v2.vel[0] == 1.8  # damped (was 2.0 before damping was added)


def test_gc_phases():
    c = Cell(address="gc", value=1)
    assert c.gc() == "merge-similar"
    assert c.gc() == "decay-old"
    assert c.gc() == "prune-weak"


def test_reach():
    a = Cell(address="a", value=1)
    b = Cell(address="b", value=2)
    c = Cell(address="c", value=3)
    b.connect(a)
    c.connect(b)
    reached = a.reach(max_depth=2)
    assert b in reached
    assert c in reached


def test_to_dict():
    c = Cell(address="d", value=42)
    d = c.to_dict()
    assert d["address"] == "d"
    assert d["value"] == 42
    assert "vibe" in d


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
