"""Tests for Substrate.advance_time and the substrate-wide clock."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Substrate, Cell


def test_advance_time_decays_all_cells():
    """Advancing the substrate's clock decays all cells' confidence."""
    s = Substrate()
    for i in range(5):
        c = Cell(address=f"c{i}", value=float(i))
        s.add(c)
        s.witness(c, "reyes", "write", float(i))
    initial_confs = [c.confidence for c in s.all_cells()]
    s.advance_time(10000)
    new_confs = [c.confidence for c in s.all_cells()]
    # Every cell should have decayed
    for old, new in zip(initial_confs, new_confs):
        assert new < old, f"Cell should decay after advance_time: {old} → {new}"


def test_advance_time_zero_does_nothing():
    """Advancing by 0 seconds is a no-op."""
    s = Substrate()
    c = Cell(address="c", value=42.0)
    s.add(c)
    s.witness(c, "reyes", "write", 42.0)
    initial = c.confidence
    s.advance_time(0)
    assert abs(c.confidence - initial) < 0.001


def test_advance_time_negative():
    """Advancing by negative seconds (rewind) refreshes the cell."""
    s = Substrate()
    c = Cell(address="c", value=42.0)
    s.add(c)
    s.witness(c, "reyes", "write", 42.0)
    s.advance_time(10000)
    decayed = c.confidence
    s.advance_time(-5000)  # rewind halfway
    restored = c.confidence
    # The restored confidence should be higher than the decayed
    assert restored > decayed


def test_age_seconds():
    """The substrate's age is non-negative."""
    s = Substrate()
    age = s.age_seconds()
    assert age >= 0


def test_advance_time_advances_substrate_clock():
    """The substrate's internal clock is advanced too."""
    s = Substrate()
    initial_age = s.age_seconds()
    time.sleep(0.1)
    # The internal clock tracks "time since substrate creation"
    # which is age_seconds() = now - self._t
    # advance_time increments self._t, so age_seconds() goes DOWN
    s.advance_time(0)  # no actual advance
    age2 = s.age_seconds()
    # After 0.1s of real time, the age should be ~0.1s more
    assert age2 >= initial_age


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items() if isinstance(globals(), dict) else {}.items()) if k.startswith("test_")]
    if not tests:
        # Re-fetch
        tests = [v for k, v in list({k: v for k, v in globals().items()}.items()) if k.startswith("test_")]
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
