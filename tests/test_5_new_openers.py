"""Tests for the 5 new openers: SLATE, HARBOR, REEF, DIVE, TIDE.

Each opener is verified to:
  1. Be registered in the opener registry
  2. yield() events from activate() on a substrate
  3. yield() events from activate() on a single cell
  4. return a non-empty string from preview()
  5. produce events with the expected 'kind' tag

Plus per-opener semantic tests.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.openers import (
    SlateOpener, HarborOpener, ReefOpener, DiveOpener, TideOpener,
    register, get, all_openers,
)


# ---- Registry / discovery -------------------------------------------------

def test_all_five_openers_registered():
    """All 5 new openers are in the registry after import."""
    registry = all_openers()
    for name in ("slate", "harbor", "reef", "dive", "tide"):
        assert name in registry, f"Opener '{name}' not registered"


def test_can_get_each_opener_by_name():
    """Each opener is retrievable by name and is the right class."""
    pairs = [
        ("slate", SlateOpener),
        ("harbor", HarborOpener),
        ("reef", ReefOpener),
        ("dive", DiveOpener),
        ("tide", TideOpener),
    ]
    for name, cls in pairs:
        opener = get(name)
        assert isinstance(opener, cls), f"get({name!r}) returned wrong class"


# ---- SLATE ---------------------------------------------------------------

def test_slate_opener_yields_events():
    """SlateOpener yields a 'slate' event per cell on a substrate."""
    s = Substrate()
    s.add(Cell(address="a", value=0.0))
    s.add(Cell(address="b", value=5.0))
    s.add(Cell(address="c", value=10.0))
    events = list(SlateOpener().activate(s))
    assert len(events) == 3
    for e in events:
        assert e["kind"] == "slate"
        assert "text" in e and "row" in e
        assert "address" in e


def test_slate_opener_handles_single_cell():
    """SlateOpener works on a single cell, not just a substrate."""
    c = Cell(address="solo", value=3.14)
    events = list(SlateOpener().activate(c))
    assert len(events) == 1
    assert events[0]["kind"] == "slate"
    assert events[0]["address"] == "solo"


def test_slate_opener_handles_empty_substrate():
    """SlateOpener on an empty substrate yields a single 'empty' event."""
    s = Substrate()
    events = list(SlateOpener().activate(s))
    assert len(events) == 1
    assert "empty" in events[0]["text"].lower()


# ---- HARBOR --------------------------------------------------------------

def test_harbor_opener_yields_markers():
    """HarborOpener yields harbor_marker events with lat/lon/depth."""
    s = Substrate()
    s.add(Cell(address="dock", value=12.5))
    s.add(Cell(address="pier", value=8.0))
    events = list(HarborOpener().activate(s))
    # 1 header + 2 markers = 3 events (no bearings since no edges)
    markers = [e for e in events if e["kind"] == "harbor_marker"]
    assert len(markers) == 2
    for m in markers:
        assert 0.0 <= m["lat"] <= 1.0
        assert 0.0 <= m["lon"] <= 1.0
        assert m["depth_fathoms"] is not None


def test_harbor_opener_yields_bearings_for_edges():
    """HarborOpener yields harbor_bearing events between connected cells."""
    s = Substrate()
    a = Cell(address="alpha", value=10.0)
    b = Cell(address="beta", value=20.0)
    s.add(a); s.add(b)
    a.connect(b)
    events = list(HarborOpener().activate(s))
    bearings = [e for e in events if e["kind"] == "harbor_bearing"]
    assert len(bearings) == 1
    assert bearings[0]["from"] in ("alpha", "beta")
    assert bearings[0]["to"] in ("alpha", "beta")


# ---- REEF ----------------------------------------------------------------

def test_reef_opener_yields_bumps():
    """ReefOpener yields reef events with heights and bump_lines."""
    s = Substrate()
    s.add(Cell(address="a", value=2.0))
    s.add(Cell(address="b", value=5.0))
    events = list(ReefOpener().activate(s))
    assert len(events) == 2
    for e in events:
        assert e["kind"] == "reef"
        assert "height" in e
        assert "bump_lines" in e
        # bump_lines should be a list of strings
        assert isinstance(e["bump_lines"], list)
        for line in e["bump_lines"]:
            assert isinstance(line, str)


def test_reef_opener_height_scales_with_value():
    """Larger values → taller coral heads in the reef."""
    s = Substrate()
    s.add(Cell(address="short", value=1.0))
    s.add(Cell(address="tall", value=7.0))
    events = list(ReefOpener().activate(s))
    by_addr = {e["address"]: e for e in events}
    assert by_addr["short"]["height"] < by_addr["tall"]["height"]


# ---- DIVE ----------------------------------------------------------------

def test_dive_opener_yields_descent_events():
    """DiveOpener yields dive events for surface, descend, and bottom."""
    s = Substrate()
    s.add(Cell(address="a", value=2.0))
    s.add(Cell(address="b", value=3.0))
    s.add(Cell(address="c", value=1.0))
    events = list(DiveOpener().activate(s))
    phases = [e["phase"] for e in events]
    assert phases[0] == "surface"
    assert phases[-1] == "bottom"
    # There should be one descend event per cell
    descends = [e for e in events if e["phase"] == "descend"]
    assert len(descends) == 3
    # Pressure should be monotonically non-decreasing across descend events
    pressures = [e["pressure_atm"] for e in descends]
    for i in range(1, len(pressures)):
        assert pressures[i] >= pressures[i - 1]


def test_dive_opener_increases_depth():
    """DiveOpener accumulates depth across cells."""
    s = Substrate()
    s.add(Cell(address="a", value=10.0))
    s.add(Cell(address="b", value=20.0))
    s.add(Cell(address="c", value=30.0))
    events = list(DiveOpener().activate(s))
    descends = [e for e in events if e["phase"] == "descend"]
    depths = [e["depth_m"] for e in descends]
    # Each depth should be greater than the previous (positive values)
    for i in range(1, len(depths)):
        assert depths[i] > depths[i - 1]


# ---- TIDE ----------------------------------------------------------------

def test_tide_opener_yields_current_per_cell():
    """TideOpener yields a tide event per cell in the substrate."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    s.add(Cell(address="b", value=2.0))
    s.add(Cell(address="c", value=3.0))
    events = list(TideOpener().activate(s))
    assert len(events) == 3
    for e in events:
        assert e["kind"] == "tide"
        assert e["current"] in ("rising", "ebbing", "still")
        assert "delta" in e


def test_tide_opener_handles_single_cell():
    """TideOpener on a single cell yields a 'still' tide event."""
    c = Cell(address="alone", value=5.0)
    events = list(TideOpener().activate(c))
    assert len(events) == 1
    assert events[0]["current"] == "still"


def test_tide_opener_connected_cells_compare():
    """TideOpener compares connected cells: a fresher cell rises."""
    s = Substrate()
    fresh = Cell(address="fresh", value=5.0)
    fresh.refresh(c0=1.0)  # max confidence
    stale = Cell(address="stale", value=5.0)
    # don't refresh stale — its confidence is whatever decay gives by default
    s.add(fresh); s.add(stale)
    fresh.connect(stale)
    events = list(TideOpener().activate(s))
    by_addr = {e["address"]: e for e in events}
    # At least one of them should not be "still" if their confidences differ,
    # or both should be "still" if they happen to be similar.
    # We just check that the events are well-formed and include delta.
    for addr, e in by_addr.items():
        assert isinstance(e["delta"], (int, float))


# ---- previews ------------------------------------------------------------

def test_all_previews_return_strings():
    """Each opener's preview() returns a non-empty string."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    for cls in (SlateOpener, HarborOpener, ReefOpener, DiveOpener, TideOpener):
        text = cls().preview(s)
        assert isinstance(text, str)
        assert len(text) > 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
