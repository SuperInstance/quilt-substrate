"""Tests for the new openers: MIDI, REST, MUD, PLATO.

These are the openers added in the architectural improvement (paper 121).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.openers import (
    MIDIOpener, RESTOpener, MUDOpener, PLATOOpener,
    register, get, all_openers,
)


def test_midi_opener_yields_notes():
    """The MIDI opener yields note events for each cell."""
    s = Substrate()
    s.add(Cell(address="a", value=10.0))
    events = list(MIDIOpener().activate(s))
    assert len(events) == 1
    e = events[0]
    assert e["kind"] == "midi"
    assert 0 <= e["note"] <= 127
    assert 0 <= e["velocity"] <= 127


def test_midi_opener_velocity_scales_with_value():
    """Larger values → larger MIDI velocity."""
    s = Substrate()
    s.add(Cell(address="low", value=1.0))
    s.add(Cell(address="high", value=10.0))
    events = list(MIDIOpener().activate(s))
    assert events[0]["velocity"] < events[1]["velocity"]


def test_rest_opener_yields_endpoints():
    """The REST opener yields endpoint events (GET, POST)."""
    s = Substrate()
    s.add(Cell(address="a", value=42.0))
    events = list(RESTOpener().activate(s))
    # 2 events per cell: GET + POST
    assert len(events) == 2
    methods = [e["method"] for e in events]
    assert "GET" in methods
    assert "POST" in methods


def test_rest_get_returns_value():
    """The REST GET event returns the cell's value."""
    s = Substrate()
    s.add(Cell(address="a", value=42.0))
    events = list(RESTOpener().activate(s))
    get_event = next(e for e in events if e["method"] == "GET")
    assert get_event["returns"]["value"] == 42.0
    assert get_event["returns"]["address"] == "a"


def test_mud_opener_yields_rooms():
    """The MUD opener yields room events with descriptions and exits."""
    s = Substrate()
    a = Cell(address="a", value=42.0)
    b = Cell(address="b", value=99.0)
    s.add(a); s.add(b)
    a.connect(b)
    events = list(MUDOpener().activate(s))
    assert len(events) == 2
    a_event = next(e for e in events if e["address"] == "a")
    assert "42" in a_event["description"]
    assert "b" in a_event["exits"]


def test_mud_opener_handles_single_cell():
    """The MUD opener handles a single cell (no substrate)."""
    c = Cell(address="only", value=42.0)
    events = list(MUDOpener().activate(c))
    assert len(events) == 1
    assert events[0]["kind"] == "mud_room"
    assert "only" in events[0]["address"]


def test_plato_opener_yields_lessons():
    """The PLATO opener yields lesson events with content."""
    s = Substrate()
    s.add(Cell(address="a", value="The cell is a system, not a value."))
    events = list(PLATOOpener().activate(s))
    assert len(events) == 1
    e = events[0]
    assert e["kind"] == "plato_lesson"
    assert "system" in e["content"]


def test_new_openers_registered():
    """The new openers are auto-registered when imported."""
    openers = all_openers()
    assert "midi" in openers
    assert "rest" in openers
    assert "mud" in openers
    assert "plato" in openers


def test_get_new_opener():
    """Get a new opener by name."""
    opener = get("midi")
    assert isinstance(opener, MIDIOpener)


def test_midi_opener_for_cell_with_no_value():
    """The MIDI opener handles a cell with None value."""
    c = Cell(address="empty", value=None)
    events = list(MIDIOpener().activate(c))
    assert len(events) == 1
    # Default velocity is 64 for non-numeric values
    assert events[0]["velocity"] == 64


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
