"""Tests for the formal Opener ABC.

The Opener ABC (openers.py) provides:
- A formal interface (activate → Iterator[Event])
- A registry (register, get, all_openers)
- 4 default openers (chart, voice, gesture, witness)
- Pluggable extension: new openers can be added without modifying the core
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.openers import (
    Opener, ChartOpener, VoiceOpener, GestureOpener, WitnessOpener,
    register, get, all_openers,
)


def test_opener_abc_is_abstract():
    """Opener is an abstract base class — can't be instantiated directly."""
    try:
        Opener()
        assert False, "Opener() should have raised TypeError"
    except TypeError:
        pass  # expected


def test_chart_opener_yields_value_events():
    """ChartOpener yields 'value' events for each cell."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    s.add(Cell(address="b", value=2.0))
    opener = ChartOpener()
    events = list(opener.activate(s))
    assert len(events) == 2
    for e in events:
        assert e["kind"] == "value"
        assert "address" in e
        assert "value" in e
        assert "confidence" in e


def test_voice_opener_yields_speech_events():
    """VoiceOpener yields 'speech' events (TTS-friendly)."""
    s = Substrate()
    s.add(Cell(address="a", value=42.5))
    opener = VoiceOpener()
    events = list(opener.activate(s))
    assert len(events) == 1
    assert events[0]["kind"] == "speech"
    assert "42.50" in events[0]["text"]
    assert "fresh" in events[0]["text"]  # high confidence


def test_voice_opener_handles_empty_substrate():
    """VoiceOpener says 'empty' for empty substrate (Fable 06: gentle)."""
    s = Substrate()
    events = list(VoiceOpener().activate(s))
    assert len(events) == 1
    assert "empty" in events[0]["text"].lower()


def test_gesture_opener_yields_tap_events():
    """GestureOpener yields tap events for touch input."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    events = list(GestureOpener().activate(s))
    assert len(events) == 1
    assert events[0]["kind"] == "tap"
    assert events[0]["id"] == "a"
    assert "tap" in events[0]
    assert "long_press" in events[0]


def test_witness_opener_yields_witness_events():
    """WitnessOpener yields witness log events."""
    s = Substrate()
    c = Cell(address="a", value=42.0)
    s.add(c)
    s.witness(c, "reyes", "write", 42.0)
    s.witness(c, "skate", "read", 42.0)
    events = list(WitnessOpener().activate(s))
    assert len(events) == 2
    for e in events:
        assert e["kind"] == "witness"
        assert e["cell"] == "a"
        assert "agent" in e
        assert "action" in e


def test_registry_starts_with_defaults():
    """The registry starts with 4 default openers."""
    openers = all_openers()
    assert "chart" in openers
    assert "voice" in openers
    assert "gesture" in openers
    assert "witness" in openers


def test_registry_register_and_get():
    """A new opener can be registered and retrieved."""
    class CustomOpener(Opener):
        def activate(self, target):
            yield {"kind": "custom", "text": "hello"}

    register("custom", CustomOpener())
    opener = get("custom")
    events = list(opener.activate(Substrate()))
    assert len(events) == 1
    assert events[0]["kind"] == "custom"


def test_registry_get_unknown_raises():
    """Getting an unknown opener raises KeyError."""
    try:
        get("nonexistent")
        assert False, "Should have raised"
    except KeyError:
        pass


def test_opener_preview_default():
    """The default preview() describes the opener."""
    opener = ChartOpener()
    preview = opener.preview(Substrate())
    assert "ChartOpener" in preview
    assert "Substrate" in preview


def test_opener_works_on_single_cell():
    """An opener can be activated on a single Cell, not just a Substrate."""
    c = Cell(address="a", value=42.0)
    events = list(ChartOpener().activate(c))
    assert len(events) == 1
    assert events[0]["address"] == "a"


def test_extension_without_modifying_core():
    """A new opener can be added without modifying openers.py.

    This proves the architecture is pluggable.
    """
    class EEGOpener(Opener):
        """An opener that reads brainwave data and activates cells."""
        def activate(self, target):
            # In a real impl, this would read from an EEG device
            yield {"kind": "eeg", "alpha": 10.5, "beta": 22.3, "target": "frontal"}

    register("eeg", EEGOpener())
    events = list(get("eeg").activate(None))  # opener works on None
    assert events[0]["kind"] == "eeg"
    assert events[0]["alpha"] == 10.5


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
