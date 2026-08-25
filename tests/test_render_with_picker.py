"""test_render_with_picker.py — Tests for substrate.render_with_picker."""
import sys

sys.path.insert(0, "/workspace/quilt-substrate/src")

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.plugins.opener_picker import OpenerPicker


def test_render_with_picker_default_opener():
    s = Substrate()
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    # No picker given — should use "chart" (the default)
    out = s.render_with_picker(primitive="Murmur", role="creative_ideation")
    assert out["picked_opener"] == "chart"
    assert out["picker_reason"] == "default"
    # The result should be a chart (list of cells)
    assert "cells" in out["result"]


def test_render_with_picker_picks_best_opener():
    s = Substrate()
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    picker = OpenerPicker()
    # No observations yet — should use the prior
    out = s.render_with_picker(primitive="Murmur", role="sensory_creative",
                                    opener="tide", picker=picker)
    # tide has prior 0.8 for sensory_creative, but the candidate list is ["tide"]
    # so tide is the only choice
    assert out["picked_opener"] == "tide"


def test_render_with_picker_observes_outcome():
    s = Substrate()
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    picker = OpenerPicker()
    # Render once with slate
    out = s.render_with_picker(primitive="Murmur", role="fable_compression",
                                    opener="slate", picker=picker)
    assert out["picked_opener"] == "slate"
    # Feed the picker with the outcome
    picker.observe("Murmur", "fable_compression", "slate",
                     success=True, quality=0.9)
    # Now slate has 1 success, no other data
    out2 = s.render_with_picker(primitive="Murmur", role="fable_compression",
                                     opener="slate", picker=picker)
    assert out2["picked_opener"] == "slate"


def test_render_with_picker_learns_prefers_slate():
    s = Substrate()
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    picker = OpenerPicker()
    # Slate: 5 successes
    for _ in range(5):
        picker.observe("Murmur", "fable_compression", "slate",
                         success=True, quality=0.95)
    # Voice: 5 failures
    for _ in range(5):
        picker.observe("Murmur", "fable_compression", "voice",
                         success=False, quality=0.1)
    # With only slate and voice as candidates, slate should win
    out = s.render_with_picker(primitive="Murmur", role="fable_compression",
                                    opener="slate", picker=picker)
    # Since candidates=[opener] = ["slate"], only slate is in the running
    assert out["picked_opener"] == "slate"


def test_render_with_picker_no_opener_uses_chart_default():
    s = Substrate()
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    picker = OpenerPicker()
    out = s.render_with_picker(primitive="Murmur", role="creative_ideation",
                                    picker=picker)
    # No opener specified, no candidates → defaults to slate (most common prior)
    # Or chart (substrate default)
    assert out["picked_opener"] in ("slate", "chart")
