"""Tests for the 5 MORE new openers: BUOY, TRAWL, SHOAL, MOORING, GALE.

Each opener is verified to:
  1. Be registered in the opener registry
  2. yield events from activate() on a substrate
  3. yield events from activate() on a single cell
  4. return a non-empty string from preview()
  5. produce events with the expected 'kind' tag

Plus per-opener semantic tests.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.openers import (
    BuoyOpener, TrawlOpener, ShoalOpener, MooringOpener, GaleOpener,
    register, get, all_openers,
)


# ---- Registry / discovery -------------------------------------------------

def test_all_five_openers_registered():
    """All 5 new openers are in the registry after import."""
    registry = all_openers()
    for name in ("buoy", "trawl", "shoal", "mooring", "gale"):
        assert name in registry, f"Opener '{name}' not registered"


def test_can_get_each_opener_by_name():
    """Each opener is retrievable by name and is the right class."""
    pairs = [
        ("buoy", BuoyOpener),
        ("trawl", TrawlOpener),
        ("shoal", ShoalOpener),
        ("mooring", MooringOpener),
        ("gale", GaleOpener),
    ]
    for name, cls in pairs:
        opener = get(name)
        assert isinstance(opener, cls), f"get({name!r}) returned wrong class"


# ---- BUOY ----------------------------------------------------------------

def test_buoy_opener_yields_events_per_cell():
    """BuoyOpener yields one buoy event per cell plus a header."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    s.add(Cell(address="b", value=2.0))
    s.add(Cell(address="c", value=3.0))
    events = list(BuoyOpener().activate(s))
    buoys = [e for e in events if e["kind"] == "buoy"]
    assert len(buoys) == 3
    # All new cells start fresh (just created), so confidences should be ~1.0
    for e in buoys:
        assert "address" in e
        assert "navigable" in e
        assert "color" in e
        assert e["color"] in ("green", "red")
    # There must be a header at the end
    headers = [e for e in events if e["kind"] == "buoy_header"]
    assert len(headers) == 1
    assert headers[0]["total"] == 3


def test_buoy_opener_flags_low_confidence_as_red():
    """Cells with low confidence get red buoys (not navigable)."""
    s = Substrate()
    fresh = Cell(address="fresh", value=1.0)
    fresh.refresh(c0=1.0)
    stale = Cell(address="stale", value=1.0)
    # Don't refresh stale — its confidence will decay naturally
    s.add(fresh)
    s.add(stale)
    # Force the stale one to have a very low confidence by manipulating decay
    stale._decay.c0 = 0.1
    events = list(BuoyOpener(threshold=0.8).activate(s))
    buoys = {e["address"]: e for e in events if e["kind"] == "buoy"}
    assert buoys["fresh"]["navigable"] is True
    assert buoys["fresh"]["color"] == "green"
    assert buoys["stale"]["navigable"] is False
    assert buoys["stale"]["color"] == "red"


def test_buoy_opener_handles_empty_substrate():
    """BuoyOpener on an empty substrate yields a header and an empty event."""
    s = Substrate()
    events = list(BuoyOpener().activate(s))
    headers = [e for e in events if e["kind"] == "buoy_header"]
    empties = [e for e in events if e["kind"] == "buoy_empty"]
    assert len(headers) == 1
    assert headers[0]["total"] == 0
    assert len(empties) == 1


def test_buoy_opener_handles_single_cell():
    """BuoyOpener works on a single cell."""
    c = Cell(address="solo", value=1.0)
    events = list(BuoyOpener().activate(c))
    buoys = [e for e in events if e["kind"] == "buoy"]
    assert len(buoys) == 1
    assert buoys[0]["address"] == "solo"


# ---- TRAWL ---------------------------------------------------------------

def test_trawl_opener_catches_in_range_values():
    """TrawlOpener catches cells whose values are within the mesh range."""
    s = Substrate()
    s.add(Cell(address="in1", value=0.3))
    s.add(Cell(address="in2", value=0.5))
    s.add(Cell(address="out1", value=1.5))
    s.add(Cell(address="out2", value=-0.5))
    events = list(TrawlOpener(lo=0.0, hi=1.0).activate(s))
    catches = [e for e in events if e["kind"] == "trawl_catch"]
    bycatch = [e for e in events if e["kind"] == "trawl_bycatch"]
    assert len(catches) == 2
    assert len(bycatch) == 2
    caught_addrs = {c["address"] for c in catches}
    assert caught_addrs == {"in1", "in2"}
    # The summary should report 2 caught, 2 escaped
    summary = [e for e in events if e["kind"] == "trawl_summary"][0]
    assert summary["caught"] == 2
    assert summary["escaped"] == 2


def test_trawl_opener_ignores_non_numeric_values():
    """TrawlOpener treats non-numeric values as escaped (not caught)."""
    s = Substrate()
    s.add(Cell(address="num", value=0.5))
    s.add(Cell(address="str", value="hello"))
    events = list(TrawlOpener(lo=0.0, hi=1.0).activate(s))
    catches = [e for e in events if e["kind"] == "trawl_catch"]
    bycatch = [e for e in events if e["kind"] == "trawl_bycatch"]
    assert len(catches) == 1
    assert catches[0]["address"] == "num"
    assert len(bycatch) == 1
    assert bycatch[0]["address"] == "str"


def test_trawl_opener_handles_empty_substrate():
    """TrawlOpener on empty substrate yields a header and empty event."""
    s = Substrate()
    events = list(TrawlOpener(lo=0.0, hi=1.0).activate(s))
    headers = [e for e in events if e["kind"] == "trawl_header"]
    empties = [e for e in events if e["kind"] == "trawl_empty"]
    assert len(headers) == 1
    assert len(empties) == 1


def test_trawl_opener_swapped_bounds_are_normalized():
    """TrawlOpener normalizes lo > hi by swapping."""
    s = Substrate()
    s.add(Cell(address="x", value=0.5))
    # Pass lo > hi — should be normalized to lo=0, hi=1
    events = list(TrawlOpener(lo=1.0, hi=0.0).activate(s))
    header = [e for e in events if e["kind"] == "trawl_header"][0]
    assert header["lo"] == 0.0
    assert header["hi"] == 1.0


# ---- SHOAL ---------------------------------------------------------------

def test_shoal_opener_yields_depth_per_cell():
    """ShoalOpener yields one depth event per cell, plus a header."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    s.add(Cell(address="b", value=2.0))
    s.add(Cell(address="c", value=3.0))
    events = list(ShoalOpener().activate(s))
    depths = [e for e in events if e["kind"] == "shoal_depth"]
    assert len(depths) == 3
    for e in depths:
        assert "address" in e
        assert "depth" in e


def test_shoal_opener_flags_sudden_change_as_shoal():
    """A sudden jump in value should be flagged as a shoal."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    s.add(Cell(address="b", value=1.1))  # small change
    s.add(Cell(address="c", value=5.0))  # big change → shoal
    events = list(ShoalOpener(threshold=0.5).activate(s))
    shoals = [e for e in events if e["kind"] in ("shoal_rise", "shoal_drop")]
    assert len(shoals) == 1
    assert shoals[0]["address"] == "c"
    # The big jump is positive, so it should be a rise
    assert shoals[0]["kind"] == "shoal_rise"
    assert abs(shoals[0]["delta"] - 3.9) < 1e-6


def test_shoal_opener_flags_sudden_drop():
    """A sudden drop in value should be flagged as shoal_drop."""
    s = Substrate()
    s.add(Cell(address="a", value=5.0))
    s.add(Cell(address="b", value=1.0))  # big drop → shoal
    events = list(ShoalOpener(threshold=0.5).activate(s))
    shoals = [e for e in events if e["kind"] in ("shoal_rise", "shoal_drop")]
    assert len(shoals) == 1
    assert shoals[0]["kind"] == "shoal_drop"


def test_shoal_opener_handles_non_numeric_values():
    """ShoalOpener gracefully handles non-numeric values."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    s.add(Cell(address="b", value="hi"))  # non-numeric
    s.add(Cell(address="c", value=2.0))
    events = list(ShoalOpener().activate(s))
    # Should yield a depth event with depth=None for "b"
    depth_for_b = [e for e in events if e["kind"] == "shoal_depth" and e["address"] == "b"]
    assert len(depth_for_b) == 1
    assert depth_for_b[0]["depth"] is None


# ---- MOORING -------------------------------------------------------------

def test_mooring_opener_ranks_busiest_cells_first():
    """MooringOpener ranks cells by degree, busiest first."""
    s = Substrate()
    # 'hub' connects to all 3 others; others are leaves
    hub = Cell(address="hub", value=1.0)
    a = Cell(address="a", value=2.0)
    b = Cell(address="b", value=3.0)
    c = Cell(address="c", value=4.0)
    s.add(hub); s.add(a); s.add(b); s.add(c)
    hub.connect(a)
    hub.connect(b)
    hub.connect(c)
    events = list(MooringOpener().activate(s))
    moorings = [e for e in events if e["kind"] == "mooring"]
    assert len(moorings) >= 1
    # hub has the highest degree
    hub_event = next(m for m in moorings if m["address"] == "hub")
    assert hub_event["rank"] == 0
    assert hub_event["degree"] == 3
    # Busyness label should be one of the known categories
    assert hub_event["busyness"] in (
        "ghost port", "quiet cove", "fishing village", "busy port", "metropolis"
    )


def test_mooring_opener_ghost_port_label():
    """A cell with no neighbors gets a 'ghost port' busyness."""
    s = Substrate()
    s.add(Cell(address="alone", value=1.0))
    events = list(MooringOpener().activate(s))
    moorings = [e for e in events if e["kind"] == "mooring"]
    assert len(moorings) == 1
    assert moorings[0]["busyness"] == "ghost port"
    assert moorings[0]["degree"] == 0


def test_mooring_opener_handles_single_cell():
    """MooringOpener on a single cell yields one mooring event."""
    c = Cell(address="solo", value=1.0)
    events = list(MooringOpener().activate(c))
    moorings = [e for e in events if e["kind"] == "mooring"]
    assert len(moorings) == 1
    assert moorings[0]["address"] == "solo"


def test_mooring_opener_respects_top_limit():
    """MooringOpener's `top` parameter limits the number of full mooring events."""
    s = Substrate()
    # Create a hub and 5 leaves
    hub = Cell(address="hub", value=1.0)
    s.add(hub)
    for i in range(5):
        leaf = Cell(address=f"leaf{i}", value=float(i))
        s.add(leaf)
        hub.connect(leaf)
    events = list(MooringOpener(top=2).activate(s))
    moorings = [e for e in events if e["kind"] == "mooring"]
    # At most `top` (2) moorings should be yielded
    assert len(moorings) == 2
    # And the top one should be the hub (degree 5)
    assert moorings[0]["address"] == "hub"


# ---- GALE ----------------------------------------------------------------

def test_gale_opener_handles_no_inferences():
    """GaleOpener yields gale_clear events for cells with no inference."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    s.add(Cell(address="b", value=2.0))
    events = list(GaleOpener().activate(s))
    clears = [e for e in events if e["kind"] == "gale_clear"]
    assert len(clears) == 2
    for c in clears:
        assert c["inference"] is None


def test_gale_opener_flags_low_inference_confidence():
    """A cell with a stale inference is flagged as a gale warning."""
    s = Substrate()
    c = Cell(address="a", value=1.0)
    s.add(c)
    c.infer("predicted", confidence=0.5)  # set inference at conf 0.5
    # Force the inference confidence to be very low
    c._inference_confidence = 0.05
    events = list(GaleOpener(danger=0.3).activate(s))
    warnings = [e for e in events if e["kind"] == "gale_warning"]
    assert len(warnings) == 1
    assert warnings[0]["address"] == "a"
    assert warnings[0]["storm"] in ("hurricane", "gale", "squall")
    assert "GALE" in warnings[0]["warning"]


def test_gale_opener_clear_for_fresh_inference():
    """A fresh inference is clear, not a gale."""
    s = Substrate()
    c = Cell(address="a", value=1.0)
    s.add(c)
    c.infer("predicted", confidence=0.95)
    events = list(GaleOpener(danger=0.3).activate(s))
    warnings = [e for e in events if e["kind"] == "gale_warning"]
    clears = [e for e in events if e["kind"] == "gale_clear"]
    assert len(warnings) == 0
    assert len(clears) == 1
    assert clears[0]["inference"] == "predicted"
    assert clears[0]["inference_confidence"] >= 0.3


def test_gale_opener_handles_empty_substrate():
    """GaleOpener on empty substrate yields a header and an empty event."""
    s = Substrate()
    events = list(GaleOpener().activate(s))
    headers = [e for e in events if e["kind"] == "gale_header"]
    empties = [e for e in events if e["kind"] == "gale_empty"]
    assert len(headers) == 1
    assert len(empties) == 1


def test_gale_opener_handles_single_cell():
    """GaleOpener works on a single cell."""
    c = Cell(address="solo", value=1.0)
    events = list(GaleOpener().activate(c))
    headers = [e for e in events if e["kind"] == "gale_header"]
    assert len(headers) == 1
    assert headers[0]["n_cells"] == 1


# ---- previews ------------------------------------------------------------

def test_all_previews_return_strings():
    """Each opener's preview() returns a non-empty string."""
    s = Substrate()
    s.add(Cell(address="a", value=1.0))
    for cls in (BuoyOpener, TrawlOpener, ShoalOpener, MooringOpener, GaleOpener):
        text = cls().preview(s)
        assert isinstance(text, str)
        assert len(text) > 0


def test_all_previews_handle_single_cell():
    """Each opener's preview() also works on a single cell."""
    c = Cell(address="solo", value=1.0)
    for cls in (BuoyOpener, TrawlOpener, ShoalOpener, MooringOpener, GaleOpener):
        text = cls().preview(c)
        assert isinstance(text, str)
        assert len(text) > 0


# ---- constructor argument preservation ----------------------------------

def test_buoy_threshold_preserved():
    """BuoyOpener stores its threshold argument."""
    o = BuoyOpener(threshold=0.5)
    assert o.threshold == 0.5


def test_trawl_bounds_preserved():
    """TrawlOpener stores its lo/hi arguments."""
    o = TrawlOpener(lo=0.1, hi=0.9)
    assert o.lo == 0.1
    assert o.hi == 0.9


def test_shoal_threshold_preserved():
    """ShoalOpener stores its threshold argument."""
    o = ShoalOpener(threshold=1.0)
    assert o.threshold == 1.0


def test_mooring_top_preserved():
    """MooringOpener stores its top argument."""
    o = MooringOpener(top=3)
    assert o.top == 3


def test_gale_danger_preserved():
    """GaleOpener stores its danger argument."""
    o = GaleOpener(danger=0.2)
    assert o.danger == 0.2


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
