"""
openers.py — The Opener ABC.

An opener is a function that renders a cell (or substrate) as something
human or machine consumable. The substrate has 10 openers (chart, list,
tensor, witness, convoy, graph, voice, telnet, gesture, flowchart).

This module provides the formal `Opener` ABC and a registry for
adding new openers.

Fable 06 (Grandmother): the opener should be usable by anyone, even
a 91-year-old. The Opener ABC makes it easy to add gentle openers.

Fable 11 (Opener completeness, paper 117, Theorem 5): for any subset
of the cell's 14-tuple, there exists an opener. The ABC is the
mechanism for that.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .substrate import Cell, Substrate


class Opener(ABC):
    """An opener is a function that renders a cell or substrate.

    Subclasses must implement `activate()`, which returns an Iterator
    of `Event`s. Events are dicts with at least a 'kind' key.

    Subclasses can optionally implement `preview()` to give a static
    description of what the opener will do (for UI hints, accessibility).
    """

    @abstractmethod
    def activate(self, target) -> Iterator[Dict[str, Any]]:
        """Activate the opener on a cell or substrate.

        Yields events as dicts. Common kinds:
        - "value" — a cell's value
        - "field" — a cell's field (address, confidence, etc.)
        - "edge" — a connection in the graph
        - "witness" — a witness log entry
        - "speech" — a phrase for TTS
        - "tap" — a tappable region
        """
        ...

    def preview(self, target) -> str:
        """A short description of what the opener does.

        For UI tooltips, for accessibility, for "what does this opener do?"
        """
        return f"{self.__class__.__name__} on {target.__class__.__name__}"


class ChartOpener(Opener):
    """Renders as a chart. Each cell becomes a point on the chart."""

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            for cell in target.all_cells():
                yield {"kind": "value", "address": cell.address, "value": cell.value, "confidence": cell.confidence}
        else:
            yield {"kind": "value", "address": target.address, "value": target.value, "confidence": target.confidence}


class VoiceOpener(Opener):
    """Renders as text suitable for TTS. Each cell becomes a phrase.

    Fable 06 (Grandmother): gentle, plain language, no jargon.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = sorted(target.all_cells(), key=lambda c: c.address)
        else:
            cells = [target]
        if not cells:
            yield {"kind": "speech", "text": "Empty substrate."}
            return
        for cell in cells:
            val = cell.value
            if isinstance(val, float):
                val_str = f"{val:.2f}"
            else:
                val_str = str(val)
            conf = "fresh" if cell.confidence > 0.7 else "stale"
            yield {"kind": "speech", "text": f"Cell {cell.address}: {val_str}. {conf}."}


class GestureOpener(Opener):
    """Renders as a JSON description of touch interactions.

    Each cell becomes a tappable region with tap, long-press, and swipe.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = target.all_cells()
        else:
            cells = [target]
        for cell in cells:
            yield {
                "kind": "tap",
                "id": cell.address,
                "tap": {"action": "observe", "target": cell.address},
                "long_press": {"action": "witness", "target": cell.address},
                "swipe_right": {"action": "refresh", "target": cell.address},
            }


class WitnessOpener(Opener):
    """Renders the witness log as a stream of events."""

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate, Cell
        if isinstance(target, Substrate):
            for cell in target.all_cells():
                for entry in cell.witness_log:
                    yield {
                        "kind": "witness",
                        "cell": cell.address,
                        "ts": entry.ts,
                        "agent": entry.agent_id,
                        "action": entry.action,
                        "value_hash": entry.value_hash,
                    }
        else:
            for entry in target.witness_log:
                yield {
                    "kind": "witness",
                    "cell": target.address,
                    "ts": entry.ts,
                    "agent": entry.agent_id,
                    "action": entry.action,
                    "value_hash": entry.value_hash,
                }


# -- Registry ---------------------------------------------------------------

_REGISTRY: Dict[str, Opener] = {}


def register(name: str, opener: Opener) -> None:
    """Register an opener by name.

    Example:
        register("voice", VoiceOpener())
        register("chart", ChartOpener())

    Use the registered opener:
        for event in _REGISTRY["voice"].activate(substrate):
            print(event)
    """
    _REGISTRY[name] = opener


def get(name: str) -> Opener:
    """Get a registered opener by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Opener '{name}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def all_openers() -> Dict[str, Opener]:
    """Return all registered openers."""
    return dict(_REGISTRY)


# Default openers
def _register_defaults():
    register("chart", ChartOpener())
    register("voice", VoiceOpener())
    register("gesture", GestureOpener())
    register("witness", WitnessOpener())


_register_defaults()



class MIDIOpener(Opener):
    """Renders cells as MIDI notes.

    Each cell becomes a note on the MIDI channel. The cell's value
    is the note's velocity, the address is the note's pitch (hashed).

    Fable 10 (Conductor): the substrate as an orchestra, the wave primitive.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = target.all_cells()
        else:
            cells = [target]
        for cell in cells:
            # Map address to a MIDI note (0-127)
            note = sum(ord(c) for c in cell.address) % 128
            # Map value to velocity (0-127)
            try:
                velocity = max(0, min(127, int(float(cell.value) * 12)))
            except (TypeError, ValueError):
                velocity = 64  # default
            yield {
                "kind": "midi",
                "note": note,
                "velocity": velocity,
                "channel": 0,
                "duration_ms": 100,
                "address": cell.address,
            }


class RESTOpener(Opener):
    """Renders cells as REST resources.

    Each cell becomes a REST endpoint. GET /cells/{address} returns the
    cell's value, headers, and links to related cells. POST writes,
    DELETE removes. Fable 11 (Paper and the Tablet): REST is the
    silence that lets the system be honest.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = target.all_cells()
        else:
            cells = [target]
        # Base path
        base = "/api/v1"
        for cell in cells:
            yield {
                "kind": "rest",
                "method": "GET",
                "path": f"{base}/cells/{cell.address}",
                "returns": {
                    "address": cell.address,
                    "value": cell.value,
                    "confidence": cell.confidence,
                    "canonical": cell.canonical,
                },
            }
            yield {
                "kind": "rest",
                "method": "POST",
                "path": f"{base}/cells/{cell.address}/witness",
                "body": {"agent": "<agent_id>", "action": "read|write|inference"},
            }


class MUDOpener(Opener):
    """Renders the substrate as a Multi-User Dungeon (MUD) room.

    Each cell is a room. Connections between cells are exits.
    The MUD is a textual world where you can navigate by typing
    compass directions.

    Fable 21 (Compass and the Graph): the substrate is a graph you
    can walk. The MUD makes the walk textual and explorable.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if not isinstance(target, Substrate):
            yield {"kind": "mud_room", "address": target.address, "description": f"You are at {target.address}."}
            return
        cells = target.all_cells()
        for cell in cells:
            exits = []
            for neighbor in cell.neighbors():
                if neighbor.address in [c.address for c in cells]:
                    exits.append(neighbor.address)
            yield {
                "kind": "mud_room",
                "address": cell.address,
                "description": f"You are at {cell.address}. Value: {cell.value}. Confidence: {cell.confidence:.2f}.",
                "exits": exits,
            }


class PLATOOpener(Opener):
    """Renders the substrate as a PLATO-style lesson.

    Each cell is a lesson unit. The PLATO system (1970s) was the
    first multi-user educational computer system. Its lessons were
    cell-like: discrete units, with prerequisites and followups.

    Fable 06 (Grandmother): the PLATO opener is for teaching.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = target.all_cells()
        else:
            cells = [target]
        for cell in cells:
            # The cell's value is the lesson content
            yield {
                "kind": "plato_lesson",
                "address": cell.address,
                "title": f"Lesson: {cell.address}",
                "content": str(cell.value),
                "prerequisites": [n.address for n in cell.neighbors()
                                  if isinstance(n, type(cell))],
                "next": [n.address for n in cell.neighbors()
                         if isinstance(n, type(cell))],
            }


# Register the new openers
# already imported above
register("midi", MIDIOpener())
register("rest", RESTOpener())
register("mud", MUDOpener())
register("plato", PLATOOpener())


# -- 5 new openers: SLATE, HARBOR, REEF, DIVE, TIDE -------------------------

class SlateOpener(Opener):
    """Renders the substrate as a hand-drawn ASCII chart on a paper slate.

    Like a notepad passed around a workshop, each cell becomes a small
    drawing: a bar (for numeric values) or a glyph (for non-numeric
    values), all laid out on a grid that looks like it was sketched by
    hand. The "paper" is finite — values that overflow the slate are
    capped with a tilde `~`.

    Fable 06 (Grandmother): the slate is a paper notebook, the gentlest
    opener of all — anyone can read it, no screen, no sound, no
    gestures. Just paper and pencil.
    """

    # Glyph ramp, low to high: . , : ; o O 0 8 @ #
    _GLYPHS = ".,:;oO08@#"

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = sorted(target.all_cells(), key=lambda c: c.address)
        else:
            cells = [target]
        if not cells:
            yield {"kind": "slate", "row": 0, "text": "(empty slate)"}
            return
        # Normalize numeric values to [0, 1] across the cells, so the
        # picture uses the full glyph range.
        numeric_vals = []
        for c in cells:
            v = c.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_vals.append(float(v))
        if numeric_vals:
            lo, hi = min(numeric_vals), max(numeric_vals)
            span = (hi - lo) or 1.0
        else:
            lo, hi, span = 0.0, 1.0, 1.0
        for i, cell in enumerate(cells):
            v = cell.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                norm = (float(v) - lo) / span
                idx = max(0, min(len(self._GLYPHS) - 1, int(norm * (len(self._GLYPHS) - 1))))
                glyph = self._GLYPHS[idx]
                bar = glyph * (idx + 1)
            else:
                # Non-numeric: just print the value as a glyph
                bar = "~"
            line = f"{cell.address:>6} | {bar:<12} | conf={cell.confidence:.2f}"
            yield {"kind": "slate", "row": i, "text": line, "address": cell.address}

    def preview(self, target) -> str:
        return "A hand-drawn paper notebook. Each cell becomes a small ASCII bar chart sketch."


class HarborOpener(Opener):
    """Renders the substrate as a nautical map of harbors and depths.

    Each cell becomes a marker on a chart with coordinates (derived
    from the cell's address) and a depth (the cell's value, taken as
    fathoms). Connections between cells are drawn as bearings between
    harbors.

    Fable 21 (Compass and the Graph): the substrate is a chart you can
    sail. The harbor opener makes the chart literal — coordinates,
    depths, bearings.
    """

    def _coords(self, address: str) -> Tuple[float, float]:
        # Deterministic 2D coords from the address string: sum of ordinals
        # split into two halves → a (lat, lon)-like pair in [0, 1].
        s = sum(ord(c) for c in address)
        lat = (s % 1000) / 1000.0
        lon = ((s // 7) % 1000) / 1000.0
        return (lat, lon)

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = target.all_cells()
            yield {
                "kind": "harbor_header",
                "title": "Nautical chart of the substrate",
                "num_harbors": len(cells),
            }
            for cell in cells:
                lat, lon = self._coords(cell.address)
                # Depth: numeric → fathoms; otherwise unknown
                if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                    depth = float(cell.value)
                else:
                    depth = None
                yield {
                    "kind": "harbor_marker",
                    "address": cell.address,
                    "lat": round(lat, 3),
                    "lon": round(lon, 3),
                    "depth_fathoms": depth,
                    "confidence": cell.confidence,
                    "anchored": cell.canonical,
                }
            # Bearings: edges between cells
            for cell in cells:
                for neighbor in cell.neighbors():
                    if neighbor.address in [c.address for c in cells] and neighbor.address > cell.address:
                        a_lat, a_lon = self._coords(cell.address)
                        b_lat, b_lon = self._coords(neighbor.address)
                        yield {
                            "kind": "harbor_bearing",
                            "from": cell.address,
                            "to": neighbor.address,
                            "delta_lat": round(b_lat - a_lat, 3),
                            "delta_lon": round(b_lon - a_lon, 3),
                        }
        else:
            lat, lon = self._coords(target.address)
            depth = target.value if isinstance(target.value, (int, float)) and not isinstance(target.value, bool) else None
            yield {
                "kind": "harbor_marker",
                "address": target.address,
                "lat": round(lat, 3),
                "lon": round(lon, 3),
                "depth_fathoms": depth,
                "confidence": target.confidence,
            }

    def preview(self, target) -> str:
        return "A nautical map. Each cell becomes a harbor marker with coordinates and depth in fathoms."


class ReefOpener(Opener):
    """Renders the substrate as a 3D coral reef.

    Each cell becomes a coral head on the sea floor. The cell's value
    is its height (in cubits above the floor); confidence is its
    width. Canonical cells are tall mature coral; stale cells are
    bleached. The reef is laid out in a grid (sorted by address) and
    each cell gets a 3D-ish ASCII bump.

    Fable 11 (Paper and the Tablet): the reef is a living record. The
    shape of the reef tells you the history of the substrate — which
    cells have grown tall, which have bleached.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = sorted(target.all_cells(), key=lambda c: c.address)
        else:
            cells = [target]
        if not cells:
            yield {"kind": "reef", "row": 0, "text": "~ empty ocean ~"}
            return
        # Build a grid: cells in a single row, sorted. Heights = values.
        # Cap height at 8 cubits for display.
        for i, cell in enumerate(cells):
            v = cell.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                height = max(0, min(8, int(round(float(v)))))
            else:
                height = 0
            # Bleached (stale) coral: low confidence uses dotted outline
            # Mature (canonical): solid block
            if cell.canonical:
                tip = "*" if height > 0 else "."
                stem = "|" * height
            else:
                tip = "o" if height > 0 else "."
                stem = ":" * height
            width = max(1, int(round(cell.confidence * 5)))
            base = "~" * width
            # Build the bump from top to bottom
            lines = []
            for h in range(height, 0, -1):
                if h == height:
                    lines.append(f"{' ' * (4 - h)}{tip}{' ' * (h - 1)}")
                else:
                    lines.append(f"{' ' * (4 - h)}{stem[h - 1]}{' ' * (h - 1)}")
            lines.append(f"   {base}")
            yield {
                "kind": "reef",
                "address": cell.address,
                "height": height,
                "width": width,
                "bleached": not cell.canonical,
                "bump_lines": lines,
            }

    def preview(self, target) -> str:
        return "A coral reef seen from the side. Each cell is a coral head — its height is the value, its width the confidence."


class DiveOpener(Opener):
    """Renders the substrate as an underwater descent.

    The diver goes cell by cell, descending deeper. Each cell's value
    becomes the pressure (in atmospheres) at that depth. The deeper
    you go, the more pressure; the more pressure, the more events you
    hear. Stale cells (low confidence) produce the muffled sounds of
    a deep, quiet place.

    Fable 19 (Oracle) + Fable 22 (Sundial): as you descend, time
    passes. Old inferences fade. The dive is a journey through the
    substrate's vertical.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = sorted(target.all_cells(), key=lambda c: c.address)
        else:
            cells = [target]
        if not cells:
            yield {"kind": "dive", "phase": "surface", "text": "The ocean is empty. Nothing to dive into."}
            return
        yield {
            "kind": "dive",
            "phase": "surface",
            "text": "You stand on the boat. The substrate lies beneath.",
            "depth_m": 0,
            "pressure_atm": 1.0,
        }
        depth = 0.0
        for i, cell in enumerate(cells):
            v = cell.value
            # Each cell adds to the depth
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                step = max(0.0, float(v))
            else:
                step = 1.0  # non-numeric: descend one meter
            depth += step
            pressure = 1.0 + depth / 10.0  # ~1 atm per 10 m
            # Sounds: higher confidence → clearer sounds
            if cell.confidence > 0.7:
                sound = "..."
            elif cell.confidence > 0.3:
                sound = ".."
            else:
                sound = "."
            # Bubbles rise as pressure changes
            yield {
                "kind": "dive",
                "phase": "descend",
                "address": cell.address,
                "depth_m": round(depth, 2),
                "pressure_atm": round(pressure, 2),
                "sound": sound,
                "line": f"{depth:6.1f}m | {pressure:4.2f} atm | {cell.address:>6} ({cell.value}) {sound*3}",
            }
        yield {
            "kind": "dive",
            "phase": "bottom",
            "text": "You have reached the bottom. The substrate is fully descended.",
            "depth_m": round(depth, 2),
            "pressure_atm": round(1.0 + depth / 10.0, 2),
        }

    def preview(self, target) -> str:
        return "An underwater descent. Each cell is a layer of the sea — its value is the depth, the pressure rises as you go deeper."


class TideOpener(Opener):
    """Renders the substrate as a tide — showing which way it's flowing.

    Each cell is compared to the average of its neighbors. If a cell
    is fresher (more canonical) than its neighbors, the tide is
    rising there. If a cell is staler (lower confidence) than its
    neighbors, the tide is ebbing. The opener yields a stream of
    "current" events: where the tide is going, and how fast.

    Fable 19 (Oracle) + Fable 22 (Sundial): the tide is the
    substrate's tendency — its direction in time. A fresh cell
    surrounded by stale cells is a high tide, a strong inference; a
    stale cell surrounded by fresh cells is a low tide, a memory
    fading.
    """

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if not isinstance(target, Substrate):
            yield {
                "kind": "tide",
                "address": target.address,
                "current": "still",
                "delta": 0.0,
                "note": "Single cell — no current to measure.",
            }
            return
        cells = target.all_cells()
        # Build address -> cell map
        by_addr = {c.address: c for c in cells}
        for cell in cells:
            neighbors = cell.neighbors()
            in_sub = [n for n in neighbors if n.address in by_addr]
            if not in_sub:
                yield {
                    "kind": "tide",
                    "address": cell.address,
                    "current": "still",
                    "delta": 0.0,
                    "note": "No neighbors in substrate.",
                }
                continue
            avg_conf = sum(n.confidence for n in in_sub) / len(in_sub)
            delta = cell.confidence - avg_conf
            if delta > 0.1:
                current = "rising"
                emoji = "▲"
            elif delta < -0.1:
                current = "ebbing"
                emoji = "▼"
            else:
                current = "still"
                emoji = "—"
            yield {
                "kind": "tide",
                "address": cell.address,
                "current": current,
                "delta": round(delta, 4),
                "cell_confidence": round(cell.confidence, 4),
                "neighbor_avg": round(avg_conf, 4),
                "icon": emoji,
                "line": f"{cell.address:>6} {emoji} {current:>7}  (Δ={delta:+.3f})",
            }

    def preview(self, target) -> str:
        return "A tide chart. Each cell is compared to its neighbors — is the tide rising (fresher) or ebbing (staler)?"


# Register the 5 new openers
register("slate", SlateOpener())
register("harbor", HarborOpener())
register("reef", ReefOpener())
register("dive", DiveOpener())
register("tide", TideOpener())


# -- 5 MORE new openers: BUOY, TRAWL, SHOAL, MOORING, GALE ------------------

class BuoyOpener(Opener):
    """Renders the substrate as a channel of channel-marker buoys.

    A buoy is a floating marker that says "you can sail here safely".
    This opener flags every cell whose confidence is above a navigable
    threshold (default 0.8) as a green buoy; cells below are red buoys
    or uncharted water. The opener yields one event per cell plus a
    header and a summary — a chart you can sail by.

    Fable 11 (Paper and the Tablet): the buoy is a marker of trust. It
    does not change the substrate; it just *declares* which cells you
    can rely on. Like a paper map's legend, the buoy separates the
    chart from the chart's claim about itself.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = target.all_cells()
        else:
            cells = [target]
        if not cells:
            yield {"kind": "buoy_header", "threshold": self.threshold, "navigable": 0, "total": 0}
            yield {"kind": "buoy_empty", "text": "No buoys. The channel is empty."}
            return
        navigable = 0
        for cell in cells:
            conf = cell.confidence
            is_navigable = conf >= self.threshold
            if is_navigable:
                navigable += 1
                color = "green"
                glyph = "○"  # green buoy
            else:
                color = "red"
                glyph = "●"  # red buoy
            yield {
                "kind": "buoy",
                "address": cell.address,
                "confidence": round(conf, 4),
                "navigable": is_navigable,
                "color": color,
                "glyph": glyph,
                "line": f"{cell.address:>6}  {glyph}  conf={conf:.2f}  "
                        f"{'NAVIGABLE' if is_navigable else 'uncharted'}",
            }
        yield {
            "kind": "buoy_header",
            "threshold": self.threshold,
            "navigable": navigable,
            "total": len(cells),
            "summary": f"{navigable}/{len(cells)} cells navigable at conf ≥ {self.threshold}",
        }

    def preview(self, target) -> str:
        return f"A channel of buoys. Cells with confidence ≥ {self.threshold} are green (navigable); below are red (uncharted)."


class TrawlOpener(Opener):
    """Renders the substrate as the catch of a fishing trawl.

    A trawl net drags through the substrate and collects only the cells
    whose numeric value falls within a given range — like a net with a
    mesh size. The opener yields each "caught" cell as a fish in the
    net, plus a header with the haul count and a "bycatch" event for
    cells that escaped the net.

    Fable 21 (Compass and the Graph): the trawl is a filter that takes
    the substrate's full body and returns only what fits the mesh. The
    metaphor is the same as `filter` in any programming language — but
    cast as fishing, because the substrate is a sea.
    """

    def __init__(self, lo: float = 0.0, hi: float = 1.0):
        if hi < lo:
            lo, hi = hi, lo
        self.lo = lo
        self.hi = hi

    def _is_in_range(self, v: Any) -> bool:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return self.lo <= float(v) <= self.hi
        return False

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = target.all_cells()
        else:
            cells = [target]
        yield {
            "kind": "trawl_header",
            "lo": self.lo,
            "hi": self.hi,
            "total_cells": len(cells),
        }
        if not cells:
            yield {"kind": "trawl_empty", "text": "The sea is empty. The trawl returns nothing."}
            return
        caught = 0
        for cell in cells:
            v = cell.value
            in_net = self._is_in_range(v)
            if in_net:
                caught += 1
                yield {
                    "kind": "trawl_catch",
                    "address": cell.address,
                    "value": v,
                    "lo": self.lo,
                    "hi": self.hi,
                    "glyph": "<><",
                    "line": f"<><  {cell.address:>6}  v={v}  (kept)",
                }
            else:
                yield {
                    "kind": "trawl_bycatch",
                    "address": cell.address,
                    "value": v,
                    "lo": self.lo,
                    "hi": self.hi,
                    "glyph": "~~",
                    "line": f"~~   {cell.address:>6}  v={v}  (escaped)",
                }
        yield {
            "kind": "trawl_summary",
            "caught": caught,
            "escaped": len(cells) - caught,
            "total": len(cells),
        }

    def preview(self, target) -> str:
        return f"A fishing trawl with mesh [{self.lo}, {self.hi}]. Cells with values inside the range are caught; others escape."


class ShoalOpener(Opener):
    """Renders the substrate as a sea floor, looking for shoals.

    A shoal is a place where the depth changes suddenly — a sandbar
    rising from the sea floor. This opener sorts cells by address
    (treated as a position along a transect) and yields a "depth event"
    for each cell. A shoal is flagged wherever the depth change between
    consecutive cells exceeds a threshold (a sudden rise or fall).

    Fable 19 (Oracle): the oracle predicts depths. The shoal is a
    *discontinuity* — a place where the oracle's prediction breaks.
    Like a sandbar visible only at low tide, a shoal is hidden until
    you traverse the chart and notice the gradient.
    """

    def __init__(self, threshold: float = 0.5):
        # Minimum |delta| in value to count as a shoal.
        self.threshold = threshold

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = sorted(target.all_cells(), key=lambda c: c.address)
        else:
            cells = [target]
        yield {
            "kind": "shoal_header",
            "threshold": self.threshold,
            "n_cells": len(cells),
        }
        if not cells:
            yield {"kind": "shoal_empty", "text": "Empty sea. No floor to chart."}
            return

        # Build a (address, numeric_value_or_none) sequence
        seq = []
        for c in cells:
            v = c.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                seq.append((c.address, float(v)))
            else:
                seq.append((c.address, None))

        # Yield a depth event for every cell so the chart is complete
        prev_val = None
        for addr, v in seq:
            if v is None:
                yield {
                    "kind": "shoal_depth",
                    "address": addr,
                    "depth": None,
                    "note": "non-numeric — no depth reading",
                }
                continue
            delta = None if prev_val is None else (v - prev_val)
            yield {
                "kind": "shoal_depth",
                "address": addr,
                "depth": v,
                "delta_from_prev": delta,
            }
            if delta is not None and abs(delta) >= self.threshold:
                # Shoal! Mark a discontinuity
                kind = "shoal_rise" if delta > 0 else "shoal_drop"
                glyph = "/\\" if delta > 0 else "\\/"
                yield {
                    "kind": kind,
                    "address": addr,
                    "delta": round(delta, 4),
                    "threshold": self.threshold,
                    "glyph": glyph,
                    "warning": f"SHOAL at {addr}: depth change {delta:+.2f} "
                               f"(≥ {self.threshold})",
                }
            prev_val = v

    def preview(self, target) -> str:
        return f"A sea-floor transect. Each cell is a depth reading. Sudden changes (|Δ| ≥ {self.threshold}) are flagged as shoals."


class MooringOpener(Opener):
    """Renders the substrate as a harbor, mooring by mooring.

    A mooring is a place where many boats are tied up — a hub of
    activity. This opener finds the cells that have the most
    connections (in-edges plus out-edges), ranks them by degree, and
    yields each as a "mooring" event. The busier the cell, the busier
    the harbor.

    Fable 11 (Paper and the Tablet): the mooring is a record of who
    came and went. A cell with many neighbors is a place of importance
    — the substrate's "downtown". A cell with no neighbors is a
    desert island; one with many is a port city.
    """

    def __init__(self, top: int = 5):
        # How many of the busiest moorings to yield as full events.
        self.top = top

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if not isinstance(target, Substrate):
            yield {
                "kind": "mooring",
                "address": target.address,
                "degree": len(target.neighbors()),
                "rank": 0,
                "note": "Single cell — degree is just its neighbor count.",
            }
            return
        cells = target.all_cells()
        # Compute degree for every cell
        degrees = []
        seen_edges = set()
        for cell in cells:
            # Use the cell's inputs + outputs as its neighbors
            deg = len(cell.neighbors())
            degrees.append((cell.address, cell, deg))
        # Sort descending by degree
        degrees.sort(key=lambda x: (-x[2], x[0]))
        yield {
            "kind": "mooring_header",
            "n_cells": len(cells),
            "n_edges": len(target.edges()),
            "top": self.top,
        }
        if not cells:
            yield {"kind": "mooring_empty", "text": "No moorings. The harbor is deserted."}
            return
        for i, (addr, cell, deg) in enumerate(degrees[: self.top]):
            busyness = (
                "ghost port" if deg == 0 else
                "quiet cove" if deg <= 1 else
                "fishing village" if deg <= 3 else
                "busy port" if deg <= 6 else
                "metropolis"
            )
            yield {
                "kind": "mooring",
                "address": addr,
                "degree": deg,
                "rank": i,
                "busyness": busyness,
                "neighbors": [n.address for n in cell.neighbors()],
                "value": cell.value,
                "confidence": round(cell.confidence, 4),
                "line": f"#{i+1}  {addr:>6}  degree={deg:>2}  ({busyness})",
            }

    def preview(self, target) -> str:
        return f"A harbor ranked by mooring. The {self.top} busiest cells (most connections) are listed, with a 'busyness' label."


class GaleOpener(Opener):
    """Renders the substrate as a weather forecast for a coming gale.

    A gale is a storm — and a stale inference is a storm about to
    break. This opener inspects each cell's *inference confidence*
    (not the canonical confidence) and flags those whose inference is
    decaying rapidly, or is already below a danger threshold. Each
    flagged cell becomes a "gale warning" event.

    Fable 19 (Oracle) + Fable 22 (Sundial): the oracle's confidence
    meets the sundial — prophecies that were once bright are now
    dimming. The gale opener is the early-warning system for prophecies
    that have grown too stale to trust. Used by agents that depend on
    inferences (paper 107, paper 117 Open Q5) to decide when to refresh.
    """

    def __init__(self, danger: float = 0.3):
        # Inference confidence below this is a gale.
        self.danger = danger

    def activate(self, target) -> Iterator[Dict[str, Any]]:
        from .substrate import Substrate
        if isinstance(target, Substrate):
            cells = target.all_cells()
        else:
            cells = [target]
        yield {
            "kind": "gale_header",
            "danger_threshold": self.danger,
            "n_cells": len(cells),
        }
        if not cells:
            yield {"kind": "gale_empty", "text": "Calm seas. The forecast is empty."}
            return
        gale_count = 0
        for cell in cells:
            inf = cell.inference
            # No inference set? Skip — nothing to warn about.
            if inf is None:
                yield {
                    "kind": "gale_clear",
                    "address": cell.address,
                    "inference": None,
                    "note": "no inference set — no gale possible",
                }
                continue
            inf_conf = cell.inference_confidence
            canonical_conf = cell.confidence
            is_gale = inf_conf < self.danger
            if is_gale:
                gale_count += 1
                # Storm level based on how far below danger
                if inf_conf < self.danger * 0.3:
                    storm = "hurricane"
                    glyph = "X"
                elif inf_conf < self.danger * 0.6:
                    storm = "gale"
                    glyph = "!"
                else:
                    storm = "squall"
                    glyph = "?"
                yield {
                    "kind": "gale_warning",
                    "address": cell.address,
                    "inference": inf,
                    "inference_confidence": round(inf_conf, 4),
                    "canonical_confidence": round(canonical_conf, 4),
                    "storm": storm,
                    "glyph": glyph,
                    "warning": f"GALE at {cell.address}: inference confidence {inf_conf:.2f} "
                               f"below danger ({self.danger})",
                }
            else:
                yield {
                    "kind": "gale_clear",
                    "address": cell.address,
                    "inference": inf,
                    "inference_confidence": round(inf_conf, 4),
                    "canonical_confidence": round(canonical_conf, 4),
                    "note": "inference is fresh — no gale",
                }
        yield {
            "kind": "gale_summary",
            "gale_count": gale_count,
            "total": len(cells),
            "danger_threshold": self.danger,
        }

    def preview(self, target) -> str:
        return f"A weather forecast. Cells whose inference confidence has decayed below {self.danger} are flagged as gales."


# Register the 5 MORE new openers
register("buoy", BuoyOpener())
register("trawl", TrawlOpener())
register("shoal", ShoalOpener())
register("mooring", MooringOpener())
register("gale", GaleOpener())
