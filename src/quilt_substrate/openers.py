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
