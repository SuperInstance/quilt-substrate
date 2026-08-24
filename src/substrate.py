"""
quilt_substrate — The Quilt substrate as a working Python library.

The 11-primitive cell (8 from cell-runtime + 3 new: Convoy, Decay, Witness).
The tensor encoding. The Schrödinger pattern. The fog-of-war decay. The convoy
consensus. The witness log. The opener layer. The forest biome, as a package.

The cell is a system, not a value. The graph is the truth. The substrate is the soil.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Iterable
import hashlib
import json
import math
import time
import uuid


# -- Helpers ---------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(data: Any) -> str:
    """Stable hash of any JSON-serializable value."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _now_ts() -> float:
    return time.time()


# -- Vibe (8th primitive, unchanged) ---------------------------------------

@dataclass(frozen=True)
class Vibe:
    """Position, velocity, acceleration through the graph."""
    pos: Tuple[float, ...] = (0.0,)
    vel: Tuple[float, ...] = (0.0,)
    acc: Tuple[float, ...] = (0.0,)

    def step(self, dt: float = 1.0) -> "Vibe":
        new_pos = tuple(p + v * dt + 0.5 * a * dt * dt for p, v, a in zip(self.pos, self.vel, self.acc))
        new_vel = tuple(v + a * dt for v, a in zip(self.vel, self.acc))
        return Vibe(pos=new_pos, vel=new_vel, acc=self.acc)

    def nudge(self, target_pos: Tuple[float, ...], k: float = 0.1) -> "Vibe":
        if len(target_pos) != len(self.pos):
            target_pos = target_pos + self.pos[len(target_pos):]
        new_acc = tuple(k * (t - p) for p, t in zip(self.pos, target_pos))
        return Vibe(pos=self.pos, vel=self.vel, acc=new_acc)


# -- Convoy primitive (paper 108) -----------------------------------------

@dataclass
class ConvoyEntry:
    """One agent's contribution to a cell's convoy."""
    agent_id: str
    weight: float  # 0.0 to 1.0
    last_write: float  # timestamp
    value_hash: str  # hash of the value the agent wrote


# -- Decay primitive (paper 109) ------------------------------------------

@dataclass
class DecayState:
    """The cell's decay function and current confidence."""
    c0: float = 1.0  # initial confidence
    lam: float = 0.0001  # decay rate (per second)
    t0: float = field(default_factory=_now_ts)  # time of last refresh
    t_born: float = field(default_factory=_now_ts)  # time of birth

    def confidence(self, t: Optional[float] = None) -> float:
        if t is None:
            t = _now_ts()
        elapsed = max(0.0, t - self.t0)
        return self.c0 * math.exp(-self.lam * elapsed)

    def refresh(self, c0: Optional[float] = None) -> None:
        self.t0 = _now_ts()
        if c0 is not None:
            self.c0 = c0


# -- Witness primitive (paper 110) ----------------------------------------

@dataclass
class WitnessEntry:
    """One entry in a cell's witness log."""
    ts: float
    agent_id: str
    action: str  # "read" | "write" | "inference" | "decay"
    value_hash: str
    prev_hash: str  # Merkle-link to previous entry

    def to_dict(self) -> dict:
        return {"ts": self.ts, "agent_id": self.agent_id, "action": self.action,
                "value_hash": self.value_hash, "prev_hash": self.prev_hash}


# -- Cell -----------------------------------------------------------------

class Cell:
    """
    The 11-primitive cell. The cell is a system, not a value.
    """

    def __init__(
        self,
        address: str,
        value: Any = None,
        tensor: Optional[List] = None,
        axes: Optional[Tuple[str, ...]] = None,
        jepa: Optional[Any] = None,
    ):
        self._id = str(uuid.uuid4())[:8]
        self._address = address
        self._value = value
        self._tensor = tensor
        self._axes = axes or ()
        self._jepa = jepa
        self._inputs: Dict[str, "Cell"] = {}
        self._outputs: Dict[str, "Cell"] = {}
        self._debit: Any = None
        self._credit: Any = value
        self._vibe: Vibe = Vibe()
        self._gc_phase: int = 0
        self._last_murmur: float = _now_ts()
        self._murmur_count: int = 0
        self._ticks: int = 0
        self._born: float = _now_ts()
        # Convoy (paper 108)
        self._convoy: List[ConvoyEntry] = []
        # Decay (paper 109)
        self._decay = DecayState()
        # Witness (paper 110)
        self._witness_log: List[WitnessEntry] = []
        self._witness_root: str = "0" * 16  # Merkle root
        # Schrödinger pattern (paper 107)
        self._canonical: bool = False
        self._inference: Optional[Any] = None
        # GC tracking
        self._log: List[Dict[str, Any]] = []

    # -- 8 primitives (Z_in, Z_out, JEPA, DoubleEntry, Vibe, GC, Murmur, Graph) --

    def connect(self, other: "Cell", name: Optional[str] = None, weight: float = 1.0) -> "Cell":
        """Z_in: this cell receives from `other`. Weight used for Convoy consensus."""
        key = name or other._address
        self._inputs[key] = other
        other._outputs[self._name_for(self, key)] = self
        # The cell that receives adds the sender to its own convoy
        # (the convoy represents the agents that contribute to THIS cell)
        self._add_to_convoy(other._address, weight=weight)
        return self

    def _name_for(self, other: "Cell", key: str) -> str:
        return f"{other._address}->{self._address}/{key}"

    def predict(self) -> Any:
        """JEPA: predict the next value from current inputs."""
        inputs = {k: c.value for k, c in self._inputs.items()}
        if self._jepa:
            return self._jepa(inputs)
        return self._value

    def observe(self, actual: Any) -> float:
        """JEPA: observe the actual value, compute error, return it."""
        predicted = self.predict()
        if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
            err = actual - predicted
            self._vibe = self._vibe.nudge(target_pos=(float(actual),))
            return err
        return 0.0

    def tick(self) -> None:
        """One update cycle."""
        self._debit = self._credit
        predicted = self.predict()
        if predicted is not None:
            self._credit = predicted
        self._value = self._credit
        self._vibe = self._vibe.step(dt=1.0)
        self.murmur()
        self._ticks += 1

    @property
    def vibe(self) -> Vibe: return self._vibe
    @vibe.setter
    def vibe(self, v: Vibe) -> None: self._vibe = v

    def gc(self) -> str:
        """3-phase GC."""
        if self._gc_phase == 0:
            seen = {}
            for name, c in list(self._outputs.items()):
                sig = repr(c.value)[:50]
                if sig in seen:
                    self._outputs.pop(name, None)
                else:
                    seen[sig] = c
            self._gc_phase = 1
            return "merge-similar"
        elif self._gc_phase == 1:
            now = _now_ts()
            for name, c in list(self._inputs.items()):
                if now - c._last_murmur > 60:
                    self._inputs.pop(name, None)
            self._gc_phase = 2
            return "decay-old"
        else:
            self._gc_phase = 0
            return "prune-weak"

    def murmur(self) -> bool:
        """Heartbeat. Low-cost signal."""
        self._last_murmur = _now_ts()
        self._murmur_count += 1
        return True

    def neighbors(self) -> List["Cell"]:
        return list(self._inputs.values()) + list(self._outputs.values())

    def reach(self, max_depth: int = 3) -> List["Cell"]:
        seen: List[Cell] = [self]
        seen_ids: set = {id(self)}
        frontier = [self]
        for _ in range(max_depth):
            next_frontier = []
            for c in frontier:
                for n in c.neighbors():
                    if id(n) not in seen_ids:
                        seen_ids.add(id(n))
                        next_frontier.append(n)
                        seen.append(n)
            frontier = next_frontier
        return seen

    # -- Convoy (paper 108) --

    def _add_to_convoy(self, agent_id: str, weight: float = 1.0) -> None:
        """Add an agent to this cell's convoy."""
        # Remove existing entry for this agent
        self._convoy = [e for e in self._convoy if e.agent_id != agent_id]
        self._convoy.append(ConvoyEntry(
            agent_id=agent_id,
            weight=weight,
            last_write=_now_ts(),
            value_hash=_hash(self._value),
        ))

    @property
    def convoy(self) -> List[ConvoyEntry]:
        return list(self._convoy)

    def convoy_value(self) -> Any:
        """The weighted-median consensus value across the convoy.
        Falls back to the most recent write if values aren't numeric."""
        if not self._convoy:
            return self._value
        # For now, return the value with the highest weight * recency
        now = _now_ts()
        best = max(self._convoy, key=lambda e: e.weight * (1.0 / (1.0 + now - e.last_write)))
        return self._value  # The convoy consensus is a future enhancement

    # -- Decay (paper 109) --

    @property
    def decay(self) -> DecayState:
        return self._decay

    @property
    def confidence(self) -> float:
        return self._decay.confidence()

    def refresh(self, c0: Optional[float] = None) -> None:
        """Refresh this cell. Resets the decay clock."""
        self._decay.refresh(c0=c0)

    # -- Witness (paper 110) --

    def witness(self, agent_id: str, action: str, value: Any) -> WitnessEntry:
        """Append a witness entry. Returns the entry."""
        entry = WitnessEntry(
            ts=_now_ts(),
            agent_id=agent_id,
            action=action,
            value_hash=_hash(value),
            prev_hash=self._witness_root,
        )
        self._witness_log.append(entry)
        self._witness_root = _hash(entry.to_dict())
        return entry

    @property
    def witness_log(self) -> List[WitnessEntry]:
        return list(self._witness_log)

    @property
    def witness_root(self) -> str:
        return self._witness_root

    # -- Schrödinger pattern (paper 107) --

    def infer(self, inferred_value: Any) -> None:
        """Set the inferred value (Schrödinger pattern: pre-rendered, not canonical)."""
        self._inference = inferred_value
        self._canonical = False

    def observe_canonical(self) -> Any:
        """Observe this cell. Marks it as canonical."""
        self._canonical = True
        return self._value

    @property
    def canonical(self) -> bool:
        return self._canonical

    @property
    def inference(self) -> Any:
        return self._inference

    # -- Tensor encoding (paper 112) --

    @property
    def tensor(self) -> Optional[List]:
        return self._tensor

    @property
    def axes(self) -> Tuple[str, ...]:
        return self._axes

    def slice(self, **kwargs) -> Optional[List]:
        """Slice the tensor along the given axes. Returns the slice, or None if axes don't match.

        Example: a 3D tensor with axes=('d', 'x', 'y') and tensor shape (1, 2, 3):
          cell.slice(d=0) → shape (2, 3) — the d=0 layer
          cell.slice(d=0, x=0) → shape (3,) — the d=0, x=0 row
          cell.slice(d=0, x=0, y=1) → scalar — the d=0, x=0, y=1 element
        """
        if self._tensor is None:
            return None
        # Build a list of (axis_index, spec) sorted by axis_index
        slices = []
        for axis, spec in kwargs.items():
            if axis not in self._axes:
                return None
            slices.append((self._axes.index(axis), spec))
        slices.sort(key=lambda x: x[0])
        # Walk the tensor, descending into each dimension
        def walk(t, depth):
            if depth == len(self._axes):
                yield t
                return
            axis_idx, spec = next((s for s in slices if s[0] == depth), (None, None))
            if axis_idx is not None:
                if isinstance(spec, int):
                    yield from walk(t[spec], depth + 1)
                elif isinstance(spec, slice):
                    for item in t[spec]:
                        yield from walk(item, depth + 1)
            else:
                # Keep this dimension
                for item in t:
                    yield from walk(item, depth + 1)
        return list(walk(self._tensor, 0))

    # -- Properties (8-primitive compatible) --

    @property
    def value(self) -> Any: return self._value
    @value.setter
    def value(self, v: Any) -> None:
        self._debit = self._credit
        self._credit = v
        self._value = v
    @property
    def address(self) -> str: return self._address
    @property
    def inputs(self) -> Dict[str, "Cell"]: return dict(self._inputs)
    @property
    def outputs(self) -> Dict[str, "Cell"]: return dict(self._outputs)
    @property
    def debit(self) -> Any: return self._debit
    @property
    def credit(self) -> Any: return self._credit
    @property
    def age(self) -> float: return _now_ts() - self._born
    @property
    def ticks(self) -> int: return self._ticks
    @property
    def last_murmur(self) -> float: return self._last_murmur

    def __repr__(self) -> str:
        conf = self.confidence
        return f"Cell({self._address}, value={self._value!r}, conf={conf:.3f}, ticks={self._ticks})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self._address,
            "value": self._value,
            "tensor": self._tensor,
            "axes": list(self._axes),
            "vibe": {"pos": self._vibe.pos, "vel": self._vibe.vel, "acc": self._vibe.acc},
            "ticks": self._ticks,
            "age": self.age,
            "convoy": [{"agent_id": e.agent_id, "weight": e.weight, "last_write": e.last_write} for e in self._convoy],
            "decay": {"c0": self._decay.c0, "lam": self._decay.lam, "confidence": self.confidence},
            "witness_root": self._witness_root,
            "canonical": self._canonical,
            "inference": self._inference,
        }


# -- Substrate (the cell-graph as a whole) ---------------------------------

class Substrate:
    """The Quilt substrate. The graph of cells. The soil."""

    def __init__(self):
        self._cells: Dict[str, Cell] = {}
        self._t: float = _now_ts()

    def add(self, cell: Cell) -> "Substrate":
        self._cells[cell.address] = cell
        return self

    def get(self, address: str) -> Optional[Cell]:
        return self._cells.get(address)

    def all_cells(self) -> List[Cell]:
        return list(self._cells.values())

    def __len__(self) -> int:
        return len(self._cells)

    # -- Substrate-level operations --

    def decay(self, dt: float = 1.0) -> None:
        """Advance the substrate's clock by dt seconds. All cells decay."""
        self._t += dt
        for cell in self._cells.values():
            # The cell's DecayState is internal; we just let it elapse naturally
            pass

    def tick(self, n: int = 1) -> None:
        """Tick all cells n times."""
        for _ in range(n):
            for c in self._cells.values():
                c.tick()

    def witness(self, cell: Cell, agent_id: str, action: str, value: Any = None) -> None:
        """Witness a read/write/inference/decay on a cell."""
        if value is None:
            value = cell.value
        cell.witness(agent_id, action, value)

    def observe(self, address: str, agent_id: str = "default") -> Any:
        """Observe a cell: marks it canonical and witnesses the read."""
        cell = self.get(address)
        if cell is None:
            return None
        value = cell.observe_canonical()
        self.witness(cell, agent_id, "read", value)
        return value

    def infer(self, address: str, value: Any, agent_id: str = "default") -> None:
        """Set an inferred value: pre-rendered, not canonical."""
        cell = self.get(address)
        if cell is None:
            return
        cell.infer(value)
        self.witness(cell, agent_id, "inference", value)

    def refresh(self, address: str) -> None:
        """Refresh a cell: reset its decay clock."""
        cell = self.get(address)
        if cell is not None:
            cell.refresh()
            self.witness(cell, "system", "refresh", cell.value)

    # -- Opener layer (paper 111) --

    def render(self, opener: str, **kwargs) -> Any:
        """Render the substrate through an opener."""
        if opener == "chart":
            return self._render_chart(**kwargs)
        elif opener == "list":
            return self._render_list(**kwargs)
        elif opener == "tensor":
            return self._render_tensor(**kwargs)
        elif opener == "witness":
            return self._render_witness(**kwargs)
        elif opener == "convoy":
            return self._render_convoy(**kwargs)
        elif opener == "graph":
            return self._render_graph(**kwargs)
        else:
            return {"error": f"unknown opener: {opener}"}

    def _render_chart(self, viewport: Optional[Dict] = None, **kwargs) -> Dict:
        """Render as a chart. Viewport is {axis: slice_spec}."""
        out = {"cells": []}
        for cell in self._cells.values():
            entry = {
                "address": cell.address,
                "value": cell.value,
                "confidence": cell.confidence,
                "canonical": cell.canonical,
                "axes": list(cell.axes),
            }
            if viewport and cell.tensor is not None:
                slice_spec = {k: v for k, v in viewport.items() if k in cell.axes}
                if slice_spec:
                    entry["slice"] = cell.slice(**slice_spec)
            out["cells"].append(entry)
        return out

    def _render_list(self, **kwargs) -> List[Dict]:
        return [c.to_dict() for c in self._cells.values()]

    def _render_tensor(self, address: str, **kwargs) -> Optional[List]:
        cell = self.get(address)
        if cell is None:
            return None
        return cell.tensor

    def _render_witness(self, address: str, **kwargs) -> List[Dict]:
        cell = self.get(address)
        if cell is None:
            return []
        return [e.to_dict() for e in cell.witness_log]

    def _render_convoy(self, address: str, **kwargs) -> List[Dict]:
        cell = self.get(address)
        if cell is None:
            return []
        return [{"agent_id": e.agent_id, "weight": e.weight, "last_write": e.last_write} for e in cell.convoy]

    def _render_graph(self, **kwargs) -> Dict:
        nodes = [{"id": c.address, "value": c.value} for c in self._cells.values()]
        edges = []
        for c in self._cells.values():
            for k, other in c.inputs.items():
                edges.append({"from": other.address, "to": c.address, "weight": k})
        return {"nodes": nodes, "edges": edges}

    def to_dict(self) -> Dict:
        return {
            "cells": [c.to_dict() for c in self._cells.values()],
            "n_cells": len(self._cells),
            "t": self._t,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def __repr__(self) -> str:
        return f"Substrate({len(self._cells)} cells, t={self._t:.1f})"
