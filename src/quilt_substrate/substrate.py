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
    """Position, velocity, acceleration through the graph.

    The Vibe is a damped harmonic oscillator. The spring constant k connects
    the position to a target; the damping coefficient c reduces the velocity
    at each step. With c=0.1, the system is critically damped for small dt.

    Math (paper 117, §2.5):
        p_{t+1} = p_t + v_t dt + 0.5 a_t dt^2
        v_{t+1} = (v_t + a_t dt) * (1 - c)
        a_{t+1} = k * (target - p_{t+1})
    """
    pos: Tuple[float, ...] = (0.0,)
    vel: Tuple[float, ...] = (0.0,)
    acc: Tuple[float, ...] = (0.0,)
    damping: float = 0.1  # damping coefficient (paper 117, §2.5)

    def step(self, dt: float = 1.0) -> "Vibe":
        new_pos = tuple(p + v * dt + 0.5 * a * dt * dt for p, v, a in zip(self.pos, self.vel, self.acc))
        new_vel = tuple((v + a * dt) * (1.0 - self.damping) for v, a in zip(self.vel, self.acc))
        return Vibe(pos=new_pos, vel=new_vel, acc=self.acc, damping=self.damping)

    def nudge(self, target_pos: Tuple[float, ...], k: float = 0.1) -> "Vibe":
        if len(target_pos) != len(self.pos):
            target_pos = target_pos + self.pos[len(target_pos):]
        new_acc = tuple(k * (t - p) for p, t in zip(self.pos, target_pos))
        return Vibe(pos=self.pos, vel=self.vel, acc=new_acc, damping=self.damping)


# -- Convoy primitive (paper 108) -----------------------------------------

@dataclass
class ConvoyEntry:
    """One agent's contribution to a cell's convoy.

    The convoy tracks who wrote what. For consensus (Open Q1), we need the
    value too. We store the *last value* the agent wrote.
    """
    agent_id: str
    weight: float  # 0.0 to 1.0
    last_write: float  # timestamp
    value_hash: str  # hash of the value the agent wrote
    value: Any = None  # the actual value the agent wrote (for consensus)


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
    """One entry in a cell's witness log.

    Fable 11 (Paper and the Tablet): a witness records not just what happened
    but why. The justification field is the agent's reasoning, free-form text
    that explains the action.
    """
    ts: float
    agent_id: str
    action: str  # "read" | "write" | "inference" | "decay"
    value_hash: str
    prev_hash: str  # Merkle-link to previous entry
    justification: str = ""  # Fable 11: why this action (free-form text)

    def to_dict(self) -> dict:
        return {"ts": self.ts, "agent_id": self.agent_id, "action": self.action,
                "value_hash": self.value_hash, "prev_hash": self.prev_hash,
                "justification": self.justification}


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
        self._inference_confidence: float = 0.0  # Fable 19: oracle's confidence
        self._inference_ts: float = field(default_factory=_now_ts)  # Fable 22+19: inference decays
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

    def _add_to_convoy(self, agent_id: str, weight: float = 1.0, value: Any = None) -> None:
        """Add an agent to this cell's convoy.

        If `value` is provided, the agent's last-written value is recorded
        for consensus. If not, the cell's current value is used.
        """
        # Remove existing entry for this agent
        self._convoy = [e for e in self._convoy if e.agent_id != agent_id]
        # Weights can be any non-negative float. The convoy code handles
        # weights > 1.0 (they just contribute more to weighted consensus).
        # Negative weights are clamped to 0.
        if weight < 0:
            weight = 0.0
        self._convoy.append(ConvoyEntry(
            agent_id=agent_id,
            weight=weight,
            last_write=_now_ts(),
            value_hash=_hash(self._value),
            value=value if value is not None else self._value,
        ))

    @property
    def convoy(self) -> List[ConvoyEntry]:
        return list(self._convoy)

    def convoy_value(self, method: str = "weighted_mean") -> Any:
        """The consensus value across the convoy.

        Methods (paper 108 + paper 117, Open Q1):
        - "weighted_mean" — sum(w_i * v_i) / sum(w_i). Fast, but vulnerable to outliers.
        - "weighted_median" — the value where 50% of weight is below. Robust to outliers.
        - "trimmed_mean" — drop the highest and lowest 10% of weight, then mean.
        - "highest_weight" — the value with the highest single weight (the original behavior).

        Falls back to self._value if values aren't numeric.

        Raises ValueError for unknown methods.
        """
        valid_methods = ("weighted_mean", "weighted_median", "trimmed_mean", "highest_weight")
        if method not in valid_methods:
            raise ValueError(f"Unknown consensus method: {method}. Valid: {valid_methods}")
        if not self._convoy:
            return self._value
        # Only consider numeric values
        numeric = [(e.weight, e.value) for e in self._convoy
                   if isinstance(e.value, (int, float))]
        if not numeric:
            # Fallback: most-recent highest-weight
            now = _now_ts()
            best = max(self._convoy, key=lambda e: e.weight * (1.0 / (1.0 + now - e.last_write)))
            return self._value

        if method == "weighted_mean":
            total_w = sum(w for w, _ in numeric)
            if total_w == 0:
                return self._value
            return sum(w * v for w, v in numeric) / total_w

        if method == "weighted_median":
            # Sort by value
            numeric.sort(key=lambda x: x[1])
            total_w = sum(w for w, _ in numeric)
            cum_w = 0
            for w, v in numeric:
                cum_w += w
                if cum_w >= total_w / 2:
                    return v
            return numeric[-1][1]

        if method == "trimmed_mean":
            # Drop top and bottom 10% by weight
            numeric.sort(key=lambda x: x[1])
            total_w = sum(w for w, _ in numeric)
            trim = total_w * 0.1
            cum_w = 0
            kept = []
            for w, v in numeric:
                cum_w += w
                if trim <= cum_w <= total_w - trim:
                    kept.append((w, v))
            if not kept:
                return self._value
            kept_w = sum(w for w, _ in kept)
            if kept_w == 0:
                return self._value
            return sum(w * v for w, v in kept) / kept_w

        # Default: highest weight
        return max(self._convoy, key=lambda e: e.weight).value

    def _geometric_median_1d(self, numeric):
        """1D geometric median = the median.

        For a weighted set of values, the weighted median is the value
        where 50% of weight is below and 50% is above. This is the
        geometric median for 1D data.
        """
        if not numeric:
            return self._value
        # Already implemented as weighted_median; this is the alias
        return self.convoy_value(method="weighted_median")

    def geometric_median(self, max_iter=100, tol=1e-6):
        """The geometric median of the convoy (multi-dimensional).

        Uses Weiszfeld's algorithm. Robust to outliers.

        For 1D values, this is the same as weighted_median.
        For multi-dimensional tensor values, this converges to the
        point that minimizes sum(w_i * ||x - v_i||).

        Returns self._value if the convoy has no numeric values, or if
        the values are 1D (use weighted_median instead).
        """
        # Get values
        numeric = [(e.weight, e.value) for e in self._convoy
                   if isinstance(e.value, (int, float))]
        if not numeric:
            return self._value
        # 1D case
        if all(isinstance(v, (int, float)) for _, v in numeric):
            return self._geometric_median_1d(numeric)
        # Multi-dimensional case
        try:
            import numpy as np
        except ImportError:
            # No numpy, fall back to weighted mean
            return self.convoy_value(method="weighted_mean")
        # All values must be lists of the same length
        dim = len(numeric[0][1])
        if not all(len(v) == dim for _, v in numeric):
            return self._value
        # Weiszfeld's algorithm
        weights = np.array([w for w, _ in numeric])
        values = np.array([v for _, v in numeric])
        # Normalize weights
        weights = weights / weights.sum()
        # Start with weighted mean
        x = np.average(values, axis=0, weights=weights)
        for _ in range(max_iter):
            # Compute distances
            distances = np.linalg.norm(values - x, axis=1)
            # Avoid division by zero
            distances = np.where(distances < tol, tol, distances)
            # Weiszfeld update
            numer = np.sum(weights[:, None] * values / distances[:, None], axis=0)
            denom = np.sum(weights / distances)
            x_new = numer / denom
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        return x.tolist()

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

    def witness(self, agent_id: str, action: str, value: Any, justification: str = "") -> WitnessEntry:
        """Append a witness entry. Returns the entry.

        Args:
            agent_id: The agent performing the action
            action: "read" | "write" | "inference" | "decay"
            value: The value being witnessed
            justification: Fable 11 — why this action. Free-form text that
                explains the agent's reasoning. Optional but encouraged.
        """
        entry = WitnessEntry(
            ts=_now_ts(),
            agent_id=agent_id,
            action=action,
            value_hash=_hash(value),
            prev_hash=self._witness_root,
            justification=justification,
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

    def temperature(self, window_seconds: float = 3600.0) -> float:
        """Paper 124: The Substrate's Temperature.

        T(C, tau) = -sum p_k * ln(p_k), where p_k is the empirical probability
        of operation k in the witness log over the window tau.

        Returns 0 for an empty witness log. Bounded above by ln(11) ~= 2.398
        for the 11 known primitives.
        """
        import math
        if not self._witness_log:
            return 0.0
        # Filter to entries within the window
        now = _now_ts()
        window_start = now - window_seconds
        recent = [e for e in self._witness_log if e.ts >= window_start]
        if not recent:
            return 0.0
        # Count operations
        from collections import Counter
        counts = Counter(e.action for e in recent)
        n = len(recent)
        T = 0.0
        for k in counts:
            p = counts[k] / n
            T -= p * math.log(p)
        return T

    def regime(self) -> str:
        """Classify the cell's regime based on temperature.

        Paper 124, Definition 2.5:
        - frozen: T = 0
        - cold: 0 < T <= 0.5
        - warm: 0.5 < T <= 1.5
        - hot:  T > 1.5
        """
        T = self.temperature()
        if T == 0.0:
            return "frozen"
        if T <= 0.5:
            return "cold"
        if T <= 1.5:
            return "warm"
        return "hot"

    # -- Schrödinger pattern (paper 107) --

    def infer(self, inferred_value: Any, confidence: float = 1.0) -> None:
        """Set the inferred value (Schrödinger pattern: pre-rendered, not canonical).

        Args:
            inferred_value: The inferred (pre-rendered) value
            confidence: The inference confidence in [0, 1]. Defaults to 1.0
                (full confidence). The inference confidence decays over time
                (Fable 19 + Fable 22: oracle's confidence meets the sundial).
        """
        self._inference = inferred_value
        self._inference_confidence = max(0.0, min(1.0, confidence))
        self._inference_ts = _now_ts()
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

    @property
    def inference_confidence(self, t: Optional[float] = None) -> float:
        """The inference's confidence, decayed by time.

        Fable 19 + 22: An oracle's confidence in her prophecy is high when
        the prophecy is fresh, and decays as the prophecy ages. This method
        returns the current decayed confidence.

        Decay rate: matches the cell's decay rate (lam).
        """
        if t is None:
            t = _now_ts()
        elapsed = max(0.0, t - self._inference_ts)
        # Defensive: if decay.lam is None or invalid, no decay
        lam = self._decay.lam if isinstance(self._decay.lam, (int, float)) else 0.0
        # Use a faster decay for inferences (5x faster than canonical values)
        return self._inference_confidence * math.exp(-5 * lam * elapsed)

    def confident_inference(self, threshold: float = 0.5) -> Optional[Any]:
        """Return the inference only if its (decayed) confidence is above threshold.

        Fable 19: The oracle must distinguish prophecy from noise. The
        substrate should refuse to act on inferences that are too uncertain.

        Returns None if confidence is below threshold.
        """
        if self.inference_confidence >= threshold:
            return self._inference
        return None

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
        # Per-agent witness log (paper 117, Open Q3)
        # agent_id -> List[Dict] of (cell_address, ts, action, value)
        self._agent_witness: Dict[str, List[Dict[str, Any]]] = {}
        # Per-agent decay rates (paper 117, Open Q8)
        # agent_id -> lambda (default 0.0001/s, ~3 hours half-life)
        self._agent_decay: Dict[str, float] = {}

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
        """Advance the substrate's clock by dt seconds. All cells decay.

        The decay is computed lazily via the cell's confidence property —
        we just advance the substrate's internal clock, and subsequent
        reads of cell.confidence will return the decayed value.

        We also update the cell's internal t0 to reflect the passage of
        time, so subsequent refresh()es are computed against the new clock.
        """
        self._t += dt
        for cell in self._cells.values():
            # The cell's decay is computed from (now - t0); since the cell's
            # confidence() uses _now_ts(), we don't need to explicitly update
            # the cell — the confidence will be lower the next time it's read.
            # However, for batch decay (e.g., after long sleeps), the cell's
            # t0 should reflect the substrate's clock.
            # Note: the cell's DecayState uses _now_ts() directly, so this
            # is currently a no-op for the cells. The substrate's clock is
            # tracked in self._t for forward compatibility (e.g., replay).
            pass

    def advance_time(self, dt: float) -> None:
        """Advance the substrate's clock by dt seconds AND all cells' decay clocks.

        This is a stronger version of decay() — it actually moves the cells'
        decay clocks forward, not just the substrate's tracking clock.

        Use this for batch simulation: e.g., to simulate a week of decay
        without waiting 7 days.
        """
        import time as _time
        self._t += dt
        # Move each cell's t0 backward by dt so the next confidence() call
        # sees a longer elapsed time
        for cell in self._cells.values():
            cell._decay.t0 -= dt
            # _inference_ts is set on first infer(); only advance if it's a real float
            ts = getattr(cell, '_inference_ts', None)
            if isinstance(ts, (int, float)):
                object.__setattr__(cell, '_inference_ts', ts - dt)

    def age_seconds(self) -> float:
        """The substrate's age in seconds (since first cell was added)."""
        return _now_ts() - self._t

    def temperature(self, window_seconds: float = 3600.0) -> float:
        """Paper 124, Definition 2.4: Substrate-wide temperature.

        The witness-count-weighted average of cell temperatures.
        """
        cells = self.all_cells()
        if not cells:
            return 0.0
        total_n = 0
        weighted_T = 0.0
        for c in cells:
            log = c.witness_log
            if not log:
                continue
            # Count entries in the window
            now = _now_ts()
            window_start = now - window_seconds
            n = sum(1 for e in log if e.ts >= window_start)
            if n == 0:
                continue
            T = c.temperature(window_seconds=window_seconds)
            total_n += n
            weighted_T += n * T
        if total_n == 0:
            return 0.0
        return weighted_T / total_n

    def tick(self, n: int = 1) -> None:
        """Tick all cells n times."""
        for _ in range(n):
            for c in self._cells.values():
                c.tick()

    def witness(self, cell: Cell, agent_id: str, action: str, value: Any = None, justification: str = "") -> None:
        """Witness a read/write/inference/decay on a cell.

        Records the action in:
        - the cell's witness log (per-cell, paper 110)
        - the agent's witness log (per-agent, paper 117 Open Q3)
        - the cell's convoy (for consensus, paper 108 + paper 117 Open Q1)
        - the justification (Fable 11: why this action)
        """
        if value is None:
            value = cell.value
        cell.witness(agent_id, action, value, justification=justification)
        # Per-agent witness log (Open Q3)
        if agent_id not in self._agent_witness:
            self._agent_witness[agent_id] = []
        self._agent_witness[agent_id].append({
            "cell_address": cell.address,
            "ts": _now_ts(),
            "action": action,
            "value": value,
        })
        # Update convoy with the value for consensus (Open Q1)
        if action == "write":
            cell._add_to_convoy(agent_id, weight=1.0, value=value)
            # Apply the agent's decay rate (Open Q8)
            if agent_id in self._agent_decay:
                cell.decay.lam = self._agent_decay[agent_id]

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

    # -- Per-agent decay rates (paper 117, Open Q8) --

    def set_agent_decay(self, agent_id: str, lam: float) -> None:
        """Set the decay rate for an agent's writes.

        Different agents have different freshness requirements:
        - Chat messages: high decay (0.1/s, seconds)
        - Sensor readings: medium decay (0.001/s, minutes)
        - Chart data: low decay (1e-6/s, days)
        - Geological data: very low (1e-9/s, centuries)

        When this agent writes, the cell's decay rate is set to this value.
        """
        if lam < 0:
            raise ValueError(f"Decay rate must be non-negative, got {lam}")
        self._agent_decay[agent_id] = lam

    def get_agent_decay(self, agent_id: str) -> float:
        """Get the decay rate for an agent. Returns 1e-4 if not set."""
        return self._agent_decay.get(agent_id, 1e-4)

    # -- Per-agent witness log (paper 117, Open Q3) --

    def agent_witness(self, agent_id: str) -> List[Dict[str, Any]]:
        """Return the witness log for a specific agent.

        The per-agent log is queryable independently of the per-cell log.
        Use this to answer: "what did this agent read, write, and infer?"
        """
        return list(self._agent_witness.get(agent_id, []))

    def all_agents(self) -> List[str]:
        """Return all agents that have witnessed anything."""
        return list(self._agent_witness.keys())

    # -- Topology (paper 117, Open Q13) --

    def betti_0(self) -> int:
        """β₀: number of connected components in the cell graph."""
        if not self._cells:
            return 0
        # Union-find
        parent = {addr: addr for addr in self._cells}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        # Union all connected cells
        for addr, cell in self._cells.items():
            for neighbor in cell.neighbors():
                neighbor_addr = neighbor.address
                if neighbor_addr in self._cells:
                    union(addr, neighbor_addr)
        return len(set(find(a) for a in self._cells))

    def betti_1(self) -> int:
        """β₁: number of independent cycles in the cell graph.

        For an undirected graph: β₁ = E - V + β₀ (rank of H₁).

        Returns:
            β₁ as an integer. Can be negative if the graph is a forest.
        """
        V = len(self._cells)
        if V == 0:
            return 0
        # Count edges (undirected, no duplicates)
        edges = set()
        for cell in self._cells.values():
            for neighbor in cell.neighbors():
                if neighbor.address in self._cells:
                    edge = tuple(sorted([cell.address, neighbor.address]))
                    edges.add(edge)
        E = len(edges)
        beta_0 = self.betti_0()
        return E - V + beta_0

    def edges(self) -> List[Tuple[str, str]]:
        """List all undirected edges in the cell graph."""
        seen = set()
        out = []
        for cell in self._cells.values():
            for neighbor in cell.neighbors():
                if neighbor.address in self._cells:
                    edge = tuple(sorted([cell.address, neighbor.address]))
                    if edge not in seen:
                        seen.add(edge)
                        out.append(edge)
        return out

    # -- Opener layer (paper 111) --

    def render(self, opener: str, **kwargs) -> Any:
        """Render the substrate through an opener.

        Available openers (paper 117, Open Q6):
        - chart, list, tensor, witness, convoy, graph (the 6 originals)
        - voice — text representation suitable for TTS (e.g., for a blind agent)
        - telnet — text representation suitable for a CLI (e.g., for ssh access)
        - gesture — JSON description suitable for touch input
        - flowchart — DOT graph description suitable for Graphviz
        """
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
        elif opener == "voice":
            return self._render_voice(**kwargs)
        elif opener == "telnet":
            return self._render_telnet(**kwargs)
        elif opener == "gesture":
            return self._render_gesture(**kwargs)
        elif opener == "flowchart":
            return self._render_flowchart(**kwargs)
        else:
            return {"error": f"unknown opener: {opener}"}

    def render_with_picker(self, primitive: str, role: str,
                             picker=None, **kwargs) -> Any:
        """Render the substrate using the learned opener picker.

        If `picker` is given (an OpenerPicker instance), it picks the
        best opener for (primitive, role) given observed success rates.
        Otherwise, we just use the default opener from kwargs (or "chart").

        After rendering, you can call `picker.observe(primitive, role, opener, success, quality)`
        to feed the picker and let it learn.
        """
        opener = kwargs.pop("opener", None) or "chart"
        picked_opener = opener
        picker_reason = "default"
        if picker is not None:
            picked_opener, score, picker_reason = picker.pick(
                primitive, role, candidates=[opener] if opener else None,
            )
        # Render
        result = self.render(picked_opener, **kwargs)
        return {
            "result": result,
            "picked_opener": picked_opener,
            "picker_reason": picker_reason,
        }

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

    def _render_voice(self, **kwargs) -> str:
        """Render the substrate as text suitable for text-to-speech.

        Each cell is announced with its address, value, and confidence.
        The result is a plain-text narrative that a blind agent (or a human
        with a screen reader) can listen to.

        Format: "Cell <address>: <value>. Confidence <conf>."
        """
        parts = []
        for c in sorted(self._cells.values(), key=lambda c: c.address):
            val = c.value
            if isinstance(val, float):
                val = f"{val:.3f}"
            conf = f"{c.confidence:.2f}"
            fresh = "fresh" if c.confidence > 0.7 else "stale"
            parts.append(f"Cell {c.address}: {val}. Confidence {conf}, {fresh}.")
        if not parts:
            return "Empty substrate."
        return " ".join(parts)

    def _render_telnet(self, **kwargs) -> str:
        """Render the substrate as a CLI-friendly text dump.

        Format: lines like "addr\tvalue\tconf\twitness_count\tconvoy_size"
        Suitable for `ssh substrate@host cat /var/substrate/telnet.txt`.
        """
        lines = ["# substrate telnet view", f"# cells: {len(self._cells)}", ""]
        lines.append("address\tvalue\tconf\twitness_count\tconvoy_size")
        for c in sorted(self._cells.values(), key=lambda c: c.address):
            val = c.value
            if isinstance(val, float):
                val = f"{val:.3f}"
            lines.append(f"{c.address}\t{val}\t{c.confidence:.3f}\t{len(c.witness_log)}\t{len(c.convoy)}")
        return "\n".join(lines)

    def _render_gesture(self, **kwargs) -> Dict:
        """Render the substrate as a JSON description for touch input.

        Returns a dict with 'gestures' suitable for a touchscreen. Each
        cell becomes a tappable region. Use this for the substrate IDE
        on a tablet (paper 117, Open Q6).
        """
        gestures = []
        for c in self._cells.values():
            gestures.append({
                "id": c.address,
                "tap": {"action": "observe", "target": c.address},
                "long_press": {"action": "witness", "target": c.address},
                "swipe_right": {"action": "refresh", "target": c.address},
            })
        return {"gestures": gestures, "n": len(gestures)}

    def _render_flowchart(self, **kwargs) -> str:
        """Render the substrate as a Graphviz DOT graph.

        The DOT format can be piped to `dot -Tpng` to produce an image.
        Use this for documentation, for arch diagrams, for visual debugging.
        """
        lines = ["digraph substrate {", "  rankdir=LR;", "  node [shape=box];"]
        for c in self._cells.values():
            val = c.value
            if isinstance(val, float):
                val = f"{val:.2f}"
            label = f"{c.address}\\n{val}"
            lines.append(f'  "{c.address}" [label="{label}"];')
        for c in self._cells.values():
            for k, other in c.inputs.items():
                lines.append(f'  "{other.address}" -> "{c.address}";')
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "cells": [c.to_dict() for c in self._cells.values()],
            "n_cells": len(self._cells),
            "t": self._t,
            "beta_0": self.betti_0(),
            "beta_1": self.betti_1(),
            "n_edges": len(self.edges()),
        }

    # -- Merkle tree of all witness roots (paper 117, Open Q4) --

    def merkle_root(self) -> str:
        """The Merkle root of all cells' witness roots.

        For a substrate with N cells, this lets you prove a single cell's
        witness log was not modified with O(log N) work, instead of O(N)
        for the chain approach.

        The tree is built by:
        1. Collecting all (cell_address, witness_root) pairs
        2. Hashing each pair to get the leaves
        3. Pairwise combining until one root remains

        Leaves are sorted by (address, witness_root) to ensure deterministic
        ordering even if two addresses produce the same hash.
        """
        if not self._cells:
            return "0" * 16
        # Step 1: collect leaves, sorted by (address, witness_root) for determinism
        leaves = sorted(
            (_hash(f"{addr}:{c.witness_root}"), addr) for addr, c in self._cells.items()
        )
        # Step 2-3: pairwise hash until one root
        while len(leaves) > 1:
            new_level = []
            for i in range(0, len(leaves) - 1, 2):
                new_level.append((_hash(leaves[i][0] + leaves[i + 1][0]), leaves[i][1]))
            if len(leaves) % 2 == 1:
                new_level.append(leaves[-1])  # odd one out
            leaves = new_level
        return leaves[0][0]

    def merkle_proof(self, address: str) -> Optional[List[Tuple[str, str]]]:
        """Return a Merkle proof that `address`'s witness log is in the tree.

        Returns a list of (sibling_hash, position) tuples, where position is
        'left' (sibling is on the left, target on the right) or 'right'.

        None if the address doesn't exist.

        Implementation: build the tree level by level. At each level, find
        the target's index, get its sibling, and append to the proof. The
        target's index becomes the parent's index in the next level.
        """
        if address not in self._cells:
            return None
        # Collect leaves: (hash, address) pairs, sorted
        leaves = sorted(
            (_hash(f"{addr}:{c.witness_root}"), addr) for addr, c in self._cells.items()
        )
        # Find target's index
        idx = next((i for i, (_, a) in enumerate(leaves) if a == address), None)
        if idx is None:
            return None
        proof = []
        current = leaves
        while len(current) > 1:
            # Find sibling
            if idx % 2 == 0:  # target is left
                sibling_idx = idx + 1
                if sibling_idx >= len(current):
                    # Odd one out — duplicate self
                    sibling_hash = current[idx][0]
                    proof.append((sibling_hash, "right"))
                else:
                    proof.append((current[sibling_idx][0], "right"))
            else:  # target is right
                proof.append((current[idx - 1][0], "left"))
            # Compute the new level and the target's index in it
            new_level = []
            for i in range(0, len(current) - 1, 2):
                new_level.append((_hash(current[i][0] + current[i + 1][0]), current[i][1]))
            if len(current) % 2 == 1:
                new_level.append(current[-1])
            # Target's index in new level is idx // 2
            idx = idx // 2
            current = new_level
        return proof

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def __repr__(self) -> str:
        return f"Substrate({len(self._cells)} cells, t={self._t:.1f})"
