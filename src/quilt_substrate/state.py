"""state.py — Persistent state for the Quilt components.

Wilson profiles, LinUCB weights, and witness events are all stored in
memory today. The cowboy needs them across restarts. This module
implements atomic JSON persistence with versioned schema.

Why JSONL? It is append-only, easy to inspect, easy to version, easy
to migrate. The cowboy's memory already uses JSONL. The substrate
should follow the same convention.

Schema versioning: each save records the schema version. The load
fails loudly if the version is unknown, so the cowboy can be told to
run a migration instead of silently corrupting state.
"""
from __future__ import annotations
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar


SCHEMA_VERSION = 1
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic save/load with atomic rename
# ---------------------------------------------------------------------------

def atomic_write_jsonl(path: str, records: List[Dict[str, Any]]):
    """Write records to a JSONL file atomically (write-temp-then-rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    tmp.rename(p)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL records, skip blanks. Returns [] if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_write_json(path: str, payload: Dict[str, Any]):
    """Write a single JSON object atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.rename(p)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a single JSON object. Returns None if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Wilson profiles — save/load
# ---------------------------------------------------------------------------

def wilson_to_dict(obs: Dict[Tuple[str, str, str], List[Tuple[float, float, int, bool]]],
                     threshold: float, window: int, half_life: float) -> Dict[str, Any]:
    """Serialize Wilson profiles state."""
    return {
        "schema": SCHEMA_VERSION,
        "kind": "wilson",
        "threshold": threshold,
        "window": window,
        "half_life": half_life,
        "saved_at": time.time(),
        "obs": {
            f"{p}|{o}|{m}": [
                {"q": q, "ts": ts, "latency_ms": lat, "success": bool(s)}
                for q, ts, lat, s in entries
            ]
            for (p, o, m), entries in obs.items()
        },
    }


def wilson_from_dict(d: Dict[str, Any]) -> Tuple[float, int, float, Dict[Tuple[str, str, str], List[Tuple[float, float, int, bool]]]]:
    """Deserialize Wilson profiles state. Returns (threshold, window, half_life, obs)."""
    if d.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"Unknown schema: {d.get('schema')}, expected {SCHEMA_VERSION}")
    threshold = d.get("threshold", 0.6)
    window = d.get("window", 200)
    half_life = d.get("half_life", 3600.0)
    obs = defaultdict(list)
    for k, entries in d.get("obs", {}).items():
        parts = k.split("|")
        if len(parts) != 3:
            continue
        p, o, m = parts
        for e in entries:
            obs[(p, o, m)].append(
                (e["q"], e["ts"], e["latency_ms"], e["success"])
            )
    return threshold, window, half_life, dict(obs)


def save_wilson(wilson, path: str):
    """Save a WilsonProfiles instance to disk."""
    payload = wilson_to_dict(dict(wilson.obs), wilson.threshold,
                                wilson.window, wilson.half_life)
    atomic_write_json(path, payload)


def load_wilson(path: str):
    """Load a WilsonProfiles instance from disk. Returns None if no file."""
    d = load_json(path)
    if d is None:
        return None
    from quilt_substrate.plugins.casting import WilsonProfiles
    threshold, window, half_life, obs = wilson_from_dict(d)
    w = WilsonProfiles(threshold=threshold, window=window, half_life_s=half_life)
    for k, v in obs.items():
        w.obs[k] = v
    return w


# ---------------------------------------------------------------------------
# LinUCB weights — save/load
# ---------------------------------------------------------------------------

def linucb_to_dict(linucb) -> Dict[str, Any]:
    """Serialize a LinUCBCaster state."""
    models = {}
    for key, m in linucb.models.items():
        key_str = f"{key[0]}|{key[1]}"
        # A is a matrix, b is a vector
        # LinUCBModel has .d (the feature dim), not .feature_dim
        d_dim = getattr(m, "d", getattr(m, "feature_dim", 64))
        models[key_str] = {
            "A": m.A.tolist() if hasattr(m.A, "tolist") else list(m.A),
            "b": m.b.tolist() if hasattr(m.b, "tolist") else list(m.b),
            "n": m.n,
            "feature_dim": d_dim,
        }
    # feature_dim is on the extractor, not the caster itself
    feature_dim = (linucb.extractor.dim if hasattr(linucb, "extractor")
                     and linucb.extractor is not None
                     and hasattr(linucb.extractor, "dim") else 64)
    return {
        "schema": SCHEMA_VERSION,
        "kind": "linucb",
        "saved_at": time.time(),
        "feature_dim": feature_dim,
        "models": models,
    }


def linucb_from_dict(d: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
    """Deserialize LinUCB state. Returns a dict of (user, app) → model data.

    The caller is responsible for plugging these into a LinUCBCaster (which
    needs a plugin reference). This keeps state.py decoupled from the
    LinUCBCaster's constructor.
    """
    if d.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"Unknown schema: {d.get('schema')}, expected {SCHEMA_VERSION}")
    models = {}
    for key_str, m_data in d.get("models", {}).items():
        user, app = key_str.split("|", 1)
        from quilt_substrate.plugins.linucb import LinUCBModel
        model = LinUCBModel(
            d=m_data.get("feature_dim", d.get("feature_dim", 64)),
        )
        try:
            import numpy as np
            model.A = np.array(m_data["A"])
            model.b = np.array(m_data["b"])
        except ImportError:
            model.A = m_data["A"]
            model.b = m_data["b"]
        model.n = m_data["n"]
        models[(user, app)] = model
    return models


def save_linucb(linucb, path: str):
    payload = linucb_to_dict(linucb)
    atomic_write_json(path, payload)


def load_linucb_models(path: str) -> Optional[Dict[Tuple[str, str], Any]]:
    """Load just the model dict. Returns None if no file."""
    d = load_json(path)
    if d is None:
        return None
    return linucb_from_dict(d)


# ---------------------------------------------------------------------------
# Generic state manager
# ---------------------------------------------------------------------------

class StateManager:
    """Manages persistent state for a Quilt deployment.

    Layout:
        state_dir/
            wilson.json         # Wilson profiles
            linucb.json         # LinUCB weights
            witness.jsonl       # Witness events (already done in deckhand_witness)
            cowboy.jsonl        # Cowboy memory (already done in cowboy)
            bridge/             # Saddle ledger
                ledger.jsonl
                frozens/
    """

    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def wilson_path(self) -> str:
        return str(self.state_dir / "wilson.json")

    @property
    def linucb_path(self) -> str:
        return str(self.state_dir / "linucb.json")

    @property
    def witness_path(self) -> str:
        return str(self.state_dir / "witness.jsonl")

    @property
    def cowboy_path(self) -> str:
        return str(self.state_dir / "cowboy.jsonl")

    def save_wilson(self, wilson):
        save_wilson(wilson, self.wilson_path)
        return self.wilson_path

    def load_wilson(self):
        return load_wilson(self.wilson_path)

    def save_linucb(self, linucb):
        save_linucb(linucb, self.linucb_path)
        return self.linucb_path

    def load_linucb_models(self):
        """Load LinUCB model data. Caller must plug into a LinUCBCaster."""
        return load_linucb_models(self.linucb_path)

    def exists(self) -> bool:
        """Return True if any state file exists."""
        return any([
            Path(self.wilson_path).exists(),
            Path(self.linucb_path).exists(),
            Path(self.witness_path).exists(),
            Path(self.cowboy_path).exists(),
        ])

    def list_files(self) -> List[str]:
        return [
            p.name for p in self.state_dir.iterdir()
            if p.is_file() and not p.name.endswith(".tmp")
        ]
