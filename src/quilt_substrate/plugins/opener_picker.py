"""opener_picker.py — Pick the right opener for a (primitive, role, context).

The substrate has many openers (chart, voice, tide, mud, slate, witness,
reef, etc.) and the right opener depends on what the user wants to see.

This is a learned picker:
- For each (primitive, role, context), track which opener was used
  and how successful the render was
- Use Wilson lower bound to pick the opener with the best track record
- Fall back to a heuristic prior if no data

The cowboy's refinement (auto-retire failing openers) feeds back here.
"""
from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Heuristic prior: which opener is best for which (primitive, role)
OPENER_PRIOR: Dict[Tuple[str, str], List[Tuple[str, float]]] = {
    # Primitives
    ("Murmur", "any"): [("tide", 0.7), ("chart", 0.5), ("slate", 0.3)],
    ("JEPA", "any"): [("reef", 0.7), ("graph", 0.5), ("chart", 0.3)],
    ("Z_in", "any"): [("witness", 0.7), ("list", 0.5), ("tensor", 0.3)],
    ("Z_out", "any"): [("voice", 0.7), ("slate", 0.5), ("rest", 0.3)],
    ("DoubleEntry", "any"): [("ledger", 0.7), ("list", 0.5), ("chart", 0.3)],
    ("Convoy", "any"): [("convoy", 0.7), ("graph", 0.5), ("flowchart", 0.3)],
    # Roles
    ("any", "fable_compression"): [("slate", 0.8), ("tide", 0.5), ("voice", 0.3)],
    ("any", "voice_narration"): [("voice", 0.8), ("slate", 0.5), ("witness", 0.3)],
    ("any", "sensory_creative"): [("tide", 0.8), ("slate", 0.5), ("voice", 0.3)],
    ("any", "math_grief"): [("reef", 0.8), ("chart", 0.5), ("slate", 0.3)],
    ("any", "creative_ideation"): [("slate", 0.7), ("voice", 0.5), ("tide", 0.3)],
    ("any", "safety_check"): [("witness", 0.8), ("list", 0.5), ("reef", 0.3)],
}


# All known openers (the substrate's full set)
ALL_OPENERS = [
    "chart", "voice", "tide", "mud", "slate", "witness", "reef",
    "graph", "list", "tensor", "ledger", "convoy", "flowchart",
    "rest", "harbor", "dive", "midi", "gesture", "plato",
]


@dataclass
class OpenerScore:
    opener: str = ""
    n: int = 0
    success: int = 0
    avg_quality: float = 0.0
    last_used: float = 0.0
    retired: bool = False


class OpenerPicker:
    """Learned opener picker with Wilson scoring + heuristic prior."""

    def __init__(self, threshold: float = 0.6, min_obs: int = 3):
        self.threshold = threshold
        self.min_obs = min_obs
        # (primitive, role, opener) → OpenerScore
        self.scores: Dict[Tuple[str, str, str], OpenerScore] = defaultdict(
            lambda: OpenerScore()
        )

    def _key(self, primitive: str, role: str, opener: str) -> Tuple[str, str, str]:
        return (primitive, role, opener)

    def observe(self, primitive: str, role: str, opener: str,
                 success: bool, quality: float = 0.5):
        """Record that an opener was used for a (primitive, role) pair."""
        k = self._key(primitive, role, opener)
        s = self.scores[k]
        s.opener = opener
        s.n += 1
        if success:
            s.success += 1
        s.avg_quality = ((s.avg_quality * (s.n - 1)) + quality) / s.n
        s.last_used = time.time()

    def _wilson_lb(self, success: int, n: int) -> float:
        if n == 0:
            return 0.5
        z = 1.96
        p = success / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
        return max(0.0, center - spread)

    def _prior(self, primitive: str, role: str, opener: str) -> float:
        """Heuristic prior for an opener (primitive, role, opener)."""
        # Check the role-specific prior
        for key, scored_list in OPENER_PRIOR.items():
            prim_key, role_key = key
            prim_match = (prim_key == primitive) or (prim_key == "any")
            role_match = (role_key == role) or (role_key == "any")
            if prim_match and role_match:
                for op, score in scored_list:
                    if op == opener:
                        return score
        return 0.3  # uniform prior

    def pick(self, primitive: str, role: str,
               candidates: Optional[List[str]] = None,
               blacklist: Optional[List[str]] = None) -> Tuple[str, float, str]:
        """Pick the best opener for (primitive, role). Returns (opener, score, reason)."""
        if candidates is None:
            candidates = list(ALL_OPENERS)
        if blacklist is None:
            blacklist = []

        scored = []
        for opener in candidates:
            if opener in blacklist:
                continue
            k = self._key(primitive, role, opener)
            s = self.scores.get(k)
            if s is not None and s.retired:
                continue
            # Wilson score
            if s is not None and s.n >= self.min_obs:
                wilson = self._wilson_lb(s.success, s.n)
            else:
                wilson = 0.5  # optimistic prior
            # Prior score
            prior = self._prior(primitive, role, opener)
            # Blend
            blend = 0.5 * prior + 0.5 * wilson
            scored.append((blend, opener, prior, wilson))

        if not scored:
            # All blacklisted: pick slate (the safe default)
            return ("slate", 0.3, "all-candidates-blacklisted: defaulted to slate")

        scored.sort(reverse=True)
        best_score, best_opener, best_prior, best_wilson = scored[0]
        reason = f"prior={best_prior:.2f}, wilson={best_wilson:.2f}"
        return (best_opener, best_score, reason)

    def retire(self, primitive: str, role: str, opener: str):
        """Mark an opener for retirement in this (primitive, role) context."""
        k = self._key(primitive, role, opener)
        s = self.scores[k]
        s.opener = opener
        s.retired = True

    def restore(self, primitive: str, role: str, opener: str):
        """Un-retire an opener."""
        k = self._key(primitive, role, opener)
        s = self.scores[k]
        s.retired = False

    def stats(self) -> Dict[str, Any]:
        n_obs = sum(s.n for s in self.scores.values())
        n_retired = sum(1 for s in self.scores.values() if s.retired)
        by_opener: Dict[str, int] = defaultdict(int)
        for s in self.scores.values():
            by_opener[s.opener] += s.n
        return {
            "n_keys": len(self.scores),
            "n_obs": n_obs,
            "n_retired": n_retired,
            "by_opener": dict(by_opener),
        }
