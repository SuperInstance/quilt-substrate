"""deckhand_witness.py — A persistent witness backed by deckhand's BM25 index.

The Quilt's Witness primitive stores events in memory. The deckhand-backed
witness stores them on disk and lets you query them later.

Each witness event is a document. deckhand's BM25 indexes the text. Queries
return the most relevant events.

The Quilt's plugin can use this as a drop-in for the in-memory witness:
- `remember(event)` — store an event
- `recall(query, k=5)` — find similar events
- `export_jsonl(path)` — export for saddle/ingest

This is the F/V EILEEN's persistent memory: the cowboy can ask "what models
did the plugin try yesterday?" and deckhand's BM25 will answer.
"""
from __future__ import annotations
import json
import os
import re
import math
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# -- Pure Python BM25 (deckhand's algorithm) ---------------------------

@dataclass
class WitnessEvent:
    """A single witness event."""
    event_id: str = ""
    ts: float = 0.0
    text: str = ""                # the searchable text
    kind: str = "unknown"         # "cast.proposed" | "cast.observed" | etc.
    model: str = ""
    opener: str = ""
    primitive: str = ""
    user: str = ""
    app: str = ""
    quality: float = 0.0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeckhandWitness:
    """A persistent witness backed by deckhand's BM25 (pure Python)."""

    def __init__(self, witness_path: str = "data/witness.jsonl",
                  auto_persist: bool = True):
        self.witness_path = Path(witness_path)
        self.witness_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_persist = auto_persist
        self.events: List[WitnessEvent] = self._load()
        # BM25 stats
        self._doc_tokens: List[List[str]] = []
        self._doc_freq: Counter = Counter()
        self._doc_lens: List[int] = []
        self._avg_dl: float = 0.0
        self._N: int = 0
        self._build_index()

    def _load(self) -> List[WitnessEvent]:
        """Load events from disk."""
        if not self.witness_path.exists():
            return []
        events = []
        with open(self.witness_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    events.append(WitnessEvent(**d))
                except (json.JSONDecodeError, TypeError):
                    pass
        return events

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build_index(self):
        """Build BM25 statistics from the current events."""
        self._doc_tokens = []
        self._doc_freq = Counter()
        self._doc_lens = []
        for e in self.events:
            tokens = self._tokenize(e.text)
            self._doc_tokens.append(tokens)
            self._doc_lens.append(len(tokens))
            for t in set(tokens):
                self._doc_freq[t] += 1
        self._N = len(self.events)
        self._avg_dl = (sum(self._doc_lens) / self._N) if self._N > 0 else 0.0

    def _idf(self, term: str) -> float:
        """Inverse document frequency for a term."""
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (self._N - df + 0.5) / (df + 0.5))

    def _bm25_score(self, query_terms: List[str], doc_idx: int,
                     k1: float = 1.5, b: float = 0.75) -> float:
        """BM25 score for a query against a document."""
        score = 0.0
        doc_tokens = self._doc_tokens[doc_idx]
        doc_len = self._doc_lens[doc_idx]
        if self._avg_dl == 0:
            return 0.0
        term_counts = Counter(doc_tokens)
        for qt in query_terms:
            if qt not in term_counts:
                continue
            tf = term_counts[qt]
            idf = self._idf(qt)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / self._avg_dl)
            score += idf * (numerator / denominator)
        return score

    def remember(self, event: Dict[str, Any]) -> str:
        """Store an event. Returns the event_id."""
        if "text" not in event:
            # Default text = a serializable representation
            event["text"] = json.dumps(event, default=str)
        if "ts" not in event:
            event["ts"] = time.time()
        if "event_id" not in event:
            event["event_id"] = f"e_{int(event['ts'] * 1000)}"
        wevent = WitnessEvent(**event)
        self.events.append(wevent)
        # Rebuild index (incremental rebuild is more efficient but this is simple)
        self._build_index()
        # Persist
        if self.auto_persist:
            with open(self.witness_path, "a") as f:
                f.write(json.dumps(wevent.to_dict()) + "\n")
        return wevent.event_id

    def recall(self, query: str, k: int = 5,
                 kind: Optional[str] = None,
                 model: Optional[str] = None,
                 user: Optional[str] = None) -> List[WitnessEvent]:
        """Find the k most relevant events for a query."""
        if not self.events:
            return []
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        # Score all events
        scored = []
        for i, e in enumerate(self.events):
            score = self._bm25_score(query_terms, i)
            # Apply metadata filters
            if kind and e.kind != kind:
                continue
            if model and e.model != model:
                continue
            if user and e.user != user:
                continue
            if score > 0:
                scored.append((score, e))
        # Sort by score desc
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def export_jsonl(self, path: str) -> int:
        """Export all events to a JSONL file (saddle-compatible format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "w") as f:
            for e in self.events:
                f.write(json.dumps(e.to_dict()) + "\n")
                n += 1
        return n

    def stats(self) -> Dict[str, Any]:
        """Return aggregate stats about the witness."""
        if not self.events:
            return {"n_events": 0, "n_unique_terms": 0, "avg_doc_len": 0.0}
        return {
            "n_events": len(self.events),
            "n_unique_terms": len(self._doc_freq),
            "avg_doc_len": self._avg_dl,
            "witness_path": str(self.witness_path),
            "models": list(set(e.model for e in self.events if e.model)),
            "kinds": list(set(e.kind for e in self.events)),
        }

    def clear(self):
        """Clear all events (and the on-disk file)."""
        self.events = []
        self._build_index()
        if self.witness_path.exists():
            self.witness_path.unlink()
