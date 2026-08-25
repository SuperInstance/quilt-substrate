"""bus.py — A simple in-process event bus.

The substrate, plugin, witness, ledger, and cowboy all want to react to
each other. The cowboy wants to retire a failing alignment the moment
the witness sees three failures. The writer wants to log every cast.

A pub/sub bus is the simplest way to wire them without coupling each
component to every other.

This is in-process only. No Kafka, no Redis. Just a list of subscribers
and a publish method. The cowboy runs locally on the F/V EILEEN, so
in-process is the right scale.

Event shape:
    {
        "ts": float,            # time.time()
        "topic": str,           # e.g. "cast.observed", "witness.appended"
        "source": str,          # who published ("substrate", "plugin", "cowboy")
        "data": dict,           # payload
    }

Topics are dot-namespaced: `<domain>.<verb>`. Examples:
- "cast.proposed" — plugin proposed a casting
- "cast.observed" — outcome observed
- "witness.appended" — witness stored an event
- "ledger.appended" — saddle ledger grew
- "cowboy.morning" — cowboy ran the morning
- "model.retired" — cowboy retired a model
- "model.promoted" — cowboy promoted a model
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Event:
    ts: float = 0.0
    topic: str = ""
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), default=str)


Subscriber = Callable[[Event], None]


class EventBus:
    """A pub/sub bus for in-process events.

    - subscribe(topic, fn) — register a callback
    - publish(topic, source, data) — fire an event
    - history(topic=None) — replay past events
    - save_jsonl(path) / load_jsonl(path) — persist event log
    """

    def __init__(self, max_history: int = 10000):
        self.subscribers: Dict[str, List[Subscriber]] = defaultdict(list)
        self.history_list: List[Event] = []
        self.max_history = max_history
        # Track per-topic counters
        self.counts: Dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, fn: Subscriber) -> Callable[[], None]:
        """Register a callback. Returns an unsubscribe function."""
        self.subscribers[topic].append(fn)

        def _unsubscribe():
            if fn in self.subscribers[topic]:
                self.subscribers[topic].remove(fn)
        return _unsubscribe

    def subscribe_pattern(self, pattern: str, fn: Subscriber) -> Callable[[], None]:
        """Subscribe to all topics matching a glob-like pattern.

        pattern uses '*' as a wildcard. e.g. 'cast.*' matches 'cast.proposed'
        and 'cast.observed'. We use simple string matching.
        """
        def _wrapped(event: Event):
            if self._matches(pattern, event.topic):
                fn(event)
        return self.subscribe(pattern, _wrapped)

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        if pattern == topic:
            return True
        if "*" not in pattern:
            return False
        # For middle wildcards like 'cast.*.observed':
        # - 'cast.X.observed' should match (X is non-empty)
        # - 'cast.observed' should NOT match (missing middle segment)
        if pattern.count("*") == 1:
            prefix, suffix = pattern.split("*", 1)
            if not prefix and not suffix:
                return True  # pattern is just "*"
            if not prefix:
                return len(topic) > 0 and topic.endswith(suffix)
            if not suffix:
                return len(topic) > 0 and topic.startswith(prefix)
            # Middle wildcard: middle must be non-empty
            if not (topic.startswith(prefix) and topic.endswith(suffix)):
                return False
            middle = topic[len(prefix):len(topic) - len(suffix)] if len(suffix) > 0 else topic[len(prefix):]
            if suffix:
                middle = topic[len(prefix):len(topic) - len(suffix)]
            else:
                middle = topic[len(prefix):]
            return len(middle) > 0
        # Multiple wildcards: fall back to split-and-match
        parts = pattern.split("*")
        if not topic.startswith(parts[0]):
            return False
        if not topic.endswith(parts[-1]):
            return False
        rest = topic[len(parts[0]):len(topic) - len(parts[-1]) if parts[-1] else len(topic)]
        idx = 0
        for part in parts[1:-1]:
            if not part:
                continue
            found = rest.find(part, idx)
            if found == -1:
                return False
            idx = found + len(part)
        return True

    def publish(self, topic: str, source: str = "", data: Optional[Dict[str, Any]] = None) -> Event:
        """Publish an event to all subscribers and record in history."""
        e = Event(ts=time.time(), topic=topic, source=source, data=data or {})
        # Add to history
        self.history_list.append(e)
        if len(self.history_list) > self.max_history:
            self.history_list.pop(0)
        self.counts[topic] += 1
        # Notify exact subscribers
        for fn in self.subscribers.get(topic, []):
            try:
                fn(e)
            except Exception:
                pass  # don't let one bad subscriber break the bus
        # Notify pattern subscribers
        for pattern, fns in self.subscribers.items():
            if "*" in pattern and self._matches(pattern, topic):
                for fn in fns:
                    try:
                        fn(e)
                    except Exception:
                        pass
        return e

    def history(self, topic: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get recent events, optionally filtered by topic."""
        if topic is None:
            return self.history_list[-limit:]
        filtered = [e for e in self.history_list if e.topic == topic]
        return filtered[-limit:]

    def counts_summary(self) -> Dict[str, int]:
        return dict(self.counts)

    def save_jsonl(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            for e in self.history_list:
                f.write(e.to_jsonl() + "\n")
        tmp.rename(p)

    def load_jsonl(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                e = Event(**d)
                self.history_list.append(e)
                self.counts[e.topic] += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "n_events": len(self.history_list),
            "n_subscribers": sum(len(s) for s in self.subscribers.values()),
            "topics": list(self.counts.keys()),
            "counts": dict(self.counts),
        }


# -- A simple bus-backed logger --------------------------------------

class BusLogger:
    """A subscriber that logs every event to a file or stdout."""

    def __init__(self, path: Optional[str] = None, stdout: bool = False):
        self.path = path
        self.stdout = stdout

    def __call__(self, event: Event):
        line = f"[{event.ts:.2f}] {event.topic:30s} source={event.source:15s} data={event.data}\n"
        if self.stdout:
            print(line, end="")
        if self.path:
            with open(self.path, "a") as f:
                f.write(line)
