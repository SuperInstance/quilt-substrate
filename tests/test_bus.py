"""test_bus.py — Tests for the in-process event bus."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/workspace/quilt-substrate/src")

from quilt_substrate.bus import EventBus, Event, BusLogger


def test_bus_subscribe_and_publish():
    bus = EventBus()
    received = []
    bus.subscribe("cast.observed", lambda e: received.append(e))
    e = bus.publish("cast.observed", source="plugin", data={"model": "PHI-4"})
    assert len(received) == 1
    assert received[0].topic == "cast.observed"
    assert received[0].data["model"] == "PHI-4"


def test_bus_multiple_subscribers():
    bus = EventBus()
    a, b = [], []
    bus.subscribe("cast.observed", lambda e: a.append(e))
    bus.subscribe("cast.observed", lambda e: b.append(e))
    bus.publish("cast.observed", source="plugin")
    assert len(a) == 1 and len(b) == 1


def test_bus_publish_with_no_subscribers():
    bus = EventBus()
    e = bus.publish("cast.observed", source="plugin")
    # Should not raise, event still recorded
    assert e.topic == "cast.observed"
    assert bus.history()[0].topic == "cast.observed"


def test_bus_history_filtered_by_topic():
    bus = EventBus()
    bus.publish("cast.proposed", source="plugin")
    bus.publish("cast.observed", source="plugin")
    bus.publish("cast.proposed", source="plugin")
    proposed = bus.history(topic="cast.proposed")
    assert len(proposed) == 2
    observed = bus.history(topic="cast.observed")
    assert len(observed) == 1


def test_bus_history_limit():
    bus = EventBus()
    for i in range(20):
        bus.publish("test.event", source="x", data={"i": i})
    last_5 = bus.history(limit=5)
    assert len(last_5) == 5
    assert last_5[-1].data["i"] == 19


def test_bus_max_history_trim():
    bus = EventBus(max_history=5)
    for i in range(20):
        bus.publish("test.event", source="x")
    assert len(bus.history_list) == 5


def test_bus_unsubscribe():
    bus = EventBus()
    received = []
    unsub = bus.subscribe("cast.observed", lambda e: received.append(e))
    bus.publish("cast.observed", source="x")
    assert len(received) == 1
    unsub()
    bus.publish("cast.observed", source="x")
    assert len(received) == 1  # no new event


def test_bus_pattern_subscription():
    bus = EventBus()
    received = []
    bus.subscribe_pattern("cast.*", lambda e: received.append(e))
    bus.publish("cast.proposed", source="x")
    bus.publish("cast.observed", source="x")
    bus.publish("ledger.appended", source="x")
    assert len(received) == 2


def test_bus_pattern_prefix_only():
    bus = EventBus()
    received = []
    bus.subscribe_pattern("cast.*", lambda e: received.append(e))
    bus.publish("cast.proposed", source="x")
    bus.publish("cast.observed", source="x")
    bus.publish("castx", source="x")
    assert len(received) == 2


def test_bus_pattern_suffix_only():
    bus = EventBus()
    received = []
    bus.subscribe_pattern("*.observed", lambda e: received.append(e))
    bus.publish("cast.observed", source="x")
    bus.publish("witness.observed", source="x")
    bus.publish("cast.proposed", source="x")
    assert len(received) == 2


def test_bus_pattern_middle_wildcard():
    bus = EventBus()
    received = []
    bus.subscribe_pattern("cast.*.observed", lambda e: received.append(e))
    bus.publish("cast.model.observed", source="x")
    bus.publish("cast.observed", source="x")
    assert len(received) == 1


def test_bus_subscriber_exception_doesnt_break():
    bus = EventBus()
    received = []

    def bad(e):
        raise ValueError("oops")

    bus.subscribe("test.event", bad)
    bus.subscribe("test.event", lambda e: received.append(e))
    bus.publish("test.event", source="x")
    # The good subscriber should still receive
    assert len(received) == 1


def test_bus_publish_returns_event():
    bus = EventBus()
    e = bus.publish("test.event", source="x", data={"k": 1})
    assert isinstance(e, Event)
    assert e.ts > 0
    assert e.topic == "test.event"
    assert e.source == "x"
    assert e.data["k"] == 1


def test_bus_event_to_jsonl():
    e = Event(ts=100.0, topic="t", source="s", data={"k": 1})
    j = e.to_jsonl()
    d = json.loads(j)
    assert d["topic"] == "t"
    assert d["data"]["k"] == 1


def test_bus_save_load_jsonl():
    bus = EventBus()
    bus.publish("a", source="x", data={"i": 1})
    bus.publish("b", source="x", data={"i": 2})
    bus.publish("c", source="x", data={"i": 3})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "events.jsonl")
        bus.save_jsonl(path)
        bus2 = EventBus()
        bus2.load_jsonl(path)
        assert len(bus2.history_list) == 3
        assert bus2.history_list[0].topic == "a"
        assert bus2.history_list[2].topic == "c"


def test_bus_counts():
    bus = EventBus()
    bus.publish("a", source="x")
    bus.publish("a", source="x")
    bus.publish("b", source="x")
    counts = bus.counts_summary()
    assert counts["a"] == 2
    assert counts["b"] == 1


def test_bus_stats():
    bus = EventBus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("b", lambda e: None)
    bus.publish("a", source="x")
    s = bus.stats()
    assert s["n_events"] == 1
    assert s["n_subscribers"] == 2


def test_bus_logger_writes_to_file():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.txt")
        logger = BusLogger(path=path)
        bus.subscribe("test.event", logger)
        bus.publish("test.event", source="plugin", data={"k": 1})
        bus.publish("test.event", source="plugin", data={"k": 2})
        with open(path) as f:
            content = f.read()
        assert "test.event" in content
        assert content.count("\n") == 2


def test_bus_pattern_no_wildcard_no_match():
    bus = EventBus()
    received = []
    bus.subscribe_pattern("cast.proposed", lambda e: received.append(e))
    bus.publish("cast.observed", source="x")
    bus.publish("cast.proposed.sub", source="x")
    assert len(received) == 0


def test_bus_complex_topology():
    """Test a realistic event flow."""
    bus = EventBus()
    cowboy_actions = []
    witness_events = []

    def cowboy_listener(e):
        if e.data.get("wilson_lb", 1.0) < 0.2 and e.data.get("n", 0) >= 3:
            cowboy_actions.append(("retire", e.data.get("model")))

    def witness_listener(e):
        witness_events.append(e)

    bus.subscribe_pattern("cast.*", cowboy_listener)
    bus.subscribe("witness.appended", witness_listener)

    # Simulate some casts
    for _ in range(5):
        bus.publish("cast.observed", source="plugin",
                       data={"model": "BROKEN", "n": 5, "wilson_lb": 0.0})
    # cowboy should retire BROKEN
    assert ("retire", "BROKEN") in cowboy_actions
    # witness listener wasn't fired (no witness.appended events yet)
    assert len(witness_events) == 0
