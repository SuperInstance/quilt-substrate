"""test_deckhand_witness.py — Tests for the deckhand-backed witness."""
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_substrate.plugins.deckhand_witness import (
    DeckhandWitness, WitnessEvent,
)


def make_witness(path: str = None) -> DeckhandWitness:
    if path is None:
        # Use a temp path
        import tempfile
        d = tempfile.mkdtemp()
        path = str(Path(d) / "witness.jsonl")
    return DeckhandWitness(witness_path=path)


def test_remember_event():
    """An event is stored and retrievable."""
    w = make_witness()
    eid = w.remember({"text": "the sailor asked the depth at lat 24.5", "kind": "cast.observed",
                       "model": "HERMES_405B"})
    assert eid is not None
    assert len(w.events) == 1


def test_recall_finds_relevant():
    """BM25 finds events relevant to a query."""
    w = make_witness()
    w.remember({"text": "the sailor asked the depth at lat 24.5", "model": "HERMES_405B"})
    w.remember({"text": "the tide is ebbing, current at 2 knots", "model": "SEED_MINI"})
    w.remember({"text": "the wind shifted NW at 0300", "model": "QWEN_0_5B"})
    results = w.recall("depth")
    assert len(results) >= 1
    # The depth event should be the most relevant
    assert "depth" in results[0].text.lower()


def test_recall_with_model_filter():
    """Recall can filter by model."""
    w = make_witness()
    w.remember({"text": "the depth at lat 24.5", "model": "HERMES_405B"})
    w.remember({"text": "the depth at lat 25.0", "model": "SEED_MINI"})
    w.remember({"text": "the wind shifted", "model": "QWEN_0_5B"})
    results = w.recall("depth", model="SEED_MINI", k=5)
    assert all(e.model == "SEED_MINI" for e in results)


def test_recall_with_kind_filter():
    """Recall can filter by kind."""
    w = make_witness()
    w.remember({"text": "the depth at lat 24.5", "kind": "cast.observed"})
    w.remember({"text": "the depth at lat 24.5", "kind": "cast.proposed"})
    results = w.recall("depth", kind="cast.observed", k=5)
    assert all(e.kind == "cast.observed" for e in results)


def test_recall_empty_witness():
    """Recall on an empty witness returns no results."""
    w = make_witness()
    assert w.recall("anything") == []


def test_recall_top_k():
    """Recall respects the k parameter."""
    w = make_witness()
    for i in range(10):
        w.remember({"text": f"depth at lat {i}", "model": f"M{i}"})
    results = w.recall("depth", k=3)
    assert len(results) == 3


def test_persistence():
    """Events persist to disk and reload."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "witness.jsonl")
        w1 = DeckhandWitness(witness_path=path)
        w1.remember({"text": "the depth at lat 24.5", "model": "HERMES_405B"})
        # Reload
        w2 = DeckhandWitness(witness_path=path)
        assert len(w2.events) == 1
        results = w2.recall("depth")
        assert len(results) == 1


def test_export_jsonl():
    """The witness exports to JSONL."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "witness.jsonl")
        export_path = str(Path(d) / "export.jsonl")
        w = DeckhandWitness(witness_path=path)
        w.remember({"text": "the depth at lat 24.5", "model": "HERMES_405B"})
        w.remember({"text": "the tide is ebbing", "model": "SEED_MINI"})
        n = w.export_jsonl(export_path)
        assert n == 2
        with open(export_path) as f:
            lines = f.readlines()
        assert len(lines) == 2


def test_stats():
    """Stats reflect the witness contents."""
    w = make_witness()
    w.remember({"text": "the depth at lat 24.5", "model": "HERMES_405B"})
    w.remember({"text": "the tide is ebbing", "model": "SEED_MINI"})
    s = w.stats()
    assert s["n_events"] == 2
    assert "HERMES_405B" in s["models"]
    assert "SEED_MINI" in s["models"]


def test_clear():
    """Clear removes all events."""
    w = make_witness()
    w.remember({"text": "x", "model": "A"})
    w.remember({"text": "y", "model": "B"})
    w.clear()
    assert len(w.events) == 0
    assert w.recall("x") == []


def test_tokenize():
    """Tokenization works on simple text."""
    w = make_witness()
    assert w._tokenize("Hello, World!") == ["hello", "world"]
    assert w._tokenize("The 2B model") == ["the", "2b", "model"]


def test_idf():
    """IDF increases for rarer terms."""
    w = make_witness()
    w.remember({"text": "the cat sat", "model": "A"})
    w.remember({"text": "the dog sat", "model": "B"})
    w.remember({"text": "the bird flew", "model": "C"})
    # "cat" is rarer than "the"
    assert w._idf("cat") > w._idf("the")


def test_bm25_score_zero_for_no_match():
    """BM25 score is 0 if no terms match."""
    w = make_witness()
    w.remember({"text": "the cat sat", "model": "A"})
    assert w._bm25_score(["elephant"], 0) == 0.0


def test_witness_event_to_dict():
    """WitnessEvent can be serialized to dict."""
    e = WitnessEvent(
        event_id="e1", ts=123.0, text="hello", kind="cast.observed",
        model="HERMES_405B", opener="voice", primitive="Murmur",
        user="casey", app="writers-room", quality=0.9, success=True,
    )
    d = e.to_dict()
    assert d["event_id"] == "e1"
    assert d["model"] == "HERMES_405B"
    assert d["quality"] == 0.9


def test_recall_returns_scored_descending():
    """Recall returns results in score-descending order."""
    w = make_witness()
    # Highly relevant: contains query multiple times
    w.remember({"text": "depth depth depth at lat 24.5", "model": "A"})
    # Less relevant: contains query once
    w.remember({"text": "wind depth at lat 25", "model": "B"})
    # Not relevant
    w.remember({"text": "the wind shifted", "model": "C"})
    results = w.recall("depth", k=3)
    # The first result should be the most depth-heavy one
    assert "depth depth" in results[0].text


def test_recall_empty_query():
    """An empty query returns no results."""
    w = make_witness()
    w.remember({"text": "the depth at lat 24.5"})
    assert w.recall("") == []


if __name__ == "__main__":
    import inspect
    tests = [v for k, v in globals().items()
              if k.startswith("test_") and callable(v) and inspect.isfunction(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
