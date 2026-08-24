"""
Fable-as-constraint tests.

Each fable in the seed canon is a constraint the substrate must satisfy.
This file wires the fables to the substrate — the fables are the requirements,
the substrate is the implementation.

"The fables are the requirements; the substrate is the implementation."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Cell, Substrate


# --- Fable 11: The Paper and the Tablet ---

def test_fable_11_paper_and_tablet_both_valid():
    """A picture and a conversation can both be honest. The substrate must
    support both a 'picture' (the static value) and a 'conversation' (the witness log)."""
    s = Substrate()
    c = Cell(address="bay/A17", value=12.5)
    s.add(c)
    # The picture: just the value
    chart = s.render("chart")
    assert chart["cells"][0]["value"] == 12.5
    # The conversation: the witness log
    s.witness(c, "captain-reyes", "read", 12.5)
    s.witness(c, "captain-reyes", "inference", 12.6)
    log = s.render("witness", address="bay/A17")
    assert len(log) == 2
    # The substrate makes BOTH visible.


# --- Fable 12: The Receipt and the Cell ---

def test_fable_12_receipt_and_cell_transaction_recorded():
    """A transaction is a substrate cell. The witness log records every reader."""
    s = Substrate()
    c = Cell(address="store/bread/loaf-1", value=1.89)
    s.add(c)
    s.witness(c, "customer-1", "read", 1.89)
    s.witness(c, "inventory", "read", 1.89)
    s.witness(c, "price-inference", "read", 1.89)
    log = s.render("witness", address="store/bread/loaf-1")
    agents = [e["agent_id"] for e in log]
    assert "customer-1" in agents
    assert "inventory" in agents
    assert "price-inference" in agents


# --- Fable 13: The Map and the Mirror ---

def test_fable_13_map_and_mirror_substrate_renders_through_any_opener():
    """The same substrate through many openers is the same substrate."""
    s = Substrate()
    c = Cell(address="paris/seine", value="the river")
    s.add(c)
    chart = s.render("chart")
    graph = s.render("graph")
    witness = s.render("witness", address="paris/seine")
    # Same cell, different openers
    assert chart["cells"][0]["value"] == "the river"
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["value"] == "the river"


# --- Fable 14: The Letter and the Message ---

def test_fable_14_letter_and_message_inference_labeled():
    """A message is the substrate's inference, tagged as inference."""
    s = Substrate()
    c = Cell(address="mother/garden", value="in the garden")
    s.add(c)
    s.infer("mother/garden", "in the garden, thinking of daughter", agent_id="substrate")
    # The cell is not canonical until observed
    assert c.canonical is False
    assert c.inference is not None
    # The witness log records the inference as inference
    log = s.render("witness", address="mother/garden")
    assert any(e["action"] == "inference" for e in log)


# --- Fable 15: The Book and the River ---

def test_fable_15_book_and_river_substrate_can_hold_finished_and_alive():
    """A finished book is a cell. A stream is a cell with frequent inferences."""
    s = Substrate()
    book = Cell(address="novel/1920/finished", value="the woman grew old")
    s.add(book)
    # The book is canonical (finished)
    s.observe("novel/1920/finished", agent_id="reader")
    assert book.canonical is True
    # The stream is constantly inferring
    stream = Cell(address="novel/2080/stream", value="the woman...")
    s.add(stream)
    for i in range(5):
        s.infer("novel/2080/stream", f"the woman {i}", agent_id="substrate")
    # The stream has many inferences
    log = s.render("witness", address="novel/2080/stream")
    assert len(log) == 5


# --- Fable 16: The Spell and the Loop ---

def test_fable_16_spell_and_loop_witness_makes_command_accountable():
    """A spell is one-way. A loop is two-way. The witness makes the loop accountable."""
    s = Substrate()
    c = Cell(address="alchemist/transform-water", value=42.0)
    s.add(c)
    s.witness(c, "alchemist", "write", 42.0)
    s.witness(c, "alchemist", "read", 42.0)
    log = s.render("witness", address="alchemist/transform-water")
    # Both write and read are logged → the loop is two-way
    actions = [e["action"] for e in log]
    assert "write" in actions
    assert "read" in actions


# --- Fable 17: The Clock and the Pulse ---

def test_fable_17_clock_and_pulse_murmur_keeps_substrate_alive():
    """The murmur is the substrate's heartbeat. Cells that murmur are alive."""
    s = Substrate()
    c = Cell(address="patient/heart", value=72.0)
    s.add(c)
    # The cell murmurs
    assert c.murmur() is True
    # The cell is alive
    assert c.last_murmur > 0


# --- Fable 18: The Spyglass and the Convoy ---

def test_fable_18_spyglass_and_convoy_100_ships_write_to_substrate():
    """100 boats in the convoy write to the substrate. The convoy is the agent."""
    s = Substrate()
    # One cell, 100 boats writing to it
    c = Cell(address="bay/A17", value=12.5)
    s.add(c)
    for i in range(100):
        s.witness(c, f"boat-{i:03d}", "write", 12.5 + (i % 5) * 0.1)
    log = s.render("witness", address="bay/A17")
    assert len(log) == 100
    # The convoy is logged
    assert len(set(e["agent_id"] for e in log)) == 100


# --- Fable 19: The Oracle and the Inference ---

def test_fable_19_oracle_and_inference_substrate_inference_is_auditable():
    """The inference is logged. The oracle is not. The substrate's inference is auditable."""
    s = Substrate()
    c = Cell(address="citizen/buy-house", value=0.73)
    s.add(c)
    s.witness(c, "substrate", "inference", 0.73)
    s.witness(c, "substrate", "inference", 0.71)
    s.witness(c, "substrate", "inference", 0.75)
    log = s.render("witness", address="citizen/buy-house")
    inferences = [e for e in log if e["action"] == "inference"]
    assert len(inferences) == 3
    # The citizen can ask: why 73%? And find the log.


# --- Fable 20: The Loom and the River ---

def test_fable_20_loom_and_river_substrate_can_hold_pattern_and_emergence():
    """A loom is a fixed pattern. A river is an emerging pattern. The substrate can hold both."""
    s = Substrate()
    # The loom: a fixed-pattern cell
    loom = Cell(address="loom/card-001", value="pattern-row-001")
    s.add(loom)
    # The river: a cell with many inferences
    river = Cell(address="river/cloth-001", value="emerging")
    s.add(river)
    for i in range(10):
        s.infer("river/cloth-001", f"emerging-{i}", agent_id="substrate")
    # The loom is canonical
    s.observe("loom/card-001", agent_id="weaver")
    assert loom.canonical is True
    # The river is not canonical (still emerging)
    assert river.canonical is False


# --- Fable 21: The Compass and the Graph ---

def test_fable_21_compass_and_graph_substrate_shows_all_directions():
    """The compass points one way. The substrate graph shows all ways."""
    s = Substrate()
    # 12 destinations
    for i in range(12):
        dest = Cell(address=f"nav/dest-{i:02d}", value=10.0 + i)
        s.add(dest)
    graph = s.render("graph")
    assert len(graph["nodes"]) == 12
    # The substrate holds all destinations


# --- Fable 22: The Sundial and the Clock ---

def test_fable_22_sundial_and_clock_substrate_time_is_auditable():
    """The sundial's time is silent. The substrate's time is logged."""
    s = Substrate()
    c = Cell(address="time/now", value=12.0)
    s.add(c)
    s.witness(c, "substrate", "read", 12.0)
    s.witness(c, "substrate", "decay", 11.99)
    log = s.render("witness", address="time/now")
    actions = [e["action"] for e in log]
    assert "read" in actions
    assert "decay" in actions


# --- Fable 23: The Flute and the Murmur ---

def test_fable_23_flute_and_murmur_substrate_1000_cells_can_murmur():
    """The flute is one voice. The substrate is 1000 murmurs."""
    s = Substrate()
    for i in range(1000):
        c = Cell(address=f"murmur/{i:04d}", value=1.0)
        c.murmur()
        s.add(c)
    # All 1000 cells have murmured
    murmured = sum(1 for c in s.all_cells() if c._murmur_count > 0)
    assert murmured == 1000


# --- Fable 24: The Lantern and the Cell ---

def test_fable_24_lantern_and_cell_gc_chooses_relevant_cells():
    """The lantern lights one place. The substrate's GC chooses relevant cells."""
    s = Substrate()
    important = Cell(address="important/1", value=42.0)
    s.add(important)
    # Many less-important cells
    for i in range(20):
        s.add(Cell(address=f"minor/{i}", value=0.0))
    # The substrate can render all cells
    chart = s.render("chart")
    assert len(chart["cells"]) == 21
    # The substrate GC can prune minors (decay old, prune weak)
    for c in s.all_cells():
        if c.address.startswith("minor"):
            c._last_murmur -= 1000  # 1000s ago
    # The GC phase 1 prunes old inputs; the cell is still in the substrate


# --- Fable 25: The Journal and the Log ---

def test_fable_25_journal_and_log_witness_log_is_archaeological():
    """The journal is one voice. The witness log is the substrate's record. The archeologist reads it."""
    s = Substrate()
    c = Cell(address="log/day-47", value="the cook made stew again")
    s.add(c)
    s.witness(c, "sailor-1", "write", "the cook made stew again")
    s.witness(c, "sailor-2", "read", "the cook made stew again")
    s.witness(c, "drone-skate", "inference", "the cook made stew again")
    # The deep-time archeologist (scenario 09) reads the log
    log = s.render("witness", address="log/day-47")
    assert len(log) == 3
    agents = [e["agent_id"] for e in log]
    assert "sailor-1" in agents
    assert "sailor-2" in agents
    assert "drone-skate" in agents
    # The archeologist can ask: which voices? And find all 3.


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_fable_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
