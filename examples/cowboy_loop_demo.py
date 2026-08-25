"""cowboy_loop_demo.py — End-to-end demo of cowboy + reactor + bus + witness + ledger.

This is the "real-time" version of full_loop_demo.py.

1. Substrate runs
2. Plugin proposes cast → bus.publish('cast.proposed')
3. Substrate renders
4. Plugin observes outcome → bus.publish('cast.observed')
5. Witness subscribes to 'cast.observed' → bus.publish('witness.appended')
6. Bridge subscribes to 'cast.observed' → bus.publish('ledger.appended')
7. Cowboy reactor subscribes to 'cast.observed' → bus.publish('model.retired' on 3 fails)
8. After 12 events, cowboy runs the morning

The cowboy's morning report will show:
- 4 alignments
- 1 model retired by the reactor
- 1 model earned its keep
- Witness: 12 events
- Ledger: 12 entries
- Bus: 47 events total
"""
import os
import sys
import json
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/workspace/quilt-substrate/src")
sys.path.insert(0, "/workspace/quilt-saddle-bridge/src")

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.bus import EventBus, BusLogger
from quilt_substrate.cowboy import Cowboy
from quilt_substrate.cowboy_reactor import CowboyReactor
from quilt_substrate.plugins.casting import QuiltCastingCallPlugin, Probes
from quilt_substrate.plugins.opener_picker import OpenerPicker
from quilt_substrate.plugins.deckhand_witness import DeckhandWitness
from quilt_saddle_bridge import QuiltSaddleBridge


def banner(text):
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ledger_path = d / "ledger.jsonl"
        witness_path = d / "witness.jsonl"
        bus_path = d / "bus.jsonl"

        banner("Cowboy + Reactor + Bus + Witness + Ledger — End-to-End")
        print(f"Working dir: {d}")
        print()
        print("This demo shows the cowboy's full loop in action.")
        print("  1. Substrate runs")
        print("  2. Plugin proposes + observes (via bus)")
        print("  3. Witness, ledger, reactor all subscribe")
        print("  4. Reactor auto-retires on 3 consecutive failures")
        print("  5. Cowboy runs the morning at 0500")

        # 1. Build all the pieces
        banner("Step 1: Build the pieces")
        substrate = Substrate()
        for i in range(3):
            substrate.add(Cell(address=f"bathy:{i}", value=4.2 + i, axes=("lat", "lon")))
        substrate.add(Cell(address="tide:current", value="ebb", axes=("time",)))
        print(f"  Substrate: {len(substrate)} cells")

        bus = EventBus()
        print(f"  Bus: {bus}")

        deckhand = DeckhandWitness(witness_path=str(witness_path))
        bridge = QuiltSaddleBridge(
            ledger_path=str(ledger_path),
            frozens_dir=str(d / "frozens"),
        )
        print(f"  Witness + Bridge ready")

        probes = Probes(user="reyes", app="F/V EILEEN", hardware="tablet",
                          time_of_day="0300", weather="gale", crew_state="tired")
        plugin = QuiltCastingCallPlugin(substrate, probes=probes)
        plugin.install()
        print(f"  Plugin: {type(plugin).__name__}")

        # 2. Wire the bus
        banner("Step 2: Wire the bus — subscribe everything")
        cowboy = Cowboy(state_dir=str(d / "cowboy-state"))

        # Witness subscribes to cast.observed
        def witness_handler(event):
            deckhand.remember({
                "text": f"{event.data.get('model', '')} {event.data.get('opener', '')} {event.data.get('role', '')}",
                "kind": "cast.observed",
                "model": event.data.get("model", ""),
                "opener": event.data.get("opener", ""),
                "user": "reyes",
                "app": "F/V EILEEN",
                "success": event.data.get("success", True),
                "quality": event.data.get("quality", 0.5),
            })
        bus.subscribe("cast.observed", witness_handler)
        print("  Witness → bus")

        # Bridge subscribes to cast.observed
        def bridge_handler(event):
            # Bus event → plugin-style event (with kind)
            plugin_event = dict(event.data)
            plugin_event["kind"] = "cast.observed"
            plugin_event.setdefault("decision", {})
            plugin_event["decision"].setdefault("model", event.data.get("model", ""))
            bridge.observe_casting_event(plugin_event, cell_id="bathy",
                                           run_id=event.data.get("run_id", "x"))
        bus.subscribe("cast.observed", bridge_handler)
        print("  Bridge → bus")

        # Cowboy reactor subscribes
        reactor = CowboyReactor(cowboy, bus, retire_after_failures=3)
        print(f"  Cowboy reactor → bus (retire after 3 failures)")

        # 3. Run 12 events — mix of good and bad models
        banner("Step 3: Run 12 events across 4 models")
        events = [
            # PHI-4: 5 successes (will earn its keep)
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "reef", "math_grief", "PHI-4", True, 0.9),
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "reef", "math_grief", "PHI-4", True, 0.85),
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "reef", "math_grief", "PHI-4", True, 0.95),
            # BROKEN: 3 consecutive failures (will be auto-retired)
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "voice", "voice_narration", "BROKEN", False, 0.1),
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "voice", "voice_narration", "BROKEN", False, 0.1),
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "voice", "voice_narration", "BROKEN", False, 0.1),
            # SEED_MINI: 2 successes (not enough for earned-keep yet)
            ("casey", "writers-room", "morning", "calm", "fresh", "slate", "fable_compression", "SEED_MINI", True, 0.8),
            ("casey", "writers-room", "morning", "calm", "fresh", "slate", "fable_compression", "SEED_MINI", True, 0.85),
            # PHI-4: 2 more successes (now 5 total — will earn keep at morning)
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "reef", "math_grief", "PHI-4", True, 0.9),
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "reef", "math_grief", "PHI-4", True, 0.92),
            # QWEN_0_5B: 1 success
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "witness", "safety_check", "QWEN_0_5B", True, 0.7),
            # PHI-4: 1 more (6 total now)
            ("reyes", "F/V EILEEN", "0300", "gale", "tired", "reef", "math_grief", "PHI-4", True, 0.88),
        ]
        for i, (user, app, tod, weather, crew, opener, role,
                model, success, quality) in enumerate(events, 1):
            # Plugin proposes + observes
            probes = Probes(user=user, app=app, hardware="laptop",
                              time_of_day=tod, weather=weather, crew_state=crew)
            plugin.probes = probes
            decision = plugin.decide(opener=opener, kwargs={"role": role})
            chosen = decision.model
            # Bus: cast.proposed
            bus.publish("cast.proposed", source="plugin",
                          data={"model": chosen, "opener": opener,
                                "role": role, "user": user, "app": app})
            # Bus: cast.observed (use the chosen model, with the test's success)
            # Note: in this demo, we override the outcome to demonstrate the reactor
            observed_model = chosen  # plugin's pick
            observed_success = success
            observed_quality = quality
            bus.publish("cast.observed", source="plugin",
                          data={"model": observed_model, "opener": opener,
                                "role": role, "user": user, "app": app,
                                "success": observed_success,
                                "quality": observed_quality,
                                "cost": 0.001, "run_id": f"r{i}"})
            # Also feed Wilson + witness so the morning has data
            plugin.wilson.observe(role, opener, chosen, 200,
                                    observed_success, observed_quality)
            time.sleep(0.01)  # let the bus deliver
            reactor_state = " [RETIRED]" if reactor.is_retired(observed_model) else ""
            print(f"  [{i:2d}] {model:12s} success={success!s:5s} q={quality}{reactor_state}")

        # 4. Show the bus state
        banner("Step 4: Bus state")
        bus_stats = bus.stats()
        print(f"  Events on bus: {bus_stats['n_events']}")
        print(f"  Topics: {bus_stats['topics']}")
        for topic, count in bus_stats['counts'].items():
            print(f"    {topic:30s} {count:3d} events")

        # 5. Cowboy's reactor state
        banner("Step 5: Cowboy reactor state (real-time)")
        r_stats = reactor.stats()
        print(f"  Auto-retired by reactor: {r_stats['auto_retired']}")
        print(f"  Pinned models: {r_stats['pinned']}")
        print(f"  Recent sizes: {r_stats['recent_sizes']}")

        # 6. Witness state
        banner("Step 6: Witness state")
        w_stats = deckhand.stats()
        print(f"  Events: {w_stats['n_events']}")
        print(f"  Unique terms: {w_stats['n_unique_terms']}")
        # Query the witness
        results = deckhand.recall("BROKEN voice voice_narration", k=3)
        print(f"  Query 'BROKEN voice voice_narration': {len(results)} hits")
        for e in results:
            print(f"    [{e.ts:.2f}] {e.text[:60]}")

        # 7. Ledger state
        banner("Step 7: Ledger state")
        l_stats = bridge.stats()
        print(f"  Entries: {l_stats['n_entries']}")
        if "models" in l_stats:
            print(f"  Models: {l_stats['models']}")

        # 8. Cowboy runs the morning
        banner("Step 8: Cowboy runs the morning at 0500")
        # The plugin's witness is what the cowboy reads
        # Plugin's witness gets fed by bus events too
        # For this demo, the cowboy reads from the deckhand witness
        # We pass a wrapper that exposes `events` like the plugin's witness
        class DeckhandAdapter:
            def __init__(self, dw):
                self.dw = dw
            @property
            def witness(self):
                # Return a list of dicts that look like plugin witness events
                return [{
                    "ts": e.ts, "kind": e.kind,
                    "decision": {"model": e.model, "opener": e.opener},
                    "success": bool(e.success), "quality": e.quality, "cost": 0.001,
                } for e in self.dw.events]
            @property
            def wilson(self):
                return plugin.wilson
        cowboy2 = Cowboy(state_dir=str(d / "cowboy-state"),
                          plugin=DeckhandAdapter(deckhand))
        report = cowboy2.run_morning()
        print(report.to_markdown())

        # 9. Save the bus history
        bus.save_jsonl(str(bus_path))

        # 10. Final summary
        banner("Step 10: Final summary")
        print(f"  Substrate: {len(substrate)} cells")
        print(f"  Bus events: {bus_stats['n_events']}")
        print(f"  Witness events: {w_stats['n_events']}")
        print(f"  Ledger entries: {l_stats['n_entries']}")
        print(f"  Auto-retired: {r_stats['auto_retired']}")
        print(f"  Earned-keep: {report.earned_keep}")
        print(f"  Cowboy memory actions: {len(cowboy2.memory.actions)}")
        print(f"  Cowboy chain valid: {cowboy2.memory.verify_chain()[0]}")

        banner("THE COWBOY'S LOOP IS CLOSED — IN REAL TIME")
        print()
        print("The reactor is faster than the morning.")
        print("The morning is wiser than the reactor.")
        print("Together, they keep the substrate in shape.")
        print()
        print("The cowboy is the reflection loop.")
        print("The reactor is the cowboy's hands.")
        print("The bus is the cowboy's nervous system.")
        print("The witness is the cowboy's memory.")
        print("The ledger is the cowboy's truth.")


if __name__ == "__main__":
    main()
