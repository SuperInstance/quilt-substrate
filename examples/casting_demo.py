"""casting_demo.py — An end-to-end demo of the Quilt casting-call plugin.

This shows the plugin wiring into a real substrate, recording events,
updating Wilson profiles, and adapting to gale vs. calm.

Run:
    cd /workspace/quilt-substrate
    python3 examples/casting_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quilt_substrate.substrate import Substrate, Cell
from quilt_substrate.plugins.casting import QuiltCastingCallPlugin, Probes


def make_substrate_with_cells():
    """A substrate with a few cells for the chart opener."""
    s = Substrate()
    s.add(Cell(address="chart:0", value=42, axes=("x", "y")))
    s.add(Cell(address="chart:1", value=84, axes=("x", "y")))
    s.add(Cell(address="chart:2", value=21, axes=("x", "y")))
    s.add(Cell(address="bathy:0", value=4.2, axes=("lat", "lon")))
    s.add(Cell(address="bathy:1", value=5.8, axes=("lat", "lon")))
    s.add(Cell(address="tide:current", value="ebb", axes=("time",)))
    return s


def banner(text):
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    banner("Quilt casting-call plugin — End-to-end demo")
    print("A 12-inch tablet on the F/V EILEEN, in a 0300 gale.")
    print()
    print("Before the plugin: the substrate's render() is direct.")
    print("After the plugin: render() is wrapped, decisions are recorded,")
    print("Wilson profiles update, the next render is informed.")

    # Step 1: Make a substrate
    substrate = make_substrate_with_cells()

    # Step 2: Make a probe for the F/V EILEEN in a gale
    probes = Probes(
        user="reyes",
        app="F/V EILEEN navigation",
        hardware="ruggedized-tablet",
        time_of_day="0300",
        weather="gale",
        crew_state="tired",
    )

    # Step 3: Install the plugin
    plugin = QuiltCastingCallPlugin(substrate, probes=probes)
    plugin.install()

    # Step 4: Render through the plugin
    banner("Step 4: First render (gale, 0300, 30% battery)")
    print("App calls substrate.render(opener='tide', role='sensory_creative')")
    print()
    result = substrate.render(opener="tide", role="sensory_creative")
    print(f"Result: {type(result).__name__}, {len(str(result))} chars")
    stats = plugin.stats()
    print(f"Stats: {stats['n_proposed']} proposed, {stats['n_observed']} observed")
    print(f"Latest decision: {plugin.witness[-2]['decision']['model']} + {plugin.witness[-2]['decision']['opener']}")
    print(f"  Rationale: {plugin.witness[-2]['decision']['rationale']}")

    # Step 5: Render more — Wilson should start to update
    banner("Step 5: 9 more renders — Wilson profiles kick in (n=3)")
    for i in range(9):
        substrate.render(opener="tide", role="sensory_creative")
    print(f"After 10 renders, Wilson profiles have {len(plugin.wilson.obs)} keys")
    for k, entries in plugin.wilson.obs.items():
        if len(entries) >= 3:
            print(f"  {k}: {len(entries)} obs, lower_bound={plugin.wilson.lower_bound(*k):.3f}, p90={plugin.wilson.latency_p90(*k):.0f}ms")

    # Step 6: Switch to calm waters
    banner("Step 6: Calmer waters (5am, wind dropped)")
    probes.set_weather("calm")
    probes.set_crew_state("fresh")
    for i in range(5):
        result = substrate.render(opener="slate", role="fable_compression")
    print(f"After 5 calm renders, stats: {plugin.stats()}")
    last = plugin.witness[-2]
    print(f"Latest model: {last['decision']['model']} (cost: ${QuiltCastingCallPlugin._get_cost(last['decision']['model']):.4f}/1k)")

    # Step 7: Drop the battery
    banner("Step 7: Battery drops to 10% — local models only")
    import unittest.mock as mock
    with mock.patch.object(probes, "_read_battery", return_value=0.10):
        for i in range(3):
            substrate.render(opener="tide", role="sensory_creative")
    last = plugin.witness[-2]
    print(f"At 10% battery, model: {last['decision']['model']}")
    print(f"  Rationale: {last['decision']['rationale']}")
    print(f"  is_fallback: {last['decision']['is_fallback']}")

    # Step 8: Stats summary
    banner("Step 8: Final stats")
    print(plugin.stats())

    # Step 9: Uninstall
    banner("Step 9: Uninstall — substrate.render is back to normal")
    plugin.uninstall()
    result = substrate.render(opener="chart")
    print(f"After uninstall, render() works: {type(result).__name__}, {len(result.get('cells', []))} cells")


def main_with_helper():
    """Main entry point with a small cost helper."""
    # The plugin doesn't have _get_cost as a method; let's use the static one
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from quilt_substrate.plugins.casting import PRIOR_ATLAS
    QuiltCastingCallPlugin._get_cost = staticmethod(lambda m: PRIOR_ATLAS[m].cost_per_1k if m in PRIOR_ATLAS else 0.0)
    main()


if __name__ == "__main__":
    main_with_helper()
