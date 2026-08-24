"""
01-the-bay-substrate.py — The bathy substrate. The Inner Sound. 100 boats.

This example implements scenario 03 (The Convoy) in the substrate.
A small substrate of 5 cells representing the bay. Each cell has a tensor
encoding (depth, x, y). The convoy records which boats have written to
each cell. The fog-of-war decay reduces confidence over time. The
witness log records every read/write/inference.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from substrate import Cell, Substrate


def main():
    # Create a substrate of 5 cells in a small bay
    s = Substrate()

    cells = [
        Cell(address="bay/A17", value=12.5,
             tensor=[[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]],
             axes=("depth", "x", "y")),
        Cell(address="bay/A18", value=11.2,
             tensor=[[[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]],
             axes=("depth", "x", "y")),
        Cell(address="bay/A19", value=15.0,
             tensor=[[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]],
             axes=("depth", "x", "y")),
        Cell(address="bay/B22", value=8.0,
             tensor=[[[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]],
             axes=("depth", "x", "y")),
        Cell(address="bay/C31", value=20.0,
             tensor=[[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]]],
             axes=("depth", "x", "y")),
    ]
    for c in cells:
        s.add(c)

    # Three boats in the convoy: Reyes, Skate, and the Inference
    boats = ["reyes", "skate", "inference"]

    # The boats read and write to the cells
    for c in cells:
        for boat in boats:
            s.witness(c, boat, "read", c.value)
            s.witness(c, boat, "inference", c.value + 0.1)

    # One cell gets a fresh write from Reyes (the convoy's most recent observation)
    s.refresh("bay/A17")

    # Render through different openers
    print("=== Chart opener ===")
    chart = s.render("chart")
    for entry in chart["cells"][:3]:
        print(f"  {entry['address']}: value={entry['value']}, confidence={entry['confidence']:.3f}, canonical={entry['canonical']}")

    print()
    print("=== Witness log for bay/A17 (the Reyes cell) ===")
    log = s.render("witness", address="bay/A17")
    for entry in log:
        print(f"  {entry['agent_id']:>10s} | {entry['action']:>10s} | hash={entry['value_hash']}")

    print()
    print("=== Convoy for bay/A17 ===")
    convoy = s.render("convoy", address="bay/A17")
    print(f"  bay/A17 has {len(convoy)} convoy entries")
    for entry in convoy:
        print(f"    {entry['agent_id']:>10s} | weight={entry['weight']} | last_write={entry['last_write']:.1f}")

    print()
    print("=== Tensor slice: depth=0 layer of bay/A17 ===")
    surface = s.render("tensor", address="bay/A17")
    if surface is not None and len(surface) > 0:
        print(f"  Shape: {len(surface[0])}x{len(surface[0][0]) if len(surface[0]) > 0 else 0}")
        print(f"  Values: {surface[0]}")

    print()
    print("=== Graph (the substrate as a graph) ===")
    graph = s.render("graph")
    print(f"  {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")


if __name__ == "__main__":
    main()
