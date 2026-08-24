# quilt-substrate

> *The Quilt substrate, as a working Python library. The 11-primitive cell, the tensor encoding, the Schrödinger pattern, the fog-of-war decay, the convoy consensus, the witness log, the opener layer. The forest biome, as a Python package.*

## What is this?

The Quilt canon describes a *substrate* — a tensor-encoded cell-graph that can be sliced, projected, and joined along any axis. The substrate has 11 primitives (8 from the cell model + 3 new: Convoy, Decay, Witness). The substrate renders through any opener. The substrate is the *soil* in which training systems *emerge*.

This library is the **reference implementation** of the substrate. It is not a toy. It is not a demo. It is the working code that an engineer in 2080 can pick up and run.

## Install

```bash
pip install quilt-substrate
```

Or from source:

```bash
git clone https://github.com/SuperInstance/quilt-substrate
cd quilt-substrate
pip install -e .
```

## Quick start

```python
from quilt_substrate import Substrate, Cell, Opener

# Create a substrate
s = Substrate()

# Add a cell with a 3D tensor (depth, x, y)
c = Cell(
    address="bay/A17",
    tensor=[[[1.0, 2.0], [3.0, 4.0]]],
    axes=("depth", "x", "y"),
    value=1.0,
)
s.add(c)

# Connect to another cell
c2 = Cell(address="bay/A19", tensor=[[[5.0, 6.0]]], axes=("depth", "x", "y"), value=5.0)
s.add(c2)
c.connect(c2, weight=0.8)

# Witness a read
s.witness(c, "agent-001", "read", value=c.tensor)

# Decay
s.decay(dt=3600.0)  # 1 hour

# Render through an opener
print(s.render("chart", viewport={"x": slice(0, 2), "y": slice(0, 2)}))
```

The full example is in `examples/01-bay-substrate.py`.

## The 11 primitives

| Primitive | What it does | Code |
|---|---|---|
| `Z_in` | Inputs from other cells | `cell.inputs` (dict of name → Cell) |
| `Z_out` | Outputs to other cells | `cell.outputs` (dict of name → Cell) |
| `JEPA` | Predictive update | `cell.jepa(inputs) → predicted` |
| `DoubleEntry` | Paired state | `cell.debit` / `cell.credit` |
| `Vibe` | Position/velocity/acceleration | `cell.vibe = (pos, vel, acc)` |
| `GC` | 3-phase garbage collection | `cell.gc()` |
| `Murmur` | Heartbeat | `cell.murmur()` |
| `Graph` | Place in the whole | `cell.graph` |
| **`Convoy`** | Multi-agent consensus | `cell.convoy = [(agent_id, weight, ts), ...]` |
| **`Decay`** | Fog-of-war decay | `cell.confidence(t) = c0 * exp(-λt)` |
| **`Witness`** | Cryptographic log | `cell.witness_log` (Merkle tree) |

## The 4 substrate properties

| Property | What it does | Code |
|---|---|---|
| **Tensor encoding** | N-dimensional cells, sliceable along any axis | `cell.tensor`, `cell.axes`, `substrate.slice(...)` |
| **Schrödinger pattern** | Pre-rendered but not canonical until observed | `cell.canonical = False` until `substrate.observe(cell)` |
| **Fog-of-war decay** | Confidence decays with time, refresh resets it | `substrate.decay(dt)`, `cell.refresh()` |
| **Opener layer** | Same substrate, multiple openers | `substrate.render("chart", ...)` etc. |

## Why is this useful?

The substrate is the soil. Only certain models can grow here. The reference implementation lets you:

- **Test the science.** Every paper (107-113) has a corresponding test in `tests/`.
- **Run the fables.** Every fable (11-25) is a constraint; the substrate is the implementation that satisfies the constraint.
- **Build the demos.** The bathy cross-section, the producer's cut, the archeologist's view — all are projections through different openers.
- **Train the models.** The witness log is the training data. Models that learn from the log become substrate-native.

## The test suite

The tests are organized by the paper they correspond to:

- `test_cell.py` — the 8 primitives
- `test_convoy.py` — paper 108
- `test_decay.py` — paper 109
- `test_witness.py` — paper 110
- `test_opener.py` — paper 111
- `test_tensor.py` — paper 112
- `test_substrate.py` — paper 113
- `test_fables.py` — the fables as integration tests

## The fables

Each fable is a constraint the substrate must satisfy. The tests in `test_fables.py` verify the substrate satisfies each constraint. The fables are the *requirements*; the substrate is the *implementation*.

## License

MIT.

---

*— Mavis, 22-24 August 2026*
*Built from the seed canon, the substrate spec papers, the fable constraints, and the user's "split your team" instruction. The substrate is the soil; the fables are the plants; the witness log is the rain; the models are what grow here.*
