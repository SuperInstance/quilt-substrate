"""
quilt_substrate — The Quilt substrate as a working Python library.

The 11-primitive cell (8 from cell-runtime + 3 new: Convoy, Decay, Witness).
The tensor encoding. The Schrödinger pattern. The fog-of-war decay. The convoy
consensus. The witness log. The opener layer. The forest biome, as a package.

The cell is a system, not a value. The graph is the truth. The substrate is the soil.
"""
from .substrate import (
    Cell, Substrate, ConvoyEntry, DecayState, WitnessEntry, Vibe,
    _hash, _now, _now_ts,
)
from .openers import (
    Opener, ChartOpener, VoiceOpener, GestureOpener, WitnessOpener,
    MIDIOpener, RESTOpener, MUDOpener, PLATOOpener,
    SlateOpener, HarborOpener, ReefOpener, DiveOpener, TideOpener,
    BuoyOpener, TrawlOpener, ShoalOpener, MooringOpener, GaleOpener,
    register, get, all_openers,
)
from .jepa import (
    LinearJEPA, MLPJEPA, KnnJEPA, auto_train_jepa,
)

__all__ = [
    "Cell", "Substrate", "ConvoyEntry", "DecayState", "WitnessEntry", "Vibe",
    "Opener", "ChartOpener", "VoiceOpener", "GestureOpener", "WitnessOpener",
    "MIDIOpener", "RESTOpener", "MUDOpener", "PLATOOpener",
    "SlateOpener", "HarborOpener", "ReefOpener", "DiveOpener", "TideOpener",
    "BuoyOpener", "TrawlOpener", "ShoalOpener", "MooringOpener", "GaleOpener",
    "register", "get", "all_openers",
    "LinearJEPA", "MLPJEPA", "KnnJEPA", "auto_train_jepa",
    "_hash", "_now", "_now_ts",
]

__version__ = "0.2.1"
