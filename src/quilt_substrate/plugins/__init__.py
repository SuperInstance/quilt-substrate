"""quilt_substrate.plugins — Plug-in layers for the substrate.

Each plugin wraps substrate.render() with additional behavior. The casting-call
plugin auto-selects (model, opener, primitive) for a given situation. Future
plugins will add: a11y, observability, cache, retry, billing.
"""
from .casting import (
    QuiltCastingCallPlugin,
    Probes,
    Situation,
    ResourceBudget,
    CastingDecision,
    WilsonProfiles,
    wilson_lower,
    PRIOR_ATLAS,
    ROLE_TO_OPENER,
)

__all__ = [
    "QuiltCastingCallPlugin",
    "Probes",
    "Situation",
    "ResourceBudget",
    "CastingDecision",
    "WilsonProfiles",
    "wilson_lower",
    "PRIOR_ATLAS",
    "ROLE_TO_OPENER",
]
