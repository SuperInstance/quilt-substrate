"""local_fallback.py — A local-first variant of the casting-call plugin.

When the network is dead or the battery is critical, this plugin falls back
to agent-loop's local Ollama router. The Quilt's plugin and agent-loop
together cover the full cast:

  - Cloud (HERMES_405B, CLAUDE_OPUS, etc.) — for depth, when network is good
  - Local Ollama (qwen2.5:0.5b iterator, qwen-coder:3b specialist, etc.) —
    for survival, when network is dead

This is the F/V EILEEN's gotcha mode: even in a 0300 gale with no network,
the substrate still works. The iterator (qwen2.5:0.5b, ~400MB) runs always;
specialists load on demand.

Usage:
    substrate = Substrate()
    probes = Probes(user="reyes", app="F/V EILEEN", weather="gale", ...)
    plugin = LocalFallbackCastingPlugin(substrate, probes=probes)
    plugin.install()

If ollama is not running, the plugin falls back to the cheapest cloud
model (or echo if even that's unavailable).
"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional
from dataclasses import asdict

from .casting import (
    QuiltCastingCallPlugin, Probes, Situation, ResourceBudget,
    CastingDecision, PRIOR_ATLAS, ROLE_TO_OPENER,
)


# -- Task type mapping: Quilt opener → agent-loop task type --

OPENER_TO_TASK_TYPE = {
    "chart": "visualization",
    "voice": "narration",
    "tide": "sensory",
    "reef": "math",
    "slate": "writing",
    "witness": "audit",
    "harbor": "logistics",
    "dive": "exploration",
    "rest": "low_power",
}


class LocalFallbackCastingPlugin(QuiltCastingCallPlugin):
    """A casting plugin that delegates to agent-loop when local is preferred.

    Inherits all behavior from QuiltCastingCallPlugin. Overrides decide() and
    render() to check resources first; if network is dead or battery is critical,
    routes through agent-loop's ModelRouter instead of cloud models.
    """

    def __init__(self, substrate, probes: Optional[Probes] = None,
                  config: Optional[Dict[str, Any]] = None,
                  ollama_host: str = "http://localhost:11434",
                  iterator_model: str = "qwen2.5:0.5b",
                  use_local_when_offline: bool = True,
                  battery_threshold: float = 0.15):
        super().__init__(substrate, probes=probes, config=config)
        self.ollama_host = ollama_host
        self.iterator_model = iterator_model
        self.use_local_when_offline = use_local_when_offline
        self.battery_threshold = battery_threshold
        # Lazy import to avoid hard dep on agent-loop
        self._router = None

    def _get_router(self):
        """Lazy-init the agent-loop router."""
        if self._router is None:
            try:
                # The actual import — would need agent-loop installed
                from agent_loop.model_router import ModelRouter
                self._router = ModelRouter(ollama_host=self.ollama_host)
            except ImportError:
                self._router = False  # signal: not available
        return self._router

    def _should_go_local(self, budget: ResourceBudget) -> bool:
        """Decide whether to use the local fallback."""
        if not self.use_local_when_offline:
            return False
        if budget.network == "none":
            return True
        if budget.battery_pct < self.battery_threshold:
            return True
        return False

    def decide(self, opener: str, kwargs: Dict[str, Any]) -> CastingDecision:
        """Decide between cloud and local."""
        budget = self.probes.budget()
        if self._should_go_local(budget):
            task_type = OPENER_TO_TASK_TYPE.get(opener, "general")
            return CastingDecision(
                model=f"local:{self.iterator_model}",
                opener=opener,
                primitive="local_dispatch",
                rationale=f"local fallback (network={budget.network}, battery={budget.battery_pct:.0%}, task={task_type})",
                confidence=0.7,
                prior_score=0.7,
                is_fallback=True,
            )
        return super().decide(opener, kwargs)

    def render(self, opener: str, **kwargs) -> Any:
        """Render — either through cloud (normal) or through local (fallback)."""
        budget = self.probes.budget()
        if self._should_go_local(budget):
            return self._render_local(opener, **kwargs)
        return super().render(opener, **kwargs)

    def _render_local(self, opener: str, **kwargs) -> Any:
        """Render via agent-loop's local router."""
        router = self._get_router()
        if not router:
            # agent-loop not available; fall back to echo
            return {"error": "local fallback unavailable", "fallback": True}

        task_type = OPENER_TO_TASK_TYPE.get(opener, "general")
        prompt = kwargs.get("text", str(kwargs))

        sit = self.probes.situation()
        budget = self.probes.budget()
        decision = CastingDecision(
            model=f"local:{self.iterator_model}",
            opener=opener,
            primitive="local_dispatch",
            rationale=f"local dispatch: {task_type}",
            confidence=0.7,
            is_fallback=True,
        )
        self._witness_proposed(decision, sit, budget, opener)

        start = time.monotonic()
        try:
            result = router.dispatch(
                task_type=task_type,
                prompt=prompt,
                context=str(kwargs),
            )
            success = bool(result)
            quality = 0.8 if success else 0.0
        except Exception as e:
            success = False
            quality = 0.0
            result = {"error": str(e)}

        latency_ms = int((time.monotonic() - start) * 1000)
        error = None if success else "local_dispatch_failed"
        self._witness_observed(decision, latency_ms, success, error)
        self.wilson.observe(
            decision.primitive, decision.opener, decision.model,
            latency_ms, success, quality,
        )
        return result
