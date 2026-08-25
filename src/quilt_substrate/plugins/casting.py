"""
quilt_substrate.plugins.casting — The Quilt casting-call plugin.

A learned casting layer that wraps `substrate.render(opener, **kwargs)` to
auto-select (model, opener, primitive) for a given (app, user, hardware, situation).

Phase 1 MVP: static priors + Bayesian decay from the casting-call atlas.
Phase 2: Wilson lower bound on observed outcomes.
Phase 3: LinUCB contextual bandits.
Phase 4: Full RL with collaborative filtering.

The witness log records every cast.proposed and cast.observed event.
The plugin learns from observed latency, success, and quality.

Author: Mavis / Casey DiGennaro
Date: 2026-08-24
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import math
import os
import platform
import socket
import subprocess
import time
from collections import defaultdict


# -- The Situation --------------------------------------------------------

@dataclass(frozen=True)
class Situation:
    """The state in which a casting decision is made."""
    user: str
    app: str
    hardware: str
    time_of_day: str = "day"     # "dawn" | "morning" | "afternoon" | "evening" | "night" | "0300"
    weather: str = "calm"         # "calm" | "windy" | "gale" | "storm"
    crew_state: str = "normal"    # "fresh" | "normal" | "tired" | "critical"

    def to_key(self) -> str:
        return f"{self.user}|{self.app}|{self.hardware}|{self.time_of_day}|{self.weather}|{self.crew_state}"

    def rationale_hash(self) -> str:
        """Short fingerprint of the situation — kept on the witness log."""
        h = hashlib.md5(self.to_key().encode()).hexdigest()[:8]
        return h


# -- The Resource Budget --------------------------------------------------

@dataclass
class ResourceBudget:
    """What the device has right now."""
    battery_pct: float = 1.0
    network: str = "wifi"          # "none" | "2g" | "3g" | "4g" | "wifi" | "ethernet"
    network_latency_ms: int = 50
    network_packet_loss: float = 0.0
    compute: str = "laptop"        # "embedded" | "laptop" | "workstation" | "cloud"
    compute_free_pct: float = 0.9
    storage_free_gb: float = 100.0

    @property
    def deadline_ms(self) -> int:
        """The user's likely wait time. Tighter under stress."""
        if self.network == "2g":
            return 2000  # 2s max
        if self.battery_pct < 0.2:
            return 3000
        if self.network == "wifi":
            return 500
        return 1000


# -- The Model Profile ----------------------------------------------------

@dataclass
class ModelProfile:
    """A model's static profile (the prior)."""
    name: str
    cost_per_1k: float
    is_local: bool
    min_battery: float
    strengths: List[str] = field(default_factory=list)
    failure_mode: str = ""


# The static atlas (Phase 1 priors — same as casting-call)
PRIOR_ATLAS: Dict[str, ModelProfile] = {
    "HERMES_405B":        ModelProfile("HERMES_405B", 0.0035, False, 0.3, ["narrator", "creative", "voice_narration"], "expensive"),
    "CLAUDE_OPUS":        ModelProfile("CLAUDE_OPUS", 0.015,  False, 0.5, ["heavy", "safety_check", "math_grief"],     "expensive, rate-limited"),
    "CLAUDE_SONNET":      ModelProfile("CLAUDE_SONNET", 0.003, False, 0.3, ["code_generation", "fallback"],              "mid-tier"),
    "QWEN3_CODER":        ModelProfile("QWEN3_CODER", 0.0005, False, 0.2, ["code_generation"],                            "code-specialist"),
    "QWEN3-MAX":          ModelProfile("QWEN3-MAX",   0.001,  False, 0.3, ["sensory_creative"],                            "sensory-direct"),
    "DEEPSEEK_V4_FLASH":  ModelProfile("DEEPSEEK_V4_FLASH", 0.0002, False, 0.2, ["sensory_creative", "voice_narration"], "fast, cheap"),
    "DEEPSEEK_V4_PRO":    ModelProfile("DEEPSEEK_V4_PRO", 0.001, False, 0.3, ["code_generation", "analysis"],          "deep"),
    "SEED_MINI":          ModelProfile("SEED_MINI",   0.0003, False, 0.2, ["creative_ideation", "fable_compression"],  "creative firehose"),
    "SEED_PRO":           ModelProfile("SEED_PRO",    0.002,  False, 0.3, ["creative_ideation"],                          "analog synth pro"),
    "NEMOTRON_ULTRA":     ModelProfile("NEMOTRON_ULTRA", 0.008, False, 0.5, ["safety_check"],                              "pipe organ, slow"),
    "GLM_5_2":            ModelProfile("GLM_5_2",     0.0006, False, 0.2, ["workhorse"],                                   "versatile"),
    "PHI-4":              ModelProfile("PHI-4",       0.0003, False, 0.2, ["math_grief", "fable_compression"],            "compact, deep"),
    "LING_FLASH":         ModelProfile("LING_FLASH",  0.0001, False, 0.2, ["fable_compression"],                          "naive questions"),
    "QWEN_0_5B":          ModelProfile("QWEN_0_5B",   0.0,    True,  0.1, ["fable_compression", "fallback"],              "local, tiny"),
    "GRANITE_3_1_2B":     ModelProfile("GRANITE_3_1_2B", 0.0, True, 0.1, ["voice_narration", "fallback"],              "local, Wesley"),
}


# Role → opener mapping (the dispatch table)
ROLE_TO_OPENER: Dict[str, List[Tuple[str, str, float]]] = {
    # role → [(model, opener, prior_score), ...]
    "voice_narration": [
        ("HERMES_405B", "voice", 0.9),
        ("CLAUDE_OPUS", "voice", 0.85),
        ("DEEPSEEK_V4_FLASH", "voice", 0.7),
    ],
    "code_generation": [
        ("QWEN3_CODER", "chart", 0.95),
        ("DEEPSEEK_V4_FLASH", "chart", 0.7),
        ("CLAUDE_SONNET", "chart", 0.65),
    ],
    "creative_ideation": [
        ("SEED_MINI", "slate", 0.95),
        ("HERMES_405B", "slate", 0.8),
        ("DEEPSEEK_V4_FLASH", "tide", 0.7),
    ],
    "safety_check": [
        ("NEMOTRON_ULTRA", "witness", 0.95),
        ("CLAUDE_OPUS", "witness", 0.85),
    ],
    "sensory_creative": [
        ("DEEPSEEK_V4_FLASH", "tide", 0.9),
        ("QWEN3-MAX", "reef", 0.85),
    ],
    "fable_compression": [
        ("SEED_MINI", "slate", 0.95),
        ("LING_FLASH", "slate", 0.9),
        ("PHI-4", "slate", 0.85),
    ],
    "math_grief": [
        ("PHI-4", "reef", 0.95),
        ("CLAUDE_OPUS", "reef", 0.85),
    ],
}


# -- The Casting Decision -------------------------------------------------

@dataclass
class CastingDecision:
    model: str
    opener: str
    primitive: str
    rationale: str
    confidence: float = 0.5
    prior_score: float = 0.5
    learned_score: float = 0.5
    is_fallback: bool = False


# -- Wilson Scoring (Phase 2) ---------------------------------------------

def wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound. n=0 → 0.0 (unbiased prior)."""
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - spread)


class WilsonProfiles:
    """Per-(primitive, opener, model) success tracking with optimistic prior."""

    def __init__(self, threshold: float = 0.6, window: int = 200, half_life_s: float = 3600.0):
        self.threshold = threshold
        self.window = window
        self.half_life = half_life_s
        # key → [(quality, ts, latency_ms, success)]
        self.obs: Dict[Tuple[str, str, str], List[Tuple[float, float, int, bool]]] = defaultdict(list)

    def observe(self, primitive: str, opener: str, model: str,
                 latency_ms: int, success: bool, quality: Optional[float] = None):
        key = (primitive, opener, model)
        q = quality if quality is not None else (1.0 if success else 0.0)
        self.obs[key].append((q, time.time(), latency_ms, success))
        # Trim to window
        self.obs[key] = self.obs[key][-self.window:]

    def lower_bound(self, primitive: str, opener: str, model: str) -> float:
        key = (primitive, opener, model)
        entries = self._decay(key)
        n = len(entries)
        if n < 3:
            return 0.5  # optimistic prior
        succ = sum(1 for q, _, _, s in entries if q >= self.threshold)
        return wilson_lower(succ, n)

    def latency_p90(self, primitive: str, opener: str, model: str) -> float:
        key = (primitive, opener, model)
        entries = self._decay(key)
        if not entries:
            return float('inf')
        lats = sorted(l for _, _, l, _ in entries)
        return lats[int(len(lats) * 0.9)]

    def rank(self, primitive: str, opener: str, candidates: List[str],
              budget_ms: Optional[int] = None) -> List[Tuple[float, str]]:
        scored = []
        for m in candidates:
            lb = self.lower_bound(primitive, opener, m)
            p90 = self.latency_p90(primitive, opener, m)
            if budget_ms and p90 > budget_ms * 2:
                continue  # hard exclude
            scored.append((lb, m))
        scored.sort(reverse=True)
        return scored

    def _decay(self, key) -> List[Tuple[float, float, int, bool]]:
        now = time.time()
        entries = self.obs.get(key, [])
        kept = [e for e in entries if (now - e[1]) < self.half_life * 4]
        self.obs[key] = kept
        return kept


# -- The Probes (real-world coupling) -------------------------------------

class Probes:
    """Snapshots the situation and resources.

    In a gale, all probes must return in <5ms or the hook becomes the bottleneck.
    """

    def __init__(self, user: str = "anonymous", app: str = "unknown",
                  hardware: str = "laptop", time_of_day: str = "day",
                  weather: str = "calm", crew_state: str = "normal"):
        self._user = user
        self._app = app
        self._hardware = hardware
        self._time_of_day = time_of_day
        self._weather = weather
        self._crew_state = crew_state
        self._battery_cache: Optional[Tuple[float, float]] = None
        self._battery_ttl = 30.0  # seconds

    def situation(self) -> Situation:
        return Situation(
            user=self._user,
            app=self._app,
            hardware=self._hardware,
            time_of_day=self._time_of_day,
            weather=self._weather,
            crew_state=self._crew_state,
        )

    def budget(self) -> ResourceBudget:
        return ResourceBudget(
            battery_pct=self._read_battery(),
            network=self._read_network(),
            network_latency_ms=self._read_latency(),
            network_packet_loss=self._read_packet_loss(),
            compute=self._read_compute(),
            compute_free_pct=self._read_cpu(),
            storage_free_gb=self._read_storage(),
        )

    def set_weather(self, weather: str):
        self._weather = weather

    def set_crew_state(self, state: str):
        self._crew_state = state

    def _read_battery(self) -> float:
        """Battery with caching (don't hammer /sys every call)."""
        now = time.time()
        if self._battery_cache and (now - self._battery_cache[1]) < self._battery_ttl:
            return self._battery_cache[0]
        # Try psutil first
        try:
            import psutil
            batt = psutil.sensors_battery()
            pct = (batt.percent / 100.0) if batt else 1.0
        except Exception:
            # Try /sys on Linux
            try:
                with open("/sys/class/power_supply/BAT0/capacity") as f:
                    pct = int(f.read().strip()) / 100.0
            except Exception:
                pct = 1.0  # unknown → assume full
        self._battery_cache = (pct, now)
        return pct

    def _read_network(self) -> str:
        try:
            socket.setdefaulttimeout(0.5)
            s = socket.socket()
            s.connect(("8.8.8.8", 53))
            s.close()
            return "wifi"
        except Exception:
            return "none"

    def _read_latency(self) -> int:
        try:
            t0 = time.monotonic()
            socket.setdefaulttimeout(1.0)
            s = socket.socket()
            s.connect(("8.8.8.8", 53))
            s.close()
            return int((time.monotonic() - t0) * 1000)
        except Exception:
            return 0

    def _read_packet_loss(self) -> float:
        # Cheap proxy: 0 if connected, 0.5 if uncertain
        return 0.0 if self._read_network() == "wifi" else 0.5

    def _read_compute(self) -> str:
        # Heuristic by platform
        sysname = platform.system()
        if sysname == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    cores = sum(1 for _ in f if "processor" in _) // 50
                if cores >= 16:
                    return "workstation"
                if cores >= 8:
                    return "laptop"
                return "embedded"
            except Exception:
                return "embedded"
        return "laptop"

    def _read_cpu(self) -> float:
        try:
            import psutil
            return 1.0 - (psutil.cpu_percent(interval=0.1) / 100.0)
        except Exception:
            return 0.9

    def _read_storage(self) -> float:
        try:
            import shutil
            _, _, free = shutil.disk_usage("/")
            return free / (1024 ** 3)
        except Exception:
            return 100.0


# -- The Plugin ----------------------------------------------------------

class QuiltCastingCallPlugin:
    """A learned casting-call that wraps substrate.render().

    Install on a substrate to auto-select (model, opener, primitive) per
    (app, user, hardware, situation) and record outcomes to the witness log.

    Usage:
        substrate = Substrate()
        plugin = QuiltCastingCallPlugin(substrate)
        plugin.install()
        # Now use the plugin's render() instead of substrate.render() directly
        result = plugin.render(opener="chart", **kwargs)
    """

    def __init__(self, substrate, probes: Optional[Probes] = None,
                  config: Optional[Dict[str, Any]] = None):
        self.substrate = substrate
        self.probes = probes or Probes()
        self.config = config or {}
        self.wilson = WilsonProfiles()
        self.witness: List[Dict[str, Any]] = []  # in-memory log; substrate has the real one
        self._max_invocations = 2  # hard ceiling

    def install(self):
        """Install the plugin. Wraps substrate.render with self.render."""
        self._original_render = self.substrate.render
        self.substrate.render = self.render
        return self

    def uninstall(self):
        """Restore the original render."""
        if hasattr(self, "_original_render"):
            self.substrate.render = self._original_render
        return self

    # -- The Cast Loop --

    def decide(self, opener: str, kwargs: Dict[str, Any]) -> CastingDecision:
        """Pick (model, opener, primitive) for the current situation."""
        sit = self.probes.situation()
        budget = self.probes.budget()
        return self._decide(opener, sit, budget, kwargs)

    def _decide(self, opener: str, sit: Situation, budget: ResourceBudget,
                  kwargs: Dict[str, Any]) -> CastingDecision:
        """The core decision function."""
        # 1. Detect the role from kwargs (or default to opener)
        role = kwargs.get("role") or self._infer_role(opener, kwargs)

        # 2. Get the candidate list from the prior atlas
        candidates = ROLE_TO_OPENER.get(role, [(self._fallback_model(), opener, 0.5)])

        # 3. Filter by resources
        valid = []
        for model, cand_opener, prior in candidates:
            if not self._can_afford(model, budget):
                continue
            valid.append((model, cand_opener, prior))

        if not valid:
            # Fallback: cheapest local
            return CastingDecision(
                model="QWEN_0_5B",
                opener=opener,
                primitive="echo",
                rationale="No candidates affordable; falling back to local echo",
                confidence=0.1,
                prior_score=0.0,
                is_fallback=True,
            )

        # 4. Rank by Wilson + prior blend
        primitive = kwargs.get("primitive", "Murmur")
        scored = []
        for model, cand_opener, prior in valid:
            wilson = self.wilson.lower_bound(primitive, cand_opener, model)
            blend = 0.5 * prior + 0.5 * wilson  # equal blend in Phase 2
            scored.append((blend, model, cand_opener, prior, wilson))
        scored.sort(reverse=True)
        best_score, best_model, best_opener, best_prior, best_wilson = scored[0]

        # 5. Apply situation overrides
        reason = f"role={role}, model={best_model}, opener={best_opener}"
        if sit.weather == "gale" and not PRIOR_ATLAS[best_model].is_local:
            reason += " (gale: local override)"
        if budget.battery_pct < 0.2:
            reason += f" (low-battery: {budget.battery_pct:.0%})"

        return CastingDecision(
            model=best_model,
            opener=best_opener,
            primitive=primitive,
            rationale=reason,
            confidence=best_score,
            prior_score=best_prior,
            learned_score=best_wilson,
        )

    def _can_afford(self, model: str, budget: ResourceBudget) -> bool:
        profile = PRIOR_ATLAS.get(model)
        if not profile:
            return False
        if profile.is_local:
            return budget.battery_pct > 0.1
        return (budget.network != "none"
                and budget.battery_pct >= profile.min_battery
                and budget.compute_free_pct > 0.05)

    def _fallback_model(self) -> str:
        return "QWEN_0_5B"

    def _infer_role(self, opener: str, kwargs: Dict[str, Any]) -> str:
        """Guess the role from the opener + kwargs."""
        # If a "role" kwarg was given, use it
        if "role" in kwargs:
            return kwargs["role"]
        # Heuristic: chart + cells = sensory_creative; voice = voice_narration
        if opener == "voice":
            return "voice_narration"
        if opener == "witness":
            return "safety_check"
        if opener == "reef":
            return "math_grief"
        if opener == "slate":
            return "fable_compression"
        return "creative_ideation"

    # -- The Render Hook (Phase 1 MVP) --

    def render(self, opener: str, **kwargs) -> Any:
        """The render_with_casting wrapper.

        Gathers situation + budget, decides, invokes substrate, records outcome.
        """
        # BEFORE
        sit = self.probes.situation()
        budget = self.probes.budget()
        decision = self._decide(opener, sit, budget, kwargs)

        # Witness: cast.proposed
        self._witness_proposed(decision, sit, budget, opener)

        # DURING — invoke the real substrate
        t0 = time.monotonic()
        success = True
        error: Optional[str] = None
        result: Any = None
        invocations = 0
        while invocations < self._max_invocations:
            invocations += 1
            try:
                result = self._original_render(decision.opener, **kwargs)
                success = True
                break
            except Exception as e:
                success = False
                error = hashlib.md5(str(e).encode()).hexdigest()[:8]
                self._witness_observed(decision, 0, success, error)
                # Try a fallback: cheaper model, same opener
                if decision.model != "QWEN_0_5B" and PRIOR_ATLAS[decision.model].is_local is False:
                    decision = self._retry_with_local(decision, sit, budget, kwargs)
                else:
                    # Already local; use echo
                    decision = self._retry_with_echo(decision, opener)
            else:
                break
        if invocations >= self._max_invocations and result is None:
            result = {"error": "max invocations exceeded", "fallback": True}

        latency_ms = int((time.monotonic() - t0) * 1000)

        # AFTER
        self._witness_observed(decision, latency_ms, success, error)
        self.wilson.observe(
            decision.primitive, decision.opener, decision.model,
            latency_ms, success,
        )

        return result

    def _retry_with_local(self, decision: CastingDecision, sit: Situation,
                            budget: ResourceBudget, kwargs: Dict[str, Any]) -> CastingDecision:
        """A retry that downgrades to a local model."""
        local_fallback = "QWEN_0_5B" if budget.battery_pct > 0.1 else None
        if local_fallback:
            return CastingDecision(
                model=local_fallback,
                opener=decision.opener,
                primitive=decision.primitive,
                rationale=f"Retry with local: {decision.model} failed",
                confidence=0.3,
                is_fallback=True,
            )
        return self._retry_with_echo(decision, decision.opener)

    def _retry_with_echo(self, decision: CastingDecision, opener: str) -> CastingDecision:
        """A retry that uses the 'echo' primitive (no model)."""
        return CastingDecision(
            model="none",
            opener="log",
            primitive="echo",
            rationale=f"Retry with echo: {decision.model} + {decision.opener} failed",
            confidence=0.1,
            is_fallback=True,
        )

    # -- Witness Log --

    def _witness_proposed(self, decision: CastingDecision, sit: Situation,
                            budget: ResourceBudget, opener: str):
        event = {
            "ts": time.time(),
            "kind": "cast.proposed",
            "decision": asdict(decision),
            "situation": asdict(sit),
            "situation_hash": sit.rationale_hash(),
            "resources": {
                "battery_pct": budget.battery_pct,
                "network": budget.network,
                "deadline_ms": budget.deadline_ms,
            },
        }
        self.witness.append(event)
        # Also try to write to substrate's witness if it has one
        if hasattr(self.substrate, "witness"):
            try:
                self.substrate.witness("cast.proposed", event)
            except Exception:
                pass

    def _witness_observed(self, decision: CastingDecision, latency_ms: int,
                           success: bool, error: Optional[str]):
        event = {
            "ts": time.time(),
            "kind": "cast.observed",
            "decision": asdict(decision),
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
        }
        self.witness.append(event)
        if hasattr(self.substrate, "witness"):
            try:
                self.substrate.witness("cast.observed", event)
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for inspection."""
        from collections import Counter
        proposed = [e for e in self.witness if e["kind"] == "cast.proposed"]
        observed = [e for e in self.witness if e["kind"] == "cast.observed"]
        models = Counter(e["decision"]["model"] for e in proposed)
        opacities = Counter(e["decision"]["opener"] for e in proposed)
        successes = sum(1 for e in observed if e["success"])
        total = len(observed)
        return {
            "n_proposed": len(proposed),
            "n_observed": len(observed),
            "success_rate": successes / total if total else 0.0,
            "models": dict(models),
            "openers": dict(opacities),
        }
