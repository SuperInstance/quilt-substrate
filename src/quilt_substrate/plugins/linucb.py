"""linucb.py — Phase 3 of the casting-call plugin: LinUCB contextual bandits.

Wilson gives us the (primitive, opener, model) score with a sample size. After
~10-20 observations, we have enough data to learn a LINEAR function of context
features. LinUCB does that.

Formula: score = w·x + alpha * sqrt(x^T A^-1 x)
- w is the learned weight vector (d-dim)
- A is the covariance matrix (d×d), init to identity
- b is the reward vector (d-dim), init to zero
- alpha controls exploration vs. exploitation

Context features:
- candidate identity: (model, opener, primitive) one-hot
- user (one-hot, learned per-user)
- time_of_day, weather, network, crew_state (one-hot)
- battery (continuous)

Warm-up: Wilson-only for n<10, linear blend 10<=n<20, LinUCB-only n>=20.

The plugin's existing decide() is unchanged; we add a `decide_linucb()` that
uses the bandit. The plugin tracks both Wilson and LinUCB in parallel and
blends the scores.

Why LinUCB? Wilson treats every (primitive, opener, model) triple as a single
bucket. LinUCB learns that at 0300 in a gale, certain triples are better than
others, AND that the same triple is good for some users but not others. The
linear function generalizes: if (calm, HERMES) works and (calm, SEED) works,
then (calm+windy, HERMES) probably works too.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .casting import (
    QuiltCastingCallPlugin, Probes, Situation, ResourceBudget,
    CastingDecision, PRIOR_ATLAS, ROLE_TO_OPENER,
)


# -- Feature taxonomy ------------------------------------------------------

TIME_OF_DAY = ["morning", "afternoon", "evening", "night", "0300"]
WEATHER = ["calm", "windy", "gale", "storm"]
NETWORK = ["none", "2g", "3g", "4g", "wifi", "ethernet"]
CREW_STATE = ["fresh", "normal", "tired", "critical"]
PRIMITIVES = ["Murmur", "Graph", "Vibe", "JEPA", "DoubleEntry", "Z_in", "Z_out",
              "GC", "Convoy", "Decay", "Witness"]
MODELS = list(PRIOR_ATLAS.keys())
OPENERS = ["chart", "voice", "gesture", "witness", "midi", "rest",
            "mud", "plato", "slate", "harbor", "reef", "dive", "tide"]


# -- Alpha (exploration budget) -------------------------------------------

def alpha_for(battery: float, weather: str, crew_state: str = "normal") -> float:
    """Compute alpha — the exploration budget for the current context.

    Higher alpha = more exploration. We shrink alpha when we can't afford to
    explore (low battery, gale, tired crew).
    """
    alpha = 1.0
    alpha *= 0.1 + 0.9 * battery
    weather_factor = {"calm": 1.0, "windy": 0.7, "gale": 0.3, "storm": 0.1}.get(weather, 1.0)
    alpha *= weather_factor
    crew_factor = {"fresh": 1.0, "normal": 1.0, "tired": 0.6, "critical": 0.2}.get(crew_state, 1.0)
    alpha *= crew_factor
    return float(np.clip(alpha, 0.05, 1.0))


# -- Feature extraction ---------------------------------------------------

class FeatureExtractor:
    """Builds a fixed-length feature vector for (candidate, situation)."""

    def __init__(self,
                  models: List[str] = None,
                  openers: List[str] = None,
                  primitives: List[str] = None,
                  users: List[str] = None):
        self.models = models or MODELS
        self.openers = openers or OPENERS
        self.primitives = primitives or PRIMITIVES
        self.users = users or ["casey", "reyes", "anonymous", "junior-dev"]
        # dimension = M + O + P + U + 5 + 4 + 1 + 6 + 4
        self.dim = (len(self.models) + len(self.openers) + len(self.primitives) + len(self.users)
                    + len(TIME_OF_DAY) + len(WEATHER) + 1 + len(NETWORK) + len(CREW_STATE))

    def __call__(self, candidate: Tuple[str, str, str],
                  situation: Situation) -> np.ndarray:
        """Return the feature vector for a (model, opener, primitive) candidate."""
        model, opener, primitive = candidate
        x = np.zeros(self.dim, dtype=np.float32)
        i = 0
        # candidate identity
        if model in self.models:
            x[i + self.models.index(model)] = 1.0
        i += len(self.models)
        if opener in self.openers:
            x[i + self.openers.index(opener)] = 1.0
        i += len(self.openers)
        if primitive in self.primitives:
            x[i + self.primitives.index(primitive)] = 1.0
        i += len(self.primitives)
        # user
        if situation.user in self.users:
            x[i + self.users.index(situation.user)] = 1.0
        i += len(self.users)
        # context
        if situation.time_of_day in TIME_OF_DAY:
            x[i + TIME_OF_DAY.index(situation.time_of_day)] = 1.0
        i += len(TIME_OF_DAY)
        if situation.weather in WEATHER:
            x[i + WEATHER.index(situation.weather)] = 1.0
        i += len(WEATHER)
        x[i] = 0.5  # battery (default mid); will be overridden if available
        i += 1
        if situation.network if hasattr(situation, 'network') else False:
            net = getattr(situation, 'network', 'wifi')
            if net in NETWORK:
                x[i + NETWORK.index(net)] = 1.0
        i += len(NETWORK)
        if situation.crew_state in CREW_STATE:
            x[i + CREW_STATE.index(situation.crew_state)] = 1.0
        i += len(CREW_STATE)
        return x

    def with_battery(self, x: np.ndarray, battery: float) -> np.ndarray:
        """Return a copy of x with the battery feature replaced."""
        result = x.copy()
        # The battery feature is at index: M + O + P + U + 5 + 4 = M + O + P + U + 9
        battery_idx = len(self.models) + len(self.openers) + len(self.primitives) + len(self.users) + 9
        result[battery_idx] = battery
        return result


# -- LinUCB model --------------------------------------------------------

class LinUCBModel:
    """One LinUCB model per (user, app) pair. Tracks A, b, and n."""

    def __init__(self, d: int, ridge: float = 1.0):
        self.d = d
        self.A = np.eye(d, dtype=np.float32) * ridge
        self.b = np.zeros(d, dtype=np.float32)
        self.n = 0

    @property
    def A_inv(self) -> np.ndarray:
        return np.linalg.inv(self.A)

    @property
    def w(self) -> np.ndarray:
        return self.A_inv @ self.b

    def predict(self, x: np.ndarray, alpha: float) -> Tuple[float, float]:
        """Return (expected, ucb_width) for a feature vector x."""
        A_inv = self.A_inv
        w = self.w
        expected = float(w @ x)
        ucb = alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
        return expected, ucb

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n += 1

    def score(self, x: np.ndarray, alpha: float) -> float:
        """UCB score: expected + alpha * confidence_width."""
        expected, ucb = self.predict(x, alpha)
        return expected + ucb


# -- The LinUCB-enhanced caster -------------------------------------------

class LinUCBCaster:
    """A LinUCB layer on top of an existing QuiltCastingCallPlugin.

    Wilson gives us n<10 stats. LinUCB gives us n>=10 contextual learning.
    We blend the two: 100% Wilson for n<10, linear blend 10<=n<20, 100% LinUCB for n>=20.
    """

    WARMUP_START = 10
    WARMUP_END = 20

    def __init__(self, plugin: QuiltCastingCallPlugin,
                  extractor: Optional[FeatureExtractor] = None):
        self.plugin = plugin
        self.extractor = extractor or FeatureExtractor()
        self.models: Dict[Tuple[str, str], LinUCBModel] = {}

    def _model_for(self, user: str, app: str) -> LinUCBModel:
        key = (user, app)
        if key not in self.models:
            self.models[key] = LinUCBModel(d=self.extractor.dim)
        return self.models[key]

    def _linucb_weight(self, n: int) -> float:
        """How much to weight LinUCB vs Wilson. 0.0 = pure Wilson, 1.0 = pure LinUCB."""
        if n < self.WARMUP_START:
            return 0.0
        if n >= self.WARMUP_END:
            return 1.0
        return (n - self.WARMUP_START) / (self.WARMUP_END - self.WARMUP_START)

    def rank(self, candidates: List[Tuple[str, str, str]],
              situation: Situation, budget: Optional[ResourceBudget] = None
              ) -> List[Tuple[float, Tuple[str, str, str]]]:
        """Rank (model, opener, primitive) candidates. Returns [(score, candidate), ...]."""
        battery = budget.battery_pct if budget else 0.5
        network = budget.network if budget else "wifi"
        alpha = alpha_for(battery, situation.weather, situation.crew_state)
        m = self._model_for(situation.user, situation.app)
        w_linucb = self._linucb_weight(m.n)
        results = []
        for cand in candidates:
            # Wilson score
            wilson_key = (cand[2], cand[1], cand[0])  # (primitive, opener, model)
            wilson_score = self.plugin.wilson.lower_bound(*wilson_key)
            # LinUCB score (sigmoid to [0,1])
            x = self.extractor(cand, situation)
            x = self.extractor.with_battery(x, battery)
            if w_linucb > 0:
                lin_score_raw = m.score(x, alpha)
                # Sigmoid to [0,1]
                lin_score = 1.0 / (1.0 + math.exp(-lin_score_raw))
            else:
                lin_score = wilson_score
            # Blend
            final = (1 - w_linucb) * wilson_score + w_linucb * lin_score
            results.append((final, cand))
        results.sort(key=lambda r: -r[0])
        return results

    def update(self, candidate: Tuple[str, str, str],
                situation: Situation, reward: float, budget: Optional[ResourceBudget] = None):
        """Record an observation: candidate (model, opener, primitive), reward in [0,1]."""
        battery = budget.battery_pct if budget else 0.5
        x = self.extractor(cand if (cand := candidate) else candidate, situation)
        x = self.extractor.with_battery(x, battery)
        self._model_for(situation.user, situation.app).update(x, reward)
        # Also feed the underlying Wilson profile
        wilson_key = (candidate[2], candidate[1], candidate[0])
        self.plugin.wilson.observe(
            wilson_key[0], wilson_key[1], wilson_key[2],
            0, reward > 0.5, reward,
        )

    def stats(self) -> Dict[str, Any]:
        """Return aggregate stats for the LinUCB layer."""
        return {
            "n_users": len(set(k[0] for k in self.models.keys())),
            "n_apps": len(set(k[1] for k in self.models.keys())),
            "n_models": len(self.models),
            "warmup_start": self.WARMUP_START,
            "warmup_end": self.WARMUP_END,
            "feature_dim": self.extractor.dim,
        }


# -- The enhanced plugin ------------------------------------------------

class LinUCBCastingPlugin(QuiltCastingCallPlugin):
    """A casting plugin that adds LinUCB contextual bandits to Wilson.

    Wilson handles n<10 (per-(primitive, opener, model) buckets).
    LinUCB handles n>=10 (per-(user, app) linear function over context).
    The blend is automatic: 100% Wilson for n<10, linear blend 10<=n<20,
    100% LinUCB for n>=20.

    The plugin tracks both layers in parallel. The existing decide() picks
    the candidates; the LinUCB layer ranks them.
    """

    def __init__(self, substrate, probes: Optional[Probes] = None,
                  config: Optional[Dict[str, Any]] = None):
        super().__init__(substrate, probes=probes, config=config)
        self.linucb = LinUCBCaster(self)
        # Track all observations for the LinUCB layer
        self._linucb_history: List[Dict[str, Any]] = []

    def decide(self, opener: str, kwargs: Dict[str, Any]) -> CastingDecision:
        """Decide using Wilson + LinUCB blend."""
        sit = self.probes.situation()
        budget = self.probes.budget()
        role = kwargs.get("role") or self._infer_role(opener, kwargs)
        # Get candidates from the prior
        candidates = ROLE_TO_OPENER.get(role, [(self._fallback_model(), opener, 0.5)])
        valid = []
        for model, cand_opener, prior in candidates:
            if not self._can_afford(model, budget):
                continue
            valid.append((model, cand_opener, role))  # (model, opener, primitive)
        if not valid:
            return CastingDecision(
                model="QWEN_0_5B", opener=opener, primitive="echo",
                rationale="no candidates affordable", confidence=0.1, is_fallback=True,
            )
        # Rank with LinUCB
        ranked = self.linucb.rank(valid, sit, budget)
        best_score, (best_model, best_opener, best_primitive) = ranked[0]
        return CastingDecision(
            model=best_model,
            opener=best_opener,
            primitive=best_primitive,
            rationale=f"Wilson+LinUCB: {best_model}+{best_opener} (score={best_score:.3f})",
            confidence=best_score,
            prior_score=PRIOR_ATLAS[best_model].strengths.__len__() / 10.0,
        )

    def render(self, opener: str, **kwargs) -> Any:
        """Render with LinUCB-enhanced decisions."""
        result = super().render(opener, **kwargs)
        # Record the outcome for the LinUCB layer
        sit = self.probes.situation()
        budget = self.probes.budget()
        if self.witness:
            observed = [e for e in self.witness if e.get("kind") == "cast.observed"]
            if observed:
                last = observed[-1]
                model = last["decision"]["model"]
                primitive = last["decision"]["primitive"]
                reward = last.get("quality", 0.5) or 0.5
                self.linucb.update((model, opener, primitive), sit, reward, budget)
                self._linucb_history.append({
                    "ts": last["ts"],
                    "model": model, "opener": opener, "primitive": primitive,
                    "reward": reward,
                    "user": sit.user, "app": sit.app,
                })
        return result

    def linucb_stats(self) -> Dict[str, Any]:
        return self.linucb.stats()
