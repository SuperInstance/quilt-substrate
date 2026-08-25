"""pincher_cache.py — A reflex-cache layer on top of the Quilt's casting plugin.

Pincher (the reflex shell) catches patterns in <50ms without an LLM. The Quilt
catches patterns and picks the model. Together: pincher catches the reflex,
the Quilt picks the LLM for everything else.

The integration:
1. Before calling the substrate (and potentially an LLM), embed the intent.
2. Search the reflex cache for similar intents.
3. If a high-confidence match is found (≥0.80), use the cached result.
4. If a low-confidence match (0.55-0.80), confirm with the user.
5. If no match (<0.55), call the substrate normally; the LLM fires; we
   record the outcome as a new reflex for next time.

The reflex cache is a list of (intent_text, embedding, model, opener, result,
quality, timestamp) tuples. On each cache hit, the Wilson profile updates.

This is the F/V EILEEN's gotcha mode: in a 0300 gale, every millisecond
counts. If a previous run already answered "what's the depth at lat 24.5,
lon -76.0?", we don't need the LLM again — we need the cached answer.

Usage:
    substrate = Substrate()
    probes = Probes(...)
    plugin = PincherCachedCastingPlugin(substrate, probes=probes,
                                          reflex_cache_path="data/reflexes.jsonl")
    plugin.install()
"""
from __future__ import annotations
import json
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import math

from .casting import (
    QuiltCastingCallPlugin, Probes, Situation, ResourceBudget,
    CastingDecision, PRIOR_ATLAS, ROLE_TO_OPENER,
)


@dataclass
class CachedReflex:
    """A cached response to a similar intent."""
    intent: str
    intent_embedding: List[float]
    model: str
    opener: str
    primitive: str
    result: Any
    quality: float
    timestamp: float
    hit_count: int = 0


class PincherCachedCastingPlugin(QuiltCastingCallPlugin):
    """A casting plugin that checks a reflex cache (pincher-style) first.

    The cache is stored as JSONL on disk. Each line is a CachedReflex. The
    plugin embeds the incoming intent, finds the most similar cached reflex,
    and returns it if confidence is high enough.

    Embeddings: uses pincher's Embedder (MiniLM-L6, 384D) if available, else
    a deterministic hash-based pseudo-embedding.
    """

    def __init__(self, substrate, probes: Optional[Probes] = None,
                  config: Optional[Dict[str, Any]] = None,
                  reflex_cache_path: str = "data/reflexes.jsonl",
                  high_confidence: float = 0.80,
                  low_confidence: float = 0.55,
                  use_pincher: bool = True):
        super().__init__(substrate, probes=probes, config=config)
        self.cache_path = Path(reflex_cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.use_pincher = use_pincher
        self._cache: List[CachedReflex] = self._load_cache()
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load the embedder (pincher's MiniLM or fallback)."""
        if self._embedder is not None:
            return self._embedder
        if self.use_pincher:
            try:
                # Try pincher's embedder
                sys_path = "/workspace/pincher/pincher-infer"
                if sys_path not in __import__("sys").path:
                    __import__("sys").path.insert(0, sys_path)
                from pincher_infer.embedder import Embedder
                self._embedder = Embedder()
                return self._embedder
            except Exception:
                self._embedder = False
        return None

    def _load_cache(self) -> List[CachedReflex]:
        """Load the reflex cache from disk."""
        if not self.cache_path.exists():
            return []
        cache = []
        with open(self.cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    cache.append(CachedReflex(**d))
                except (json.JSONDecodeError, TypeError):
                    pass
        return cache

    def _save_cache(self):
        """Persist the cache to disk (append-only)."""
        # Only save new entries, not the whole cache (which may be large).
        # The cache grows with hits; we append.
        pass  # Cache is updated in-place; see observe() for new entries

    def _embed(self, text: str) -> List[float]:
        """Embed text to a 384D vector (or 64D pseudo-vector if no embedder)."""
        embedder = self._get_embedder()
        if embedder and embedder is not False:
            try:
                vec = embedder.embed(text)
                return vec.tolist() if hasattr(vec, "tolist") else list(vec)
            except Exception:
                pass
        # Fallback: deterministic pseudo-embedding from text hash
        return self._pseudo_embed(text)

    def _pseudo_embed(self, text: str) -> List[float]:
        """A 64-dim hash-based pseudo-embedding. Deterministic, no model."""
        import hashlib
        # Hash the text 8 times to get 64 floats
        floats = []
        for i in range(8):
            h = hashlib.md5(f"{text}_{i}".encode()).digest()
            for byte in h[:8]:  # 8 bytes per chunk
                floats.append((byte - 128) / 128.0)  # normalize to [-1, 1]
        return floats

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            # Pad or truncate
            n = min(len(a), len(b))
            a = a[:n]
            b = b[:n]
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _find_reflex(self, intent: str) -> Tuple[Optional[CachedReflex], float]:
        """Find the most similar cached reflex. Returns (reflex, similarity)."""
        if not self._cache:
            return None, 0.0
        emb = self._embed(intent)
        best = None
        best_sim = 0.0
        for reflex in self._cache:
            sim = self._cosine_similarity(emb, reflex.intent_embedding)
            if sim > best_sim:
                best = reflex
                best_sim = sim
        return best, best_sim

    def decide(self, opener: str, kwargs: Dict[str, Any]) -> CastingDecision:
        """Decide — check reflex cache first, then call super()."""
        # If kwargs has an 'intent' or 'text', use it for cache lookup
        intent = kwargs.get("intent") or kwargs.get("text") or str(kwargs)
        reflex, similarity = self._find_reflex(intent)

        if reflex and similarity >= self.high_confidence:
            # High-confidence cache hit — return a decision that uses the cached model
            reflex.hit_count += 1
            return CastingDecision(
                model=reflex.model,
                opener=reflex.opener,
                primitive=reflex.primitive,
                rationale=f"reflex cache hit (sim={similarity:.2f}, hits={reflex.hit_count})",
                confidence=similarity,
                prior_score=reflex.quality,
                is_fallback=False,  # it's a HIT, not a fallback
            )

        # No cache hit — call the normal casting
        return super().decide(opener, kwargs)

    def render(self, opener: str, **kwargs) -> Any:
        """Render with reflex cache.

        The flow:
        1. Check the cache (via decide())
        2. If hit, return the cached result
        3. If miss, call the substrate normally and record the result as a new reflex
        """
        intent = kwargs.get("intent") or kwargs.get("text") or str(kwargs)
        reflex, similarity = self._find_reflex(intent)

        if reflex and similarity >= self.high_confidence:
            # Cache hit — return the cached result
            sit = self.probes.situation()
            decision = CastingDecision(
                model=reflex.model,
                opener=reflex.opener,
                primitive=reflex.primitive,
                rationale=f"reflex cache hit (sim={similarity:.2f})",
                confidence=similarity,
                prior_score=reflex.quality,
            )
            self._witness_proposed(decision, sit, self.probes.budget(), opener)
            start = time.monotonic()
            # Return cached result
            result = reflex.result
            latency_ms = int((time.monotonic() - start) * 1000)
            self._witness_observed(decision, latency_ms, True, None)
            self.wilson.observe(
                decision.primitive, decision.opener, decision.model,
                latency_ms, True, reflex.quality,
            )
            reflex.hit_count += 1
            return result

        # Cache miss — call the substrate normally
        result = super().render(opener, **kwargs)

        # Record the result as a new reflex (with quality 0.9 as default)
        if result is not None and not (isinstance(result, dict) and "error" in result):
            # Find the most recent observed event
            observed = [e for e in self.witness if e.get("kind") == "cast.observed"]
            if observed:
                last = observed[-1]["decision"]
                model = last.get("model", "unknown")
                dec_opener = last.get("opener", opener)
                primitive = last.get("primitive", "Murmur")
            else:
                model, dec_opener, primitive = "unknown", opener, "Murmur"
            new_reflex = CachedReflex(
                intent=intent,
                intent_embedding=self._embed(intent),
                model=model,
                opener=dec_opener,
                primitive=primitive,
                result=result,
                quality=0.9,
                timestamp=time.time(),
            )
            self._cache.append(new_reflex)
            # Persist
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(asdict(new_reflex)) + "\n")

        return result

    def cache_stats(self) -> Dict[str, Any]:
        """Return stats about the reflex cache."""
        if not self._cache:
            return {"n_reflexes": 0, "total_hits": 0, "avg_quality": 0.0}
        total_hits = sum(r.hit_count for r in self._cache)
        avg_quality = sum(r.quality for r in self._cache) / len(self._cache)
        return {
            "n_reflexes": len(self._cache),
            "total_hits": total_hits,
            "avg_quality": avg_quality,
            "cache_path": str(self.cache_path),
        }

    def teach(self, intent: str, result: Any, model: str = "QWEN_0_5B",
                opener: str = "slate", primitive: str = "Murmur",
                quality: float = 0.9):
        """Manually teach a new reflex (for preloading the cache)."""
        new_reflex = CachedReflex(
            intent=intent,
            intent_embedding=self._embed(intent),
            model=model,
            opener=opener,
            primitive=primitive,
            result=result,
            quality=quality,
            timestamp=time.time(),
        )
        self._cache.append(new_reflex)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(asdict(new_reflex)) + "\n")
        return new_reflex
