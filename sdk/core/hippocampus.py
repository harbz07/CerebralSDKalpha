from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional
from .types import MemoryEvent

CATCH_22_THRESHOLD = 22
DEFAULT_CLASS_BIAS: Dict[str, float] = {"Chaos": 0.0, "Foundation": 2.0, "Glow": 1.0}

class Hippocampus:
    """Vector LTM with Catch-22 gate and cosine recall."""
    def __init__(self, similarity_threshold: float = 0.80, top_k: int = 5, class_bias: Dict[str, float] | None = None):
        self.sim_threshold = similarity_threshold
        self.top_k = top_k
        self.bias = class_bias or dict(DEFAULT_CLASS_BIAS)
        self._events: List[MemoryEvent] = []
        self._norms: List[float] = []

    def _total_score(self, ev: MemoryEvent) -> float:
        return float(sum(ev.scores.values()) + self.bias.get(ev.event_class, 0.0))

    def consolidate(self, event: MemoryEvent) -> bool:
        """Store event if sum(scores)+bias ≥ 22 or 'pin' tag present. Requires embedding."""
        if "pin" in event.tags or self._total_score(event) >= CATCH_22_THRESHOLD:
            if event.embedding is None:
                raise ValueError("Hippocampus.consolidate: embedding required")
            vec = np.asarray(event.embedding, dtype=np.float32)
            self._events.append(event)
            self._norms.append(float(np.linalg.norm(vec)))
            return True
        return False

    def recall(self, query_embedding: List[float] | np.ndarray, top_k: Optional[int] = None) -> List[MemoryEvent]:
        if not self._events:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-8
        sims = []
        for ev, n in zip(self._events, self._norms):
            v = np.asarray(ev.embedding, dtype=np.float32)  # type: ignore
            sims.append(float(np.dot(v, q) / (n * qn)))
        k = top_k or self.top_k
        ranked = sorted(zip(sims, self._events), key=lambda t: t[0], reverse=True)
        return [ev for sim, ev in ranked[:k] if sim >= self.sim_threshold]

    def stats(self) -> Dict[str, int]:
        dims = 0
        if self._events and self._events[0].embedding is not None:
            dims = int(np.asarray(self._events[0].embedding).size)
        return {"count": len(self._events), "dims": dims}
