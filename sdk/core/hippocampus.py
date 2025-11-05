from __future__ import annotations
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Callable
from .types import MemoryEvent

CATCH_22_THRESHOLD = 22
DEFAULT_CLASS_BIAS: Dict[str, float] = {"Chaos": 0.0, "Foundation": 2.0, "Glow": 1.0}

class Hippocampus:
    """Vector LTM with Catch-22 gate and cosine recall.

    Enhanced with:
    - Temporal decay for aging memories
    - Batch consolidation
    - Forgetting mechanisms
    - Importance-weighted recall
    - Tag-based filtering
    """
    def __init__(
        self,
        similarity_threshold: float = 0.80,
        top_k: int = 5,
        class_bias: Dict[str, float] | None = None,
        temporal_decay: bool = False,
        decay_halflife: Optional[timedelta] = None
    ):
        """Initialize Hippocampus.

        Args:
            similarity_threshold: Minimum cosine similarity for recall
            top_k: Default number of results to return
            class_bias: Additive bias for event classes
            temporal_decay: Whether to apply temporal decay during recall
            decay_halflife: Half-life for temporal decay (None = 30 days default)
        """
        self.sim_threshold = similarity_threshold
        self.top_k = top_k
        self.bias = class_bias or dict(DEFAULT_CLASS_BIAS)
        self.temporal_decay = temporal_decay
        self.decay_halflife = decay_halflife or timedelta(days=30)
        self._events: List[MemoryEvent] = []
        self._norms: List[float] = []
        self._access_counts: List[int] = []  # Track access frequency

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
            self._access_counts.append(0)
            return True
        return False

    def consolidate_batch(self, events: List[MemoryEvent]) -> Tuple[int, int]:
        """Consolidate multiple events at once.

        Returns:
            Tuple of (accepted_count, rejected_count)
        """
        accepted = 0
        rejected = 0
        for event in events:
            if self.consolidate(event):
                accepted += 1
            else:
                rejected += 1
        return (accepted, rejected)

    def _apply_temporal_decay(self, similarity: float, timestamp: datetime, now: datetime) -> float:
        """Apply temporal decay to similarity score."""
        if not self.temporal_decay:
            return similarity

        age = now - timestamp
        half_lives = age / self.decay_halflife
        decay_factor = 0.5 ** half_lives.total_seconds()
        return similarity * max(decay_factor, 0.1)  # Floor at 10%

    def recall(
        self,
        query_embedding: List[float] | np.ndarray,
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        filter_class: Optional[str] = None,
        importance_weight: float = 0.0
    ) -> List[MemoryEvent]:
        """Recall memories by similarity to query embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return (None = use default)
            filter_tags: Only return events with these tags
            exclude_tags: Exclude events with these tags
            filter_class: Only return events of this class
            importance_weight: Weight for importance boosting (0.0 = pure similarity)

        Returns:
            List of matching MemoryEvent objects
        """
        if not self._events:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-8
        now = datetime.utcnow()

        scored: List[Tuple[float, int, MemoryEvent]] = []

        for idx, (ev, n) in enumerate(zip(self._events, self._norms)):
            # Apply tag filters
            if filter_tags and not any(tag in ev.tags for tag in filter_tags):
                continue
            if exclude_tags and any(tag in ev.tags for tag in exclude_tags):
                continue
            if filter_class and ev.event_class != filter_class:
                continue

            # Compute cosine similarity
            v = np.asarray(ev.embedding, dtype=np.float32)  # type: ignore
            sim = float(np.dot(v, q) / (n * qn))

            # Apply temporal decay if enabled
            sim = self._apply_temporal_decay(sim, ev.timestamp, now)

            # Optionally boost by importance
            if importance_weight > 0:
                sim = sim * (1.0 + importance_weight * ev.importance)

            scored.append((sim, idx, ev))

        # Sort by score
        scored.sort(key=lambda t: t[0], reverse=True)

        # Take top-k above threshold
        k = top_k or self.top_k
        results = []
        for sim, idx, ev in scored[:k]:
            if sim >= self.sim_threshold:
                self._access_counts[idx] += 1  # Track access
                results.append(ev)

        return results

    def forget_low_importance(self, min_importance: float = 0.3, keep_pinned: bool = True) -> int:
        """Remove events below a certain importance threshold.

        Args:
            min_importance: Minimum importance to keep
            keep_pinned: Keep events with 'pin' tag regardless of importance

        Returns:
            Number of events removed
        """
        indices_to_keep = []
        for idx, ev in enumerate(self._events):
            if keep_pinned and "pin" in ev.tags:
                indices_to_keep.append(idx)
            elif ev.importance >= min_importance:
                indices_to_keep.append(idx)

        removed = len(self._events) - len(indices_to_keep)

        self._events = [self._events[i] for i in indices_to_keep]
        self._norms = [self._norms[i] for i in indices_to_keep]
        self._access_counts = [self._access_counts[i] for i in indices_to_keep]

        return removed

    def forget_old(self, older_than: timedelta, keep_pinned: bool = True) -> int:
        """Remove events older than the specified age.

        Args:
            older_than: Remove events older than this duration
            keep_pinned: Keep events with 'pin' tag regardless of age

        Returns:
            Number of events removed
        """
        now = datetime.utcnow()
        cutoff = now - older_than

        indices_to_keep = []
        for idx, ev in enumerate(self._events):
            if keep_pinned and "pin" in ev.tags:
                indices_to_keep.append(idx)
            elif ev.timestamp >= cutoff:
                indices_to_keep.append(idx)

        removed = len(self._events) - len(indices_to_keep)

        self._events = [self._events[i] for i in indices_to_keep]
        self._norms = [self._norms[i] for i in indices_to_keep]
        self._access_counts = [self._access_counts[i] for i in indices_to_keep]

        return removed

    def forget_least_accessed(self, keep_top_n: int, keep_pinned: bool = True) -> int:
        """Keep only the most frequently accessed memories.

        Args:
            keep_top_n: Number of memories to keep
            keep_pinned: Keep events with 'pin' tag regardless of access

        Returns:
            Number of events removed
        """
        if len(self._events) <= keep_top_n:
            return 0

        # Separate pinned and unpinned
        pinned_indices = []
        unpinned_indices = []

        for idx, ev in enumerate(self._events):
            if keep_pinned and "pin" in ev.tags:
                pinned_indices.append(idx)
            else:
                unpinned_indices.append(idx)

        # Sort unpinned by access count
        unpinned_indices.sort(key=lambda i: self._access_counts[i], reverse=True)

        # Keep top N unpinned + all pinned
        keep_unpinned = unpinned_indices[:keep_top_n]
        indices_to_keep = sorted(pinned_indices + keep_unpinned)

        removed = len(self._events) - len(indices_to_keep)

        self._events = [self._events[i] for i in indices_to_keep]
        self._norms = [self._norms[i] for i in indices_to_keep]
        self._access_counts = [self._access_counts[i] for i in indices_to_keep]

        return removed

    def get_by_tags(self, tags: List[str], match_all: bool = False) -> List[MemoryEvent]:
        """Retrieve memories by tags without using embeddings.

        Args:
            tags: Tags to filter by
            match_all: If True, event must have ALL tags; if False, ANY tag

        Returns:
            List of matching events
        """
        if match_all:
            return [ev for ev in self._events if all(tag in ev.tags for tag in tags)]
        else:
            return [ev for ev in self._events if any(tag in ev.tags for tag in tags)]

    def get_by_class(self, event_class: str) -> List[MemoryEvent]:
        """Retrieve all memories of a specific class."""
        return [ev for ev in self._events if ev.event_class == event_class]

    def get_recent(self, n: int = 10) -> List[MemoryEvent]:
        """Get the n most recently consolidated memories."""
        return self._events[-n:] if n <= len(self._events) else self._events

    def stats(self) -> Dict[str, any]:
        """Get detailed statistics about stored memories."""
        dims = 0
        if self._events and self._events[0].embedding is not None:
            dims = int(np.asarray(self._events[0].embedding).size)

        by_class: Dict[str, int] = {}
        total_importance = 0.0
        pinned_count = 0

        for ev in self._events:
            by_class[ev.event_class] = by_class.get(ev.event_class, 0) + 1
            total_importance += ev.importance
            if "pin" in ev.tags:
                pinned_count += 1

        avg_importance = total_importance / len(self._events) if self._events else 0.0

        return {
            "count": len(self._events),
            "dims": dims,
            "by_class": by_class,
            "avg_importance": avg_importance,
            "pinned": pinned_count,
            "total_accesses": sum(self._access_counts)
        }
