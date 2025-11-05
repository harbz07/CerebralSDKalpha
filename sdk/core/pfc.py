from __future__ import annotations
from collections import deque
from datetime import datetime, timedelta
from typing import List, Deque, Tuple, Optional, Dict, Callable
import heapq

from .types import MemoryEvent

class PrefrontalCache:
    """Working-memory buffer with O(1) push/evict; selection by recency×importance.

    Enhanced with:
    - Attention-based weighting
    - Temporal decay mechanisms
    - Tag-based filtering
    - Custom scoring functions
    """
    def __init__(
        self,
        max_events: int = 1000,
        decay_halflife: Optional[timedelta] = None,
        attention_boost: float = 1.5
    ):
        """Initialize PrefrontalCache.

        Args:
            max_events: Maximum number of events to store
            decay_halflife: Half-life for temporal decay (None = no decay)
            attention_boost: Multiplier for events with 'attention' tag
        """
        self._buf: Deque[MemoryEvent] = deque(maxlen=max_events)
        self.decay_halflife = decay_halflife
        self.attention_boost = attention_boost

    def push(self, event: MemoryEvent) -> None:
        """Add an event to working memory."""
        self._buf.append(event)

    def recent(self, n: int = 20) -> List[MemoryEvent]:
        """Get n most recent events, newest first."""
        return list(reversed(list(self._buf)[-n:]))

    def by_tags(self, tags: List[str], match_all: bool = False) -> List[MemoryEvent]:
        """Filter events by tags.

        Args:
            tags: Tags to filter by
            match_all: If True, event must have ALL tags; if False, ANY tag

        Returns:
            List of matching events, newest first
        """
        if match_all:
            matches = [ev for ev in self._buf if all(tag in ev.tags for tag in tags)]
        else:
            matches = [ev for ev in self._buf if any(tag in ev.tags for tag in tags)]
        return list(reversed(matches))

    def by_class(self, event_class: str) -> List[MemoryEvent]:
        """Get all events of a specific class (Chaos, Foundation, Glow)."""
        matches = [ev for ev in self._buf if ev.event_class == event_class]
        return list(reversed(matches))

    def _compute_score(
        self,
        event: MemoryEvent,
        index: int,
        total: int,
        now: Optional[datetime] = None,
        custom_scorer: Optional[Callable[[MemoryEvent], float]] = None
    ) -> float:
        """Compute weighted score for an event.

        Score = (recency_weight × importance) × attention_boost × decay_factor × custom
        """
        # Base recency weight
        recency_weight = (index + 1) / total
        score = recency_weight * event.importance

        # Attention boost
        if "attention" in event.tags or "urgent" in event.tags:
            score *= self.attention_boost

        # Temporal decay
        if self.decay_halflife is not None and now is not None:
            age = now - event.timestamp
            half_lives = age / self.decay_halflife
            decay_factor = 0.5 ** half_lives.total_seconds()
            score *= max(decay_factor, 0.1)  # Floor at 10% of original score

        # Custom scoring function
        if custom_scorer is not None:
            score *= custom_scorer(event)

        return score

    def sample_for_context(
        self,
        budget: int = 8,
        custom_scorer: Optional[Callable[[MemoryEvent], float]] = None,
        filter_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None
    ) -> List[str]:
        """Sample events for context using weighted selection.

        Args:
            budget: Maximum number of events to return
            custom_scorer: Optional function to compute additional weight
            filter_tags: If provided, only include events with these tags
            exclude_tags: If provided, exclude events with these tags

        Returns:
            List of event content strings, ordered by score (descending)
        """
        if budget <= 0 or not self._buf:
            return []

        items = list(self._buf)

        # Apply tag filters
        if filter_tags:
            items = [ev for ev in items if any(tag in ev.tags for tag in filter_tags)]
        if exclude_tags:
            items = [ev for ev in items if not any(tag in ev.tags for tag in exclude_tags)]

        if not items:
            return []

        last = len(items)
        now = datetime.utcnow()

        # Compute weighted scores
        scored: List[Tuple[float, int, MemoryEvent]] = [
            (self._compute_score(ev, i, last, now, custom_scorer), i, ev)
            for i, ev in enumerate(items)
        ]

        top = heapq.nlargest(budget, scored, key=lambda t: t[0])
        # Order primarily by score (higher first) and secondarily by recency
        top_sorted = sorted(top, key=lambda t: (t[0], t[1]), reverse=True)
        return [ev.content for _, _, ev in top_sorted]

    def get_attention_events(self, n: int = 5) -> List[MemoryEvent]:
        """Get events tagged with 'attention' or 'urgent', newest first."""
        attention = [ev for ev in self._buf if "attention" in ev.tags or "urgent" in ev.tags]
        return list(reversed(attention))[-n:]

    def clear_old(self, older_than: timedelta) -> int:
        """Remove events older than the specified timedelta.

        Returns:
            Number of events removed
        """
        now = datetime.utcnow()
        cutoff = now - older_than
        original_len = len(self._buf)

        # Filter out old events
        self._buf = deque(
            (ev for ev in self._buf if ev.timestamp >= cutoff),
            maxlen=self._buf.maxlen
        )

        return original_len - len(self._buf)

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if not self._buf:
            return {
                "count": 0,
                "capacity": self._buf.maxlen or 0,
                "by_class": {}
            }

        by_class: Dict[str, int] = {}
        for ev in self._buf:
            by_class[ev.event_class] = by_class.get(ev.event_class, 0) + 1

        return {
            "count": len(self._buf),
            "capacity": self._buf.maxlen or 0,
            "by_class": by_class
        }

    def __len__(self) -> int:  # pragma: no cover
        return len(self._buf)
