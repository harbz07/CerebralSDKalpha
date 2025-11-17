from __future__ import annotations
from collections import deque
from datetime import datetime
from typing import List, Deque, Tuple
import heapq

from .types import MemoryEvent

class PrefrontalCache:
    """Working-memory buffer with O(1) push/evict; selection by recency×importance."""
    def __init__(self, max_events: int = 1000):
        self._buf: Deque[MemoryEvent] = deque(maxlen=max_events)

    def push(self, event: MemoryEvent) -> None:
        self._buf.append(event)

    def recent(self, n: int = 20) -> List[MemoryEvent]:
        return list(reversed(list(self._buf)[-n:]))

    def sample_for_context(self, budget: int = 8) -> List[str]:
        if budget <= 0 or not self._buf:
            return []
        items = list(self._buf)                   # oldest .. newest
        last = len(items)
        # weight = importance primary with recency as a mild boost
        scored: List[Tuple[float, int, MemoryEvent]] = [
            (ev.importance * (1.0 + ((i + 1) / last)), i, ev)
            for i, ev in enumerate(items)
        ]
        top = heapq.nlargest(budget, scored, key=lambda t: t[0])
        # newest-first within chosen
        top_sorted = sorted(top, key=lambda t: t[1], reverse=True)
        return [ev.content for _, _, ev in top_sorted]

    def __len__(self) -> int:  # pragma: no cover
        return len(self._buf)
