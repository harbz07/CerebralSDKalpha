"""Core dataclasses and helpers shared across the Cerebral SDK memory modules.

The original repository shipped an RFC in place of code.  This module replaces
that document with concrete, well‑typed implementations that the rest of the
SDK (Hippocampus, PrefrontalCache, ParietalGraph) can import without relying on
stubbed behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

SCORE_DIMENSIONS: Sequence[str] = (
    "error_severity",
    "novelty",
    "foundation_weight",
    "rlhf_weight",
)


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid ISO timestamp: {value!r}") from exc
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


@dataclass(slots=True)
class MemoryEvent:
    """Container for all memory subsystems.

    Attributes mirror the schema from the Memory API RFC.  The class performs
    minimal validation so callers receive early feedback if required score
    dimensions are missing.
    """

    id: str
    timestamp: datetime
    content: str
    event_class: str
    scores: MutableMapping[str, float]
    importance: float
    novelty: float
    tags: List[str] = field(default_factory=list)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    embedding: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        self.timestamp = _coerce_timestamp(self.timestamp)
        missing: List[str] = [dim for dim in SCORE_DIMENSIONS if dim not in self.scores]
        if missing:
            raise ValueError(f"scores missing dimensions: {', '.join(missing)}")

    @property
    def total_score(self) -> float:
        """Return the Catch‑22 score (sum of the four canonical dimensions)."""

        return float(sum(self.scores[dim] for dim in SCORE_DIMENSIONS))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "event_class": self.event_class,
            "scores": dict(self.scores),
            "importance": self.importance,
            "novelty": self.novelty,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }
        if self.embedding is not None:
            data["embedding"] = list(self.embedding)
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryEvent":
        return cls(
            id=str(payload["id"]),
            timestamp=payload["timestamp"],
            content=str(payload.get("content", "")),
            event_class=str(payload.get("event_class", "")),
            scores=dict(payload.get("scores", {})),
            importance=float(payload.get("importance", 0.0)),
            novelty=float(payload.get("novelty", 0.0)),
            tags=list(payload.get("tags", [])),
            metadata=dict(payload.get("metadata", {})),
            embedding=list(payload["embedding"]) if payload.get("embedding") is not None else None,
        )

from typing import Sequence, List

class VectorIndex:
    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    def search(self, query: Sequence[float], top_k: int) -> List[tuple[str, float]]:
        raise NotImplementedError

__all__ = ["MemoryEvent", "SCORE_DIMENSIONS", "VectorIndex"]

