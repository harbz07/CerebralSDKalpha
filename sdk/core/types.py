"""Core data structures for the Cerebral SDK memory triad.

This module replaces the earlier RFC-only placeholder with a lightweight
runtime representation of memory events and composer configuration.  The goal
is to keep validation local to the dataclasses so that downstream modules such
as :mod:`sdk.core.hippocampus` and :mod:`sdk.core.context` can rely on a stable
interface without repeating boilerplate checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

SCORE_DIMENSIONS: tuple[str, ...] = (
    "error_severity",
    "novelty",
    "foundation_weight",
    "rlhf_weight",
)


def _ensure_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    """Validate and normalise the score payload.

    The Memory API RFC mandates four scoring dimensions.  The helper keeps the
    contract flexible enough for callers to omit a field (it is implicitly
    treated as zero) while guarding against completely unrelated keys that may
    hint at a bug in the ingest pipeline.
    """

    normalised: Dict[str, float] = {dim: float(scores.get(dim, 0.0)) for dim in SCORE_DIMENSIONS}

    unexpected = sorted(set(scores) - set(SCORE_DIMENSIONS))
    if unexpected:
        raise ValueError(f"Unexpected score dimensions: {unexpected}")

    return normalised


@dataclass(slots=True)
class MemoryEvent:
    """In-memory representation of a scored memory event.

    Parameters mirror the schema agreed in the Memory API RFC.  The dataclass
    is intentionally strict about known scoring dimensions and value types to
    prevent subtle errors during consolidation.
    """

    id: str
    timestamp: datetime
    content: str
    event_class: str
    scores: Dict[str, float]
    importance: float
    novelty: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime instance")

        if isinstance(self.scores, MutableMapping):
            # The helper will create a new dict so copy() is not required.
            self.scores = _ensure_scores(self.scores)
        else:
            self.scores = _ensure_scores(dict(self.scores))

        self.tags = list(self.tags)
        self.metadata = dict(self.metadata)

        if self.embedding is not None:
            self.embedding = [float(v) for v in self.embedding]

    @property
    def score_total(self) -> float:
        """Return the sum of the four score dimensions.

        The helper keeps Hippocampus' Catch-22 logic readable during tests and
        acts as a convenient computed field for reporting.
        """

        return float(sum(self.scores[dim] for dim in SCORE_DIMENSIONS))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the event to a JSON-friendly dictionary."""

        data: Dict[str, Any] = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "event_class": self.event_class,
            "scores": dict(self.scores),
            "importance": float(self.importance),
            "novelty": float(self.novelty),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }
        if self.embedding is not None:
            data["embedding"] = list(self.embedding)
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryEvent":
        """Construct an event from a mapping-like payload."""

        timestamp = payload.get("timestamp")
        if isinstance(timestamp, str):
            timestamp_dt = datetime.fromisoformat(timestamp)
        elif isinstance(timestamp, datetime):
            timestamp_dt = timestamp
        else:
            raise TypeError("timestamp must be an ISO string or datetime")

        return cls(
            id=str(payload["id"]),
            timestamp=timestamp_dt,
            content=str(payload.get("content", "")),
            event_class=str(payload.get("event_class", "")),
            scores=dict(payload.get("scores", {})),
            importance=float(payload.get("importance", 0.0)),
            novelty=float(payload.get("novelty", 0.0)),
            tags=list(payload.get("tags", [])),
            metadata=dict(payload.get("metadata", {})),
            embedding=list(payload.get("embedding")) if payload.get("embedding") is not None else None,
        )


@dataclass(slots=True)
class ComposerConfig:
    """Configuration flags consumed by :class:`ContextComposer`.

    The booleans map directly to the CLI toggles defined in the evaluation
    harness RFC.  The dataclass keeps the relationship between flags explicit
    and performs a single derived normalisation step: when ``only_pfc`` is set
    we implicitly disable KG and LTM access regardless of the other fields.
    """

    use_pfc: bool = True
    use_kg: bool = True
    use_ltm: bool = True
    only_pfc: bool = False

    @classmethod
    def from_flags(
        cls,
        *,
        no_kg: bool = False,
        no_ltm: bool = False,
        only_pfc: bool = False,
    ) -> "ComposerConfig":
        """Create a config object from CLI-style switches."""

        use_kg = not no_kg and not only_pfc
        use_ltm = not no_ltm and not only_pfc
        use_pfc = True  # The composer always relies on the cache when available.
        return cls(use_pfc=use_pfc, use_kg=use_kg, use_ltm=use_ltm, only_pfc=only_pfc)

    def enabled_sources(self) -> Iterable[str]:
        """Yield a stable list of enabled memory sources."""

        if self.use_pfc:
            yield "pfc"
        if self.use_kg:
            yield "kg"
        if self.use_ltm:
            yield "ltm"

