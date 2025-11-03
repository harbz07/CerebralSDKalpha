from __future__ import annotations

"""Core datatypes and protocols for the Cerebral SDK memory stack.

The RFC stored in this repository primarily lived in ``types.py`` so we
consolidate the prose into docstrings and executable code.  The module exports
lightweight dataclasses used across the PrefrontalCache, Hippocampus and
ParietalGraph implementations, as well as a handful of helper functions used by
both runtime and tests.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

Number = float


@dataclass(slots=True)
class MemoryEvent:
    """Immutable record representing a single memory observation."""

    id: str
    timestamp: datetime
    content: str
    event_class: str
    scores: Mapping[str, Number]
    importance: Number
    novelty: Number
    tags: List[str] = field(default_factory=list)
    metadata: MutableMapping[str, Number | str] = field(default_factory=dict)
    embedding: Optional[Sequence[Number]] = None

    def total_score(self, *, class_bias: Mapping[str, Number]) -> Number:
        base = sum(self.scores.values())
        return base + float(class_bias.get(self.event_class, 0.0))


@dataclass(slots=True)
class Retrieval:
    """Container for retrieval results emitted by subsystems."""

    source: str
    items: List[str]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AblationConfig:
    """Toggle switches used by the evaluation harness and ContextComposer."""

    no_kg: bool = False
    no_ltm: bool = False
    only_pfc: bool = False

    @classmethod
    def from_flags(
        cls,
        *,
        no_kg: bool = False,
        no_ltm: bool = False,
        only_pfc: bool = False,
    ) -> "AblationConfig":
        if only_pfc and (no_kg or no_ltm):
            raise ValueError("--only-pfc already implies --no-kg and --no-ltm")
        return cls(no_kg=no_kg or only_pfc, no_ltm=no_ltm or only_pfc, only_pfc=only_pfc)


class VectorIndex(Protocol):
    """Protocol implemented by FAISS/Numpy vector indexes."""

    dim: int

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[Number]]) -> None: ...

    def search(self, query: Sequence[Number], top_k: int) -> List[Tuple[str, float]]: ...


@dataclass
class Report:
    """Structure returned by evaluation harness runs."""

    metrics: Dict[str, Number]
    provenance: Dict[str, object]


def normalise(vectors: Iterable[Sequence[Number]]) -> List[List[Number]]:
    import numpy as np

    arr = np.asarray(list(vectors), dtype=np.float32)
    if arr.size == 0:
        return []
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()


def chunk(iterable: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(iterable), size):
        yield list(iterable[i : i + size])
