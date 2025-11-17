from __future__ import annotations

"""Template driven synthetic event generator used in tests/CI."""

from datetime import datetime, timedelta
from typing import Iterator
import random

from sdk.core.types import MemoryEvent

TEMPLATE = {
    "Chaos": {"scores": {"error_severity": 6, "novelty": 6, "foundation_weight": 4, "rlhf_weight": 6}},
    "Foundation": {"scores": {"error_severity": 5, "novelty": 5, "foundation_weight": 7, "rlhf_weight": 7}},
    "Glow": {"scores": {"error_severity": 4, "novelty": 7, "foundation_weight": 4, "rlhf_weight": 8}},
}


def generate_events(
    count: int,
    *,
    seed: int = 7,
    base_time: datetime | None = None,
    embedding_dim: int = 4,
) -> Iterator[MemoryEvent]:
    rng = random.Random(seed)
    now = base_time or datetime.utcnow()
    for i in range(count):
        event_class = rng.choice(list(TEMPLATE.keys()))
        scores = TEMPLATE[event_class]["scores"]
        embedding = [0.0] * embedding_dim
        embedding[i % embedding_dim] = 1.0
        yield MemoryEvent(
            id=f"synthetic-{i}",
            timestamp=now + timedelta(seconds=i),
            content=f"Synthetic event {i} for {event_class}",
            event_class=event_class,
            scores=scores,
            importance=1.0,
            novelty=0.5,
            tags=["pin"] if i % 11 == 0 else [],
            metadata={"seed": seed},
            embedding=embedding,
        )
