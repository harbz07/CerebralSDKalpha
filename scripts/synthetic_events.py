"""Template-driven synthetic MemoryEvent generation."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from typing import Iterable, List

from sdk.core.types import MemoryEvent, SCORE_DIMENSIONS


def _score_template(seed: int) -> dict[str, float]:
    base = 5 + (seed % 3)  # 5..7 keeps totals above threshold once summed
    return {dim: float(base + ((seed + i) % 2)) for i, dim in enumerate(SCORE_DIMENSIONS)}


def generate_events(count: int, *, start: datetime | None = None) -> List[MemoryEvent]:
    """Return ``count`` synthetic events satisfying the Catch-22 rule."""

    now = start or datetime.utcnow()
    events: List[MemoryEvent] = []
    for i in range(count):
        scores = _score_template(i)
        events.append(
            MemoryEvent(
                id=f"synthetic-{i:03d}",
                timestamp=now + timedelta(seconds=i),
                content=f"Synthetic memory {i}",
                event_class="Foundation" if i % 3 else "Glow",
                scores=scores,
                importance=0.5 + (i % 5) * 0.1,
                novelty=0.2 + (i % 7) * 0.05,
                tags=["synthetic"],
                embedding=[random.random() for _ in range(4)],
            )
        )
    return events


def _events_to_json(events: Iterable[MemoryEvent]) -> str:
    return "\n".join(json.dumps(event.to_dict()) for event in events)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of events to generate")
    parser.add_argument("--output", type=str, default="-", help="File path or '-' for stdout")
    args = parser.parse_args(list(argv) if argv is not None else None)

    events = generate_events(args.count)
    payload = _events_to_json(events)

    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()

