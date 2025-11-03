from __future__ import annotations

"""Reporting utilities writing JSON + Markdown summaries."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict
import json

from .metrics import mean_reciprocal_rank, recall_at_k


@dataclass(slots=True)
class RunSummary:
    run_id: str
    metrics: Dict[str, float]

    def to_markdown(self) -> str:
        lines = [f"### Eval Run {self.run_id}"]
        for key, value in sorted(self.metrics.items()):
            lines.append(f"- **{key}**: {value:.3f}")
        return "\n".join(lines)


def write_json_report(data: Dict[str, float], *, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    path = directory / f"{run_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"metrics": data, "timestamp": datetime.utcnow().isoformat()}, handle, indent=2)
    return path
