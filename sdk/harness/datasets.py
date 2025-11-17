from __future__ import annotations

"""Dataset loaders for curated QA/entities/snippets fixtures."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import json

DATASET_ROOT = Path("knowledge/datasets")


@dataclass(slots=True)
class QAExample:
    id: str
    query: str
    answers: List[str]
    embedding: List[float]


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_qa_dataset(root: Path | None = None) -> List[QAExample]:
    root = root or DATASET_ROOT
    path = root / "qa.jsonl"
    return [QAExample(**row) for row in _load_jsonl(path)]


def load_entities_dataset(root: Path | None = None) -> List[dict]:
    root = root or DATASET_ROOT
    return list(_load_jsonl(root / "entities.jsonl"))


def load_snippets_dataset(root: Path | None = None) -> List[dict]:
    root = root or DATASET_ROOT
    return list(_load_jsonl(root / "snippets.jsonl"))
