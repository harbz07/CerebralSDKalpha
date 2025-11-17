from __future__ import annotations

"""Hippocampus – vector long term memory backed by SQLite + FAISS/Numpy."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json
import sqlite3

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

from .types import MemoryEvent, VectorIndex

CATCH_22_THRESHOLD = 22
DEFAULT_CLASS_BIAS: Dict[str, float] = {"Chaos": 0.0, "Foundation": 2.0, "Glow": 1.0}


class _NumpyIndex(VectorIndex):
    def __init__(self, dim: int = 0) -> None:
        self.dim = dim
        self._matrix = np.zeros((0, dim), dtype=np.float32) if dim else np.zeros((0, 0), dtype=np.float32)
        self._ids: List[str] = []

    def _ensure_dim(self, dim: int) -> None:
        if self.dim == 0:
            self.dim = dim
            self._matrix = np.zeros((0, dim), dtype=np.float32)
        elif dim != self.dim:
            raise ValueError(f"Embedding dimension {dim} incompatible with existing dim {self.dim}")

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self._ensure_dim(arr.shape[1])
        self._matrix = np.vstack([self._matrix, arr]) if self._matrix.size else arr
        self._ids.extend(ids)

    def search(self, query: Sequence[float], top_k: int) -> List[tuple[str, float]]:
        if not self._ids:
            return []
        q = np.asarray(query, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q = q.copy()
        else:
            q = q / q_norm
        sims = self._matrix @ q
        idx = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in idx]


def _faiss_index(dim: int) -> VectorIndex:
    if faiss is None:
        return _NumpyIndex(dim)
    index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]

    class _Wrapper(VectorIndex):
        def __init__(self) -> None:
            self.dim = dim
            self._index = index
            self._ids: List[str] = []

        def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
            if not vectors:
                return
            arr = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            self._index.add(arr)
            self._ids.extend(ids)

        def search(self, query: Sequence[float], top_k: int) -> List[tuple[str, float]]:
            if not self._ids:
                return []
            q = np.asarray(query, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm:
                q = q / q_norm
            scores, idx = self._index.search(q[np.newaxis, :], top_k)
            return [
                (self._ids[i], float(scores[0, j]))
                for j, i in enumerate(idx[0])
                if 0 <= i < len(self._ids)
            ]

    return _Wrapper()


class Hippocampus:
    """Vector long-term memory with Catch-22 gating and persistence."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.80,
        top_k: int = 5,
        class_bias: Optional[Dict[str, float]] = None,
        storage_path: Optional[Path | str] = None,
    ) -> None:
        self.sim_threshold = similarity_threshold
        self.top_k = top_k
        self.bias = class_bias or dict(DEFAULT_CLASS_BIAS)
        self._conn = sqlite3.connect(str(storage_path or ":memory:"))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._events: Dict[str, MemoryEvent] = {}
        self._index: VectorIndex = _NumpyIndex()
        self._load_events()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                event_class TEXT NOT NULL,
                scores TEXT NOT NULL,
                importance REAL NOT NULL,
                novelty REAL NOT NULL,
                tags TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding TEXT
            )
            """
        )
        self._conn.commit()

    def _load_events(self) -> None:
        cur = self._conn.execute("SELECT * FROM events ORDER BY timestamp")
        rows = cur.fetchall()
        for row in rows:
            event = MemoryEvent(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                content=row["content"],
                event_class=row["event_class"],
                scores=json.loads(row["scores"]),
                importance=row["importance"],
                novelty=row["novelty"],
                tags=json.loads(row["tags"]),
                metadata=json.loads(row["metadata"]),
                embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            )
            self._events[event.id] = event
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        vectors: List[Sequence[float]] = []
        ids: List[str] = []
        dim = 0
        for event in self._events.values():
            if event.embedding is None:
                continue
            vec = list(map(float, event.embedding))
            vectors.append(vec)
            ids.append(event.id)
            dim = len(vec)
        self._index = _faiss_index(dim) if dim else _NumpyIndex()
        if vectors:
            self._index.add(ids, vectors)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def _total_score(self, event: MemoryEvent) -> float:
        return float(event.total_score(class_bias=self.bias))

    def consolidate(self, event: MemoryEvent) -> bool:
        if "pin" not in event.tags and self._total_score(event) < CATCH_22_THRESHOLD:
            return False
        if event.embedding is None:
            raise ValueError("Hippocampus.consolidate: embedding required")
        payload = (
            event.id,
            event.timestamp.isoformat(),
            event.content,
            event.event_class,
            json.dumps(dict(event.scores)),
            float(event.importance),
            float(event.novelty),
            json.dumps(list(event.tags)),
            json.dumps(dict(event.metadata)),
            json.dumps(list(map(float, event.embedding))) if event.embedding else None,
        )
        self._conn.execute(
            """
            INSERT INTO events (
                id, timestamp, content, event_class, scores, importance, novelty, tags, metadata, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                timestamp=excluded.timestamp,
                content=excluded.content,
                event_class=excluded.event_class,
                scores=excluded.scores,
                importance=excluded.importance,
                novelty=excluded.novelty,
                tags=excluded.tags,
                metadata=excluded.metadata,
                embedding=excluded.embedding
            """,
            payload,
        )
        self._conn.commit()
        self._events[event.id] = event
        self._rebuild_index()
        return True

    def recall(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: Optional[int] = None,
    ) -> List[MemoryEvent]:
        if not self._events:
            return []
        k = top_k or self.top_k
        hits = self._index.search(query_embedding, k)
        results: List[MemoryEvent] = []
        for event_id, score in hits:
            if score < self.sim_threshold:
                continue
            ev = self._events.get(event_id)
            if ev:
                results.append(ev)
        return results[:k]

    def stats(self) -> Dict[str, int]:
        dims = 0
        if self._events:
            sample = next(iter(self._events.values()))
            if sample.embedding is not None:
                dims = len(sample.embedding)
        return {"count": len(self._events), "dims": dims}
