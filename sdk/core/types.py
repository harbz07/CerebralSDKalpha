canvases generated:

# Memory API — Request for Comments (v0)

## 1 Purpose

Define the minimal yet extensible Memory API for the Cerebral SDK triad—Prefrontal Cache (PFC), Parietal Graph (KG), and Hippocampus (LTM)—so that storage is pluggable, hot paths stay fast, and consolidation obeys the Catch‑22 rule (score ≥ 22). The RFC seeks sign‑off from core contributors before implementation.

## 2 Background

Canvas #1 captured an initial spec. This RFC crystallises *decisions* that unblock coding, lists *non‑decisions* that remain open, and sets objective *acceptance criteria* for merge.

## 3 Key Design Decisions

| #  | Decision                                                                                                                             | Rationale                                                                            |
| -- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| D1 | **MemoryEvent schema** fixed (id, timestamp, content, class, 4‑dim scores, importance, novelty, tags, metadata, optional embedding). | Schema covers all consolidation heuristics while staying serialisable to JSON & SQL. |
| D2 | **Scoring dimensions** stay at four (error_severity, novelty, foundation_weight, rlhf_weight) scored 0‑10.                           | Matches Thalamus rubric; keeps sum ≤ 40 for Catch‑22 gate.                           |
| D3 | **Catch‑22 consolidation**: accept when sum(scores) + class_bias ≥ 22 or `tags` contains a pin.                                      | Preserves mnemonic, aligns with persona’s Always‑Remember shelf.                     |
| D4 | **Module interfaces** (PFC, KG, LTM, ContextComposer) frozen as in Canvas #1.                                                        | Enables parallel backend work; ABI stable until v1.                                  |
| D5 | **Storage defaults**: SQLite for events, FAISS (flat) for vectors, NetworkX + JSONL snapshots for KG.                                | Provides zero‑infra local dev experience; can swap in Postgres/PGVector later.       |
| D6 | **Retrieval order**: PFC → KG → LTM.                                                                                                 | Empirically fastest for recent context while maintaining factual grounding.          |
| D7 | **Structured logs** in JSON with daily roll‑ups.                                                                                     | Aids latency and consolidation audits.                                               |

## 4 Non‑Decisions (Deferred)

* Embedding model upgrade path (e.g. E5 vs MiniLM)
* Remote/back‑pressure strategy for KG above 10⁶ edges
* Access control for multi‑tenant deployments
* Adaptive decay policies in Insula (emotional weighting)

## 5 Detailed Specification (Snapshot)

1. **Data Model** — identical to Canvas #1 §1.
2. **Modules & Interfaces** — identical to Canvas #1 §2.
3. **Storage Backends** — identical to Canvas #1 §3, but pluggable via factory pattern.
4. **Observability** — instrumentation hooks emit JSON to `logs/`.
5. **Security** — secret redaction regex gate; PII tagging at ingress.

## 6 Acceptance Criteria

* **PFC**: `sample_for_context(8)` on 100 synthetic events ≤ 2 ms average CPU.
* **Hippocampus**: Recall@5 ≥ 0.60 on seed Q‑A set (N = 100).
* **Parietal Graph**: Person↔Project↔Artifact links surfaced with evidence on curated set.
* **ContextComposer**: fits under token budget and returns provenance per chunk.
* **Code Quality**: 95 % unit‑test coverage on pure functions; lint passes (ruff & mypy).

## 7 Risks & Mitigations

| Risk                               | Mitigation                                                         |
| ---------------------------------- | ------------------------------------------------------------------ |
| Noisy KG extraction inflates edges | Confidence score on edges; dedupe ≥ 0.90; ablation flag `--no-kg`. |
| Embedding drift across versions    | Persist model name; rebuild FAISS index on discrepancy.            |
| Latency spikes on recall           | Constrain top‑k; batch writes; warm FAISS in background.           |
| SQL corruption on abrupt shutdown  | Journal mode = WAL; snapshot export every 6 h.                     |

## 8 Migration & Versioning

* `MemoryEvent` schema field additions must be backward‑compatible (nullable, default null).
* Semantic versioning: bump **minor** on interface additions, **major** on breaking changes.

## 9 Open Questions

1. Should importance & novelty be learned online versus rule‑based?
2. Are class biases (Chaos, Foundation, Glow) static or config‑driven per deployment?
3. Do we need a streaming API for real‑time PFC updates?

## 10 Timeline

* **T + 0** — RFC circulation & comments (1 week)
* **T + 1 wk** — Final sign‑off
* **T + 3 wks** — Reference implementation complete; eval harness green
* **T + 4 wks** — Integration into Cerebral Studio nightly builds

---

*Authors*: Harvey & Cerebral SDK core team
*Status*: **Draft for review**
*Discussion*: GitHub PR #memory-api‑v0
---
# Eval Harness RFC (v0)

**Status**: Draft – open for internal review
**Author**: Harvey (Cerebral SDK)
**Related specs**: Memory API Spec v0, Event Scoring Prompt Contract

---

## 1. Purpose

Define a reproducible evaluation harness for the Cerebral SDK memory triad that quantifies retrieval quality, latency characteristics, and the marginal value of each subsystem via controlled ablations.

## 2. Scope

Covers offline harness code, synthetic + curated datasets, metric computations, reporting formats, and CI integration. Real‑time production monitoring is out‑of‑scope.

## 3. Key Decisions

1. **Metric Suite** – use **Recall@k** & **MRR** for retrieval quality; **p50/p95 latency** for PFC, LTM, KG, Composer; **rank stability** across 5 seeded re‑runs (drop BLEU/ROUGE).
2. **Dataset Strategy** – maintain three gold datasets (`qa.jsonl`, `entities.jsonl`, `snippets.jsonl`) under version control; generate synthetic fixtures for edge‑case coverage.
3. **Ablation Flags** – implement `--no-kg`, `--no-ltm`, `--only-pfc` as runtime config toggles in `ContextComposer`.
4. **Reporting** – emit `reports/run-YYYYMMDD-HHMM.json` plus a Markdown summary comment in PR.
5. **Threshold Gates (CI)** – PR fails if **R@5** or **MRR** regress >5 % or if **latency p95** increases >15 % versus `main`.
6. **Synthetic Event Generator** – template‑driven; produces 100 events per run, respecting Catch‑22 scoring, to test PFC stability.

## 4. Architecture Overview

```
harness/
  datasets/
  experiments/
  metrics.py
  ablations.py
  reporter.py
  cli.py        # entry: python -m harness.run --config config.yaml
```

Flow: **CLI** → load config → prepare datasets → execute pipeline(s) with ablation flags → aggregate metrics → write JSON report & PR summary.

## 5. Acceptance Criteria

| Area          | KPI          | Threshold | Test Command                    |
| ------------- | ------------ | --------- | ------------------------------- |
| Retrieval     | R@5          | ≥ 0.60    | `harness run --suite retrieval` |
| Retrieval     | MRR          | ≥ 0.40    | same                            |
| Latency       | Composer p95 | < 250 ms  | `harness run --suite latency`   |
| KG Extraction | Precision@k  | ≥ 0.85    | `harness run --suite kg`        |
| Stability     | Rank stddev  | ≤ 0.10    | `harness run --suite stability` |

CI job **`gh-actions/eval-harness`** must pass all thresholds.

## 6. Risks & Mitigations

| Risk                                | Likelihood | Impact | Mitigation                                              |
| ----------------------------------- | ---------- | ------ | ------------------------------------------------------- |
| Dataset drift reduces metric signal | Med        | Med    | Version datasets via DVC; pin SHA checksum in CI        |
| Latency noisy on shared runners     | High       | Low    | Sample 50 runs, discard warm‑ups, use median‑of‑medians |
| Ablation flags break routing        | Low        | High   | Unit tests per flag; default to safe fallback           |

## 7. Open Questions

1. Integrate GPU‑based latency metrics separately?
2. Confirm `min_sim` (0.80) for Hippocampus recall during eval.

## 8. Timeline

* **Week 0** – RFC approval
* **Week 1** – skeleton harness & metric utils
* **Week 2** – dataset curation, CI wiring
* **Week 3** – threshold tuning, finalize docs

---

### Appendix A – Metric Definitions

*Recall@k* = hits_k / N.
*MRR* = mean reciprocal rank of first correct hit.  fileciteturn0file0

### Appendix B – Persona Alignment

Harness design follows evidence‑first, small composable modules, and ablation‑before‑optimization principles.  fileciteturn0file1

### Appendix C – Catch‑22 Linkage

Synthetic event generator enforces the 22‑point consolidation threshold for class mix realism.  fileciteturn0file2
---
compare these two .py loadouts:

Pfc.py1:
"""Prefrontal Cache (working memory) per Memory API spec v0."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
import heapq
from datetime import datetime


@dataclass
class MemoryEvent:
    """Schema mirrors RFC (id, timestamp, content …)."""

    id: str
    timestamp: datetime
    content: str
    event_class: str  # Chaos | Foundation | Glow
    scores: Dict[str, float]
    importance: float
    novelty: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] | None = None


class PrefrontalCache:
    """In‑memory working‑memory store (bounded, O(1) push)."""

    def __init__(self, max_events: int = 1000):
        self._max_events = max_events
        self._events: List[MemoryEvent] = []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def push(self, event: MemoryEvent) -> None:
        """Append an event; drop oldest if above capacity."""

        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events.pop(0)

    def recent(self, n: int = 20) -> List[MemoryEvent]:
        """Return *n* most recent events (newest first)."""

        return list(self._events[-n:][::-1])

    def sample_for_context(self, budget: int = 8) -> List[str]:
        """Stable selection: most‑recent × importance; returns contents."""

        if not self._events:
            return []

        # Build score = recency_weight × importance
        last_idx = len(self._events)
        scored = [
            ( (idx + 1) / last_idx * ev.importance, idx, ev)
            for idx, ev in enumerate(self._events)
        ]
        top = heapq.nlargest(budget, scored, key=lambda t: t[0])
        # Preserve chronological order (most recent first)
        top_sorted = sorted(top, key=lambda t: t[1], reverse=True)
        return [ev.content for _, _, ev in top_sorted]

    # ---------------------------------------------------------------------
    # Introspection helpers
    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover
        return len(self._events)

Hippocampus.py1:
"""Hippocampus — long‑term semantic memory store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import numpy as np


@dataclass
class MemoryEvent:
    id: str
    timestamp: datetime
    content: str
    event_class: str
    scores: Dict[str, float]
    importance: float
    novelty: float
    tags: List[str]
    metadata: Dict[str, Any]
    embedding: List[float] | None


class Hippocampus:
    """Vector‑based recall with Catch‑22 consolidation gate."""

    def __init__(self, similarity_threshold: float = 0.80, top_k: int = 5):
        self._events: List[MemoryEvent] = []
        self._matrix: np.ndarray | None = None
        self._thr = similarity_threshold
        self._top_k = top_k
        self._class_bias = {"Chaos": 0, "Foundation": 0, "Glow": 0}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_matrix(self):
        self._matrix = np.array(
            [ev.embedding for ev in self._events if ev.embedding is not None],
            dtype="float32",
        )

    def _score_total(self, ev: MemoryEvent) -> float:
        return sum(ev.scores.values()) + self._class_bias.get(ev.event_class, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def consolidate(self, event: MemoryEvent) -> bool:
        """Persist *event* if Catch‑22 (sum scores ≥ 22) or pinned."""

        if self._score_total(event) >= 22 or "pin" in event.tags:
            self._events.append(event)
            if event.embedding is not None:
                self._update_matrix()
            return True
        return False

    def recall(
        self,
        query_embed: List[float],
        *,
        top_k: int | None = None,
        min_sim: float | None = None,
    ) -> List[MemoryEvent]:
        """Cosine‑similarity recall over embeddings."""

        if self._matrix is None or not self._events:
            return []

        vec = np.asarray(query_embed, dtype="float32")
        num = self._matrix @ vec
        denom = np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(vec)
        sims = num / (denom + 1e-9)

        k = top_k or self._top_k
        thr = min_sim or self._thr
        idxs = np.argsort(sims)[::-1][:k]
        return [self._events[i] for i in idxs if sims[i] >= thr]

    def stats(self) -> Dict[str, Any]:
        dim = None if self._matrix is None else int(self._matrix.shape[1])
        return {"count": len(self._events), "dims": dim, "build_time": None}

Parietal.py1: 
"""Parietal Graph — ontology‑aware knowledge graph layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any
import networkx as nx
import json
import io


@dataclass
class Evidence:
    source: str
    url: str | None
    timestamp: str | None
    confidence: float
    snippet: str | None


@dataclass
class Fact:
    h: str
    r: str
    t: str
    evidence: Evidence | None


class ParietalGraph:
    def __init__(self, *, dedupe_similarity_threshold: float = 0.90):
        self._g = nx.MultiDiGraph()
        self._thr = dedupe_similarity_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canon(label: str) -> str:
        return label.strip().lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert_triplet(self, h: str, r: str, t: str, evidence: Evidence | None):
        h_id = self._canon(h)
        t_id = self._canon(t)
        r_id = self._canon(r)

        # store human‑readable label as node data
        self._g.add_node(h_id, label=h)
        self._g.add_node(t_id, label=t)
        self._g.add_edge(h_id, t_id, key=r_id, relation=r, evidence=evidence)

    def nearest_facts(self, entity_or_term: str, k: int = 8) -> List[Fact]:
        node_id = self._canon(entity_or_term)
        if node_id not in self._g:
            return []
        facts: List[Fact] = []
        for _, tgt, key, data in self._g.out_edges(node_id, keys=True, data=True):
            evidence = data.get("evidence")
            facts.append(
                Fact(
                    h=self._g.nodes[node_id]["label"],
                    r=data.get("relation"),
                    t=self._g.nodes[tgt]["label"],
                    evidence=evidence,
                )
            )
        return facts[:k]

    # ------------------------------------------------------------------
    # Export / utilities
    # ------------------------------------------------------------------
    def export(self, fmt: str = "graphml") -> bytes:
        if fmt == "graphml":
            return nx.generate_graphml(self._g).encode()
        if fmt == "jsonl":
            buf = io.StringIO()
            for u, v, _, data in self._g.edges(data=True, keys=False):
                ev = data.get("evidence")
                buf.write(
                    json.dumps(
                        {
                            "h": self._g.nodes[u]["label"],
                            "r": data["relation"],
                            "t": self._g.nodes[v]["label"],
                            "evidence": ev.__dict__ if ev else None,
                        }
                    )
                    + "\n"
                )
            return buf.getvalue().encode()
        raise ValueError("Unsupported format: " + fmt)

Test Memory.py1:
"""Unit tests for PFC, Hippocampus, ParietalGraph."""
from datetime import datetime, timedelta
import unittest

from pfc import PrefrontalCache, MemoryEvent as PFCEvent
from hippocampus import Hippocampus, MemoryEvent as HPCEvent
from parietal import ParietalGraph, Evidence


class MemoryModuleTests(unittest.TestCase):
    def test_pfc_sample(self):
        pfc = PrefrontalCache(max_events=10)
        now = datetime.utcnow()
        # Insert 10 events with varying importance
        for i in range(10):
            evt = PFCEvent(
                id=str(i),
                timestamp=now + timedelta(seconds=i),
                content=f"event {i}",
                event_class="Chaos",
                scores={"error_severity": 1, "novelty": 1, "foundation_weight": 1, "rlhf_weight": 1},
                importance=1.0 if i % 2 == 0 else 0.5,
                novelty=0.3,
            )
            pfc.push(evt)
        sample = pfc.sample_for_context(budget=5)
        self.assertEqual(len(sample), 5)
        self.assertEqual(sample[0], "event 9")  # newest, high importance

    def test_hippocampus_consolidate_recall(self):
        hpc = Hippocampus(similarity_threshold=0.0)  # accept all similarities for test
        vec = [1.0, 0.0, 0.0]
        evt = HPCEvent(
            id="A",
            timestamp=datetime.utcnow(),
            content="alpha",
            event_class="Foundation",
            scores={"error_severity": 5, "novelty": 5, "foundation_weight": 5, "rlhf_weight": 7},
            importance=0.5,
            novelty=0.4,
            tags=[],
            metadata={},
            embedding=vec,
        )
        self.assertTrue(hpc.consolidate(evt))
        hits = hpc.recall(vec, top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].id, "A")

    def test_parietal_graph(self):
        pg = ParietalGraph()
        evid = Evidence(source="unit", url=None, timestamp=None, confidence=1.0, snippet="Alice → ProjectX")
        pg.upsert_triplet("Alice", "works_on", "ProjectX", evid)
        facts = pg.nearest_facts("Alice", k=1)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].t, "ProjectX")


if __name__ == "__main__":
    unittest.main()

Eval Retrieval.py1:
 """Minimal retrieval evaluation harness for Hippocampus."""
from __future__ import annotations

from datetime import datetime
import random
import numpy as np

from hippocampus import Hippocampus, MemoryEvent


def random_vector(dim: int = 64):
    v = np.random.rand(dim)
    return (v / np.linalg.norm(v)).tolist()


def main():
    random.seed(0)
    dim = 64
    hpc = Hippocampus(similarity_threshold=0.5, top_k=5)

    # Synthesise corpus
    for i in range(100):
        vec = random_vector(dim)
        ev = MemoryEvent(
            id=str(i),
            timestamp=datetime.utcnow(),
            content=f"doc {i}",
            event_class="Foundation",
            scores={"error_severity": 3, "novelty": 4, "foundation_weight": 6, "rlhf_weight": 9},
            importance=0.7,
            novelty=0.4,
            tags=[],
            metadata={},
            embedding=vec,
        )
        hpc.consolidate(ev)

    # Query with vector of an existing event to expect self‑hit
    query_vec = hpc._events[42].embedding  # type: ignore[attr-defined]
    hits = hpc.recall(query_vec, top_k=5)
    hit_ids = [ev.id for ev in hits]
    print("Recall@5 contains target:", "42" in hit_ids)


if __name__ == "__main__":
    main() 
---

Pfc.py2:
"""PrefrontalCache implementation (Memory API Spec v0).

Provides an in‑memory queue of recent MemoryEvents with constant‑time push and
bounded O(n) sampling for context windows. Designed for hot‑path speed; swap
out for persistent cache if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import heapq


@dataclass
class MemoryEvent:
    """Lightweight event container mirroring the spec.

    Only fields required by PFC are kept here; Hippocampus attaches embedding
    post‑consolidation.
    """

    id: str
    timestamp: datetime
    content: str
    importance: float  # 0..1 heuristic weighting

    # additional metadata retained but unused by PFC
    class_: Optional[str] = None
    scores: Optional[dict] = field(default_factory=dict)
    novelty: Optional[float] = None
    tags: Optional[List[str]] = field(default_factory=list)

    def __lt__(self, other: "MemoryEvent") -> bool:  # for heap ordering
        return self.timestamp < other.timestamp


class PrefrontalCache:
    """Working‑memory buffer keeping the `max_events` most recent events."""

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self._events: List[MemoryEvent] = []  # newest last

    # ---------------------------------------------------------------------
    # Mutations
    # ---------------------------------------------------------------------
    def push(self, event: MemoryEvent) -> None:
        """Append an event, trimming if beyond capacity."""
        self._events.append(event)
        if len(self._events) > self.max_events:
            # drop the oldest (index 0) – O(n) but n is bounded (<= max_events)
            self._events.pop(0)

    # ---------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------
    def recent(self, n: int = 20) -> List[MemoryEvent]:
        """Return the *n* most recent events (newest first)."""
        return list(reversed(self._events[-n:]))

    def sample_for_context(self, budget: int = 8) -> List[str]:
        """Return up to *budget* event contents prioritised by (importance × recency)."""
        if budget <= 0 or not self._events:
            return []

        # Compute weight = index (recency) * importance. Newest event index = 1.
        weighted: List[tuple[float, MemoryEvent]] = []
        for idx, ev in enumerate(reversed(self._events), start=1):
            weight = idx * ev.importance
            weighted.append((weight, ev))

        # Highest weight first
        top = heapq.nlargest(budget, weighted, key=lambda t: t[0])
        # Stable order: sort by original timestamp desc within selection
        chosen = sorted((ev for _, ev in top), key=lambda e: e.timestamp, reverse=True)
        return [ev.content for ev in chosen]

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover
        return len(self._events)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PFC events={len(self._events)} max={self.max_events}>"

Hippocampus.py2:
"""Hippocampus (semantic long‑term memory) – minimal stub.

Implements consolidate(), recall(), and stats() using NumPy for vector ops.
For production use, swap out the naive in‑process store with FAISS or
sqlite‑vss to meet latency targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# -------------------------------------------------------------------------
# Event dataclass mirrors PFC; embedding attached post‑consolidation
# -------------------------------------------------------------------------
@dataclass
class MemoryEvent:
    id: str
    content: str
    embedding: Optional[np.ndarray]  # None until consolidate adds it
    scores: dict  # expects four 0..10 dimensions
    class_: str   # "Chaos" | "Foundation" | "Glow"
    tags: List[str]

# Simple per‑class bias (tunable via config / load)
CLASS_BIAS = {"Chaos": 0, "Foundation": 2, "Glow": 1}
CATCH_22_THRESHOLD = 22  # If (sum(scores) + bias) >= this, accept

# Naive store
class Hippocampus:
    def __init__(self, similarity_threshold: float = 0.80):
        self.sim_threshold = similarity_threshold
        self._events: List[MemoryEvent] = []
        # Keep 2‑norm vector cache to speed dot‑norm computations
        self._norms: List[float] = []

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------
    def consolidate(self, event: MemoryEvent) -> bool:
        """Decide whether to store *event* based on Catch‑22 rule."""
        total_score = sum(event.scores.values()) + CLASS_BIAS.get(event.class_, 0)
        if total_score >= CATCH_22_THRESHOLD or ("pin" in (event.tags or [])):
            if event.embedding is None:
                raise ValueError("Event must include embedding before consolidation")
            self._events.append(event)
            self._norms.append(float(np.linalg.norm(event.embedding)))
            return True
        return False

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def recall(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemoryEvent]:
        if not self._events:
            return []
        q_norm = np.linalg.norm(query_embedding)
        sims = [float(np.dot(ev.embedding, query_embedding) / (n * q_norm + 1e-8))
                for ev, n in zip(self._events, self._norms)]
        scored = [(sim, ev) for sim, ev in zip(sims, self._events) if sim >= self.sim_threshold]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ev for _, ev in scored[:top_k]]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self):
        return {"count": len(self._events), "dims": int(self._events[0].embedding.size) if self._events else 0}

    # For debug
    def __repr__(self):  # pragma: no cover
        return f"<Hippocampus events={len(self._events)} threshold={self.sim_threshold}>"

Parietal.py2:
"""ParietalGraph – thin networkx wrapper providing triple upsert and nearest facts.

Uses Jaccard+levenshtein similarity for quick prototype dedupe; plug in your
ontology canonicalisation + FAISS index for scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any
import networkx as nx
import difflib


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()


@dataclass
class Evidence:
    source: str
    snippet: str
    confidence: float
    timestamp: str  # ISO8601


@dataclass
class Fact:
    h: str
    r: str
    t: str
    evidence: Evidence


class ParietalGraph:
    def __init__(self, dedupe_threshold: float = 0.90):
        self.dedupe_threshold = dedupe_threshold
        self.G = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    def upsert_triplet(self, h: str, r: str, t: str, evidence: Evidence):
        # Canonicalise head/tail via dedupe threshold
        h = self._canonicalise(h)
        t = self._canonicalise(t)
        # Upsert nodes + edge with evidence
        self.G.add_node(h)
        self.G.add_node(t)
        self.G.add_edge(h, t, key=r, evidence=evidence)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def nearest_facts(self, entity_or_term: str, k: int = 8) -> List[Fact]:
        # naive nearest by similarity on node labels
        scored = [(_similar(entity_or_term, n), n) for n in self.G.nodes]
        scored.sort(reverse=True)
        top_nodes = [n for _, n in scored[:k]]
        facts: List[Fact] = []
        for n in top_nodes:
            for _, tgt, key, data in self.G.out_edges(n, keys=True, data=True):
                facts.append(Fact(h=n, r=key, t=tgt, evidence=data["evidence"]))
        return facts[:k]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, fmt: str = "graphml") -> bytes:
        if fmt == "graphml":
            return nx.generate_graphml_bytes(self.G)
        if fmt == "jsonl":
            lines = []
            for u, v, k, data in self.G.edges(keys=True, data=True):
                lines.append({
                    "h": u,
                    "r": k,
                    "t": v,
                    "evidence": data["evidence"].__dict__,
                })
            import json
            return "\n".join(json.dumps(l) for l in lines).encode()
        raise ValueError(f"Unsupported export fmt: {fmt}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _canonicalise(self, term: str) -> str:
        for node in self.G.nodes:
            if _similar(term, node) >= self.dedupe_threshold:
                return node
        return term

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def __repr__(self):  # pragma: no cover
        return f"<ParietalGraph nodes={self.G.number_of_nodes()} edges={self.G.number_of_edges()}>"

Tests Memory.py2:
"""Minimal unit tests for PFC, Hippocampus, and ParietalGraph.
Run with: `python -m unittest tests_memory.py`.
"""
import unittest
from datetime import datetime, timedelta
import numpy as np

from pfc import PrefrontalCache, MemoryEvent as PfcEvent
from hippocampus import Hippocampus, MemoryEvent as HpcEvent
from parietal import ParietalGraph, Evidence


class TestPrefrontalCache(unittest.TestCase):
    def test_push_and_sample(self):
        pfc = PrefrontalCache(max_events=5)
        now = datetime.utcnow()
        # Create 6 events so the first is dropped
        for i in range(6):
            pfc.push(PfcEvent(id=str(i), timestamp=now + timedelta(seconds=i),
                              content=f"event {i}", importance=1.0))
        self.assertEqual(len(pfc), 5)
        sample = pfc.sample_for_context(budget=3)
        # newest events should dominate
        self.assertEqual(sample[0], "event 5")


class TestHippocampus(unittest.TestCase):
    def test_consolidate_and_recall(self):
        hpc = Hippocampus(similarity_threshold=0.5)  # lenient for test
        emb_a = np.array([1.0, 0.0])
        emb_b = np.array([0.9, 0.1])
        ev = HpcEvent(id="a", content="alpha", embedding=emb_a,
                      scores={"error_severity":5, "novelty":5, "foundation_weight":5, "rlhf_weight":8},
                      class_="Foundation", tags=[])
        accepted = hpc.consolidate(ev)
        self.assertTrue(accepted)
        results = hpc.recall(query_embedding=emb_b, top_k=1)
        self.assertEqual(results[0].id, "a")


class TestParietalGraph(unittest.TestCase):
    def test_upsert_and_nearest(self):
        pg = ParietalGraph(dedupe_threshold=0.8)
        ev = Evidence(source="doc", snippet="foo", confidence=0.9, timestamp="2025-01-01T00:00:00Z")
        pg.upsert_triplet("Cat", "likes", "Milk", ev)
        facts = pg.nearest_facts("Cat", k=1)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].t, "Milk")


if __name__ == "__main__":
    unittest.main()

Eval Retrieval.py2:
"""Minimal retrieval evaluation script.

Usage::
    python eval_retrieval.py

Generates synthetic events, consolidates some into Hippocampus, and computes
R@5 + MRR over a toy QA dataset.
"""
from __future__ import annotations

import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np

from hippocampus import Hippocampus, MemoryEvent as HpcEvent
from pfc import PrefrontalCache, MemoryEvent as PfcEvent
from parietal import ParietalGraph, Evidence


def recall_at_k(results, answers, k):
    hits = sum(1 for res, ans in zip(results, answers) if any(a in res[:k] for a in ans))
    return hits / len(results)


def mrr(results, answers):
    rr_total = 0.0
    for res, ans in zip(results, answers):
        rank = next((i + 1 for i, r in enumerate(res) if any(a == r for a in ans)), None)
        if rank:
            rr_total += 1 / rank
    return rr_total / len(results)


def synthetic_embeddings(n, dim=8):
    vecs = np.random.randn(n, dim).astype(float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms


def main():
    # Init subsystems
    pfc = PrefrontalCache()
    hpc = Hippocampus()
    pg = ParietalGraph()

    # Synthetic dataset — 10 Q/A pairs
    qa = [
        {"query": "alpha", "answers": ["alpha"], "embedding": np.array([1,0,0,0,0,0,0,0])},
        {"query": "bravo", "answers": ["bravo"], "embedding": np.array([0,1,0,0,0,0,0,0])},
    ]

    # Consolidate events into Hippocampus
    now = datetime.utcnow()
    for i, item in enumerate(qa):
        emb = item["embedding"]
        ev = HpcEvent(id=str(i), content=item["answers"][0], embedding=emb,
                      scores={"error_severity":5, "novelty":5, "foundation_weight":5, "rlhf_weight":8},
                      class_="Foundation", tags=[])
        hpc.consolidate(ev)

    # Run recall for each query embedding
    results = []
    answers = [item["answers"] for item in qa]
    for item in qa:
        res_events = hpc.recall(query_embedding=item["embedding"], top_k=5)
        results.append([e.content for e in res_events])

    r5 = recall_at_k(results, answers, k=5)
    m = mrr(results, answers)

    report = {"R@5": r5, "MRR": m}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
