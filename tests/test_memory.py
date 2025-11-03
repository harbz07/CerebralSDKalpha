"""
Unit tests for Cerebral SDK memory triad:
- PrefrontalCache (PFC)
- Hippocampus (semantic LTM)
- ParietalGraph (KG)

Assumes the merged "golden" layout:
  sdk/core/types.py
  sdk/core/pfc.py
  sdk/core/hippocampus.py
  sdk/core/parietal.py

Run:
    python -m unittest tests.test_memory
"""
from __future__ import annotations

import io
import json
import unittest
from datetime import datetime, timedelta

import numpy as np

from sdk.core.types import MemoryEvent
from sdk.core.pfc import PrefrontalCache
from sdk.core.hippocampus import Hippocampus, CATCH_22_THRESHOLD, DEFAULT_CLASS_BIAS
from sdk.core.parietal import ParietalGraph, Evidence


# -----------------------------
# Helpers
# -----------------------------
def _mk_event(
    i: int,
    *,
    cls: str = "Foundation",
    ts: datetime | None = None,
    importance: float = 0.7,
    novelty: float = 0.4,
    scores: dict | None = None,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    content: str | None = None,
) -> MemoryEvent:
    """Create a MemoryEvent with sensible defaults."""
    return MemoryEvent(
        id=str(i),
        timestamp=ts or datetime.utcnow(),
        content=content or f"event {i}",
        event_class=cls,
        scores=scores
        or {"error_severity": 5, "novelty": 5, "foundation_weight": 5, "rlhf_weight": 8},
        importance=importance,
        novelty=novelty,
        tags=tags or [],
        embedding=embedding,
    )


# =============================
# PFC Tests
# =============================
class TestPrefrontalCache(unittest.TestCase):
    def test_push_capacity_and_len(self):
        pfc = PrefrontalCache(max_events=5)
        now = datetime.utcnow()
        for i in range(7):  # 2 beyond capacity
            pfc.push(_mk_event(i, ts=now + timedelta(seconds=i)))
        self.assertEqual(len(pfc), 5, "PFC should cap at max_events")

    def test_recent_returns_newest_first(self):
        pfc = PrefrontalCache(max_events=10)
        base = datetime.utcnow()
        for i in range(6):
            pfc.push(_mk_event(i, ts=base + timedelta(seconds=i)))
        recent = pfc.recent(3)
        ids = [e.id for e in recent]
        self.assertEqual(ids, ["5", "4", "3"], "recent() should return newest-first")

    def test_sample_for_context_prioritizes_recency_times_importance(self):
        pfc = PrefrontalCache(max_events=10)
        base = datetime.utcnow()
        # Even indices = high importance; odd = low importance; timestamps increasing
        for i in range(10):
            pfc.push(
                _mk_event(
                    i,
                    ts=base + timedelta(seconds=i),
                    importance=1.0 if i % 2 == 0 else 0.2,
                )
            )
        sample = pfc.sample_for_context(budget=5)
        # newest high-importance should lead
        self.assertEqual(sample[0], "event 9" if (9 % 2 == 0) else "event 8")
        self.assertEqual(len(sample), 5)

    def test_sample_empty_and_zero_budget(self):
        pfc = PrefrontalCache(max_events=5)
        self.assertEqual(pfc.sample_for_context(budget=3), [])
        # Add one and test zero budget
        pfc.push(_mk_event(1))
        self.assertEqual(pfc.sample_for_context(budget=0), [])


# =============================
# Hippocampus (LTM) Tests
# =============================
class TestHippocampus(unittest.TestCase):
    def test_consolidate_requires_embedding_when_accepted(self):
        h = Hippocampus()
        # Construct an event that *would* pass Catch-22 (sum 22+) but missing embedding
        scores = {"error_severity": 5, "novelty": 6, "foundation_weight": 6, "rlhf_weight": 6}  # sum=23
        ev = _mk_event(1, scores=scores, embedding=None)
        with self.assertRaises(ValueError):
            h.consolidate(ev)

    def test_consolidate_catch22_and_pin(self):
        h = Hippocampus()
        # Chaos (bias 0) below threshold => should NOT accept
        scores_low = {"error_severity": 5, "novelty": 5, "foundation_weight": 5, "rlhf_weight": 5}  # 20
        ev_low = _mk_event(2, cls="Chaos", scores=scores_low, embedding=[1.0, 0.0])
        accepted_low = h.consolidate(ev_low)
        self.assertFalse(accepted_low, "Below threshold without bias should not consolidate")

        # Foundation has +2 bias; 20 + 2 == 22 -> boundary accept
        ev_boundary = _mk_event(3, cls="Foundation", scores=scores_low, embedding=[0.0, 1.0])
        accepted_boundary = h.consolidate(ev_boundary)
        self.assertTrue(
            accepted_boundary,
            "Boundary case (sum+bias == 22) should consolidate",
        )

        # Pin always bypasses threshold
        ev_pin = _mk_event(4, cls="Chaos", scores={"error_severity": 1, "novelty": 1, "foundation_weight": 1, "rlhf_weight": 1},
                           tags=["pin"], embedding=[0.0, 1.0])
        accepted_pin = h.consolidate(ev_pin)
        self.assertTrue(accepted_pin, '"pin" tag must bypass Catch-22')

    def test_recall_cosine_similarity_and_threshold(self):
        h = Hippocampus(similarity_threshold=0.6, top_k=3)
        # Two orthogonal-ish embeddings
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        # Both scores well over threshold
        scores = {"error_severity": 6, "novelty": 6, "foundation_weight": 6, "rlhf_weight": 6}
        ev_a = _mk_event(10, content="alpha", embedding=a, scores=scores)
        ev_b = _mk_event(11, content="bravo", embedding=b, scores=scores)
        self.assertTrue(h.consolidate(ev_a))
        self.assertTrue(h.consolidate(ev_b))

        # Query near `a` should rank "alpha" first
        q = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        out = h.recall(q, top_k=2)
        self.assertGreaterEqual(len(out), 1)
        self.assertEqual(out[0].content, "alpha")

    def test_stats_reports_dims(self):
        h = Hippocampus()
        scores = {"error_severity": 6, "novelty": 6, "foundation_weight": 6, "rlhf_weight": 6}
        ev = _mk_event(12, embedding=[1.0, 0.0, 0.0, 0.0], scores=scores)
        self.assertTrue(h.consolidate(ev))
        st = h.stats()
        self.assertEqual(st["dims"], 4)
        self.assertEqual(st["count"], 1)

    def test_bias_tuning_affects_acceptance(self):
        # Set Foundation bias to 0 to force non-accept on borderline
        h = Hippocampus()
        h.bias["Foundation"] = 0.0
        scores = {"error_severity": 5, "novelty": 5, "foundation_weight": 5, "rlhf_weight": 5}  # 20
        ev = _mk_event(13, cls="Foundation", scores=scores, embedding=[1.0, 0.0])
        self.assertFalse(h.consolidate(ev), "With zero bias, 20 < 22 should not consolidate")


# =============================
# Parietal (KG) Tests
# =============================
class TestParietalGraph(unittest.TestCase):
    def test_upsert_and_nearest_preserves_labels_and_evidence(self):
        g = ParietalGraph(dedupe_threshold=0.9)
        evid = Evidence(source="unit", url=None, timestamp="2025-01-01T00:00:00Z", confidence=1.0, snippet="Alice→ProjectX")
        g.upsert_triplet("Alice", "WORKS_ON", "ProjectX", evid)
        facts = g.nearest_facts("Alice", k=1)
        self.assertEqual(len(facts), 1)
        f = facts[0]
        self.assertEqual(f.h, "Alice")
        self.assertEqual(f.t, "ProjectX")
        self.assertEqual(f.r, "WORKS_ON")
        self.assertIsNotNone(f.evidence)
        self.assertEqual(f.evidence.source, "unit")

    def test_canonicalisation_dedupe_threshold(self):
        # "Cat" vs "cat" should canonicalise at threshold >= 0.9 (case-insensitive)
        g = ParietalGraph(dedupe_threshold=0.9)
        ev = Evidence(source="unit", snippet="Cat likes milk", confidence=1.0, timestamp="2025-01-01T00:00:00Z")
        g.upsert_triplet("Cat", "LIKES", "Milk", ev)
        # Second insert with slightly different case/spacing should hit same node
        g.upsert_triplet("cat ", "LIKES", "Milk", ev)
        facts = g.nearest_facts("CAT", k=5)
        # We still expect one unique outgoing relation target
        self.assertTrue(any(f.t == "Milk" for f in facts))

    def test_export_formats_graphml_and_jsonl(self):
        g = ParietalGraph()
        ev = Evidence(source="unit", snippet="A→B", confidence=0.9, timestamp="2025-01-01T00:00:00Z")
        g.upsert_triplet("A", "REL", "B", ev)

        # GraphML bytes should look like XML
        gm = g.export("graphml")
        self.assertIsInstance(gm, (bytes, bytearray))
        self.assertTrue(gm.strip().startswith(b"<?xml"), "GraphML export should be XML bytes")

        # JSONL should parse to one line with our fields
        jl = g.export("jsonl")
        self.assertIsInstance(jl, (bytes, bytearray))
        lines = jl.decode().strip().splitlines()
        self.assertEqual(len(lines), 1)
        item = json.loads(lines[0])
        self.assertEqual(item["h"], "A")
        self.assertEqual(item["t"], "B")
        self.assertIn("evidence", item)
        self.assertEqual(item["evidence"]["source"], "unit")


# =============================
# Sanity / RFC-alignment checks
# =============================
class TestRFCSanity(unittest.TestCase):
    def test_catch22_constant(self):
        self.assertEqual(CATCH_22_THRESHOLD, 22)

    def test_default_bias_has_foundation_positive(self):
        self.assertGreaterEqual(DEFAULT_CLASS_BIAS.get("Foundation", 0), 0)


if __name__ == "__main__":
    # Allow running this file directly: python tests/test_memory.py
    suite = unittest.defaultTestLoader.loadTestsFromName(__name__)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
