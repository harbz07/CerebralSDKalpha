"""Unit tests for Cerebral SDK memory triad and evaluation harness."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from sdk.core.composer import ContextComposer
from sdk.core.hippocampus import Hippocampus, CATCH_22_THRESHOLD, DEFAULT_CLASS_BIAS
from sdk.core.parietal import ParietalGraph, Evidence
from sdk.core.pfc import PrefrontalCache
from sdk.core.types import AblationConfig, MemoryEvent
from sdk.harness.datasets import load_qa_dataset
from sdk.harness.metrics import mean_reciprocal_rank, recall_at_k
from sdk.harness.run import run_suite
from sdk.harness.synthetic import generate_events


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


class TestPrefrontalCache(unittest.TestCase):
    def test_push_capacity_and_len(self):
        pfc = PrefrontalCache(max_events=5)
        now = datetime.utcnow()
        for i in range(7):
            pfc.push(_mk_event(i, ts=now + timedelta(seconds=i)))
        self.assertEqual(len(pfc), 5)

    def test_recent_returns_newest_first(self):
        pfc = PrefrontalCache(max_events=10)
        base = datetime.utcnow()
        for i in range(6):
            pfc.push(_mk_event(i, ts=base + timedelta(seconds=i)))
        recent = pfc.recent(3)
        ids = [e.id for e in recent]
        self.assertEqual(ids, ["5", "4", "3"])

    def test_sample_for_context_prioritizes_recency_times_importance(self):
        pfc = PrefrontalCache(max_events=10)
        base = datetime.utcnow()
        for i in range(10):
            pfc.push(
                _mk_event(
                    i,
                    ts=base + timedelta(seconds=i),
                    importance=1.0 if i % 2 == 0 else 0.2,
                )
            )
        sample = pfc.sample_for_context(budget=5)
        self.assertEqual(len(sample), 5)
        self.assertIn(sample[0], {"event 8", "event 9"})

    def test_sample_empty_and_zero_budget(self):
        pfc = PrefrontalCache(max_events=5)
        self.assertEqual(pfc.sample_for_context(budget=3), [])
        pfc.push(_mk_event(1))
        self.assertEqual(pfc.sample_for_context(budget=0), [])


class TestHippocampus(unittest.TestCase):
    def test_consolidate_requires_embedding_when_accepted(self):
        h = Hippocampus()
        scores = {"error_severity": 5, "novelty": 6, "foundation_weight": 6, "rlhf_weight": 6}
        ev = _mk_event(1, scores=scores, embedding=None)
        with self.assertRaises(ValueError):
            h.consolidate(ev)

    def test_consolidate_catch22_and_pin(self):
        h = Hippocampus()
        scores_low = {"error_severity": 5, "novelty": 5, "foundation_weight": 5, "rlhf_weight": 5}
        ev_low = _mk_event(2, cls="Chaos", scores=scores_low, embedding=[1.0, 0.0])
        accepted_low = h.consolidate(ev_low)
        self.assertFalse(accepted_low)

        ev_boundary = _mk_event(3, cls="Foundation", scores=scores_low, embedding=[0.0, 1.0])
        accepted_boundary = h.consolidate(ev_boundary)
        self.assertTrue(accepted_boundary)

        ev_pin = _mk_event(4, cls="Chaos", scores={"error_severity": 1, "novelty": 1, "foundation_weight": 1, "rlhf_weight": 1},
                           tags=["pin"], embedding=[0.0, 1.0])
        accepted_pin = h.consolidate(ev_pin)
        self.assertTrue(accepted_pin)

    def test_recall_cosine_similarity_and_threshold(self):
        h = Hippocampus(similarity_threshold=0.6, top_k=3)
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        scores = {"error_severity": 6, "novelty": 6, "foundation_weight": 6, "rlhf_weight": 6}
        ev_a = _mk_event(10, content="alpha", embedding=a, scores=scores)
        ev_b = _mk_event(11, content="bravo", embedding=b, scores=scores)
        self.assertTrue(h.consolidate(ev_a))
        self.assertTrue(h.consolidate(ev_b))

        q = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        out = h.recall(q, top_k=2)
        self.assertGreaterEqual(len(out), 1)
        self.assertEqual(out[0].content, "alpha")

    def test_stats_reports_dims_and_sqlite_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = f"{tmp}/ltm.db"
            scores = {"error_severity": 6, "novelty": 6, "foundation_weight": 6, "rlhf_weight": 6}
            h = Hippocampus(storage_path=db)
            event = _mk_event(12, embedding=[1.0, 0.0, 0.0, 0.0], scores=scores)
            self.assertTrue(h.consolidate(event))
            stats = h.stats()
            self.assertEqual(stats["dims"], 4)
            self.assertEqual(stats["count"], 1)

            h2 = Hippocampus(storage_path=db)
            recalled = h2.recall([1.0, 0.0, 0.0, 0.0], top_k=1)
            self.assertEqual(len(recalled), 1)
            self.assertEqual(recalled[0].id, event.id)

    def test_bias_tuning_affects_acceptance(self):
        h = Hippocampus()
        h.bias["Foundation"] = 0.0
        scores = {"error_severity": 5, "novelty": 5, "foundation_weight": 5, "rlhf_weight": 5}
        ev = _mk_event(13, cls="Foundation", scores=scores, embedding=[1.0, 0.0])
        self.assertFalse(h.consolidate(ev))


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
        g = ParietalGraph(dedupe_threshold=0.9)
        ev = Evidence(source="unit", snippet="Cat likes milk", confidence=1.0, timestamp="2025-01-01T00:00:00Z")
        g.upsert_triplet("Cat", "LIKES", "Milk", ev)
        g.upsert_triplet("cat ", "LIKES", "Milk", ev)
        facts = g.nearest_facts("CAT", k=5)
        self.assertTrue(any(f.t == "Milk" for f in facts))

    def test_export_formats_graphml_and_jsonl(self):
        g = ParietalGraph()
        ev = Evidence(source="unit", snippet="A→B", confidence=0.9, timestamp="2025-01-01T00:00:00Z")
        g.upsert_triplet("A", "REL", "B", ev)

        gm = g.export("graphml")
        self.assertIsInstance(gm, (bytes, bytearray))
        self.assertTrue(gm.strip().startswith(b"<?xml"))

        jl = g.export("jsonl")
        lines = jl.decode().strip().splitlines()
        self.assertEqual(len(lines), 1)
        item = json.loads(lines[0])
        self.assertEqual(item["h"], "A")
        self.assertEqual(item["t"], "B")
        self.assertEqual(item["evidence"]["source"], "unit")


class TestContextComposer(unittest.TestCase):
    def test_ablation_flags(self):
        pfc = PrefrontalCache(max_events=4)
        for event in generate_events(4):
            pfc.push(event)
        hippocampus = Hippocampus()
        for event in generate_events(2, seed=1):
            hippocampus.consolidate(event)
        parietal = ParietalGraph()
        parietal.upsert_triplet("Alice", "WORKS_ON", "Alpha", None)

        composer = ContextComposer(pfc=pfc, hippocampus=hippocampus, parietal=parietal, ablations=AblationConfig())
        result = composer.compose(query="Alice", query_embedding=[1.0, 0.0, 0.0, 0.0])
        self.assertIn("pfc", result.segments)
        self.assertIn("ltm", result.segments)
        self.assertIn("kg", result.segments)

        only = ContextComposer(pfc=pfc, hippocampus=hippocampus, parietal=parietal,
                               ablations=AblationConfig.from_flags(only_pfc=True))
        result_only = only.compose(query="Alice", query_embedding=[1.0, 0.0, 0.0, 0.0])
        self.assertEqual(set(result_only.segments), {"pfc"})

    def test_ablation_flag_validation(self):
        with self.assertRaises(ValueError):
            AblationConfig.from_flags(no_kg=True, only_pfc=True)


class TestHarness(unittest.TestCase):
    def test_metrics(self):
        results = [["a", "b"], ["c"], []]
        answers = [["a"], ["c"], ["d"]]
        self.assertAlmostEqual(recall_at_k(results, answers, 1), 2 / 3)
        self.assertAlmostEqual(mean_reciprocal_rank(results, answers), (1 + 1 + 0) / 3)

    def test_dataset_loader(self):
        qa = load_qa_dataset()
        self.assertGreaterEqual(len(qa), 3)
        self.assertEqual(qa[0].answers[0], "Alpha project is owned by Alice")

    def test_run_suite_default_thresholds(self):
        metrics = run_suite(AblationConfig())
        self.assertGreaterEqual(metrics["Recall@5"], 0.60)
        self.assertGreaterEqual(metrics["MRR"], 0.40)


class TestRFCSanity(unittest.TestCase):
    def test_catch22_constant(self):
        self.assertEqual(CATCH_22_THRESHOLD, 22)

    def test_default_bias_has_foundation_positive(self):
        self.assertGreaterEqual(DEFAULT_CLASS_BIAS.get("Foundation", 0), 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromName(__name__)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
