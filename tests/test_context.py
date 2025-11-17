from __future__ import annotations

import unittest
from datetime import datetime

from sdk.core.context import ContextComposer
from sdk.core.hippocampus import Hippocampus
from sdk.core.parietal import Evidence, ParietalGraph
from sdk.core.pfc import PrefrontalCache
from sdk.core.types import ComposerConfig, MemoryEvent


class DummyPFC(PrefrontalCache):
    def __init__(self):
        super().__init__(max_events=5)
        self.calls = 0

    def sample_for_context(self, budget: int = 8):  # type: ignore[override]
        self.calls += 1
        return [f"pfc-{i}" for i in range(min(budget, len(self._buf)))]


class DummyHippocampus(Hippocampus):
    def __init__(self):
        super().__init__(similarity_threshold=0.0, top_k=5)
        self.calls = 0

    def recall(self, query_embedding, top_k=None):  # type: ignore[override]
        self.calls += 1
        return list(self._events)


class DummyParietal(ParietalGraph):
    def __init__(self):
        super().__init__(dedupe_threshold=0.0)
        self.calls = 0

    def nearest_facts(self, entity_or_term: str, k: int = 8):  # type: ignore[override]
        self.calls += 1
        return super().nearest_facts(entity_or_term, k=k)


class TestContextComposer(unittest.TestCase):
    def _mk_event(self, idx: int) -> MemoryEvent:
        return MemoryEvent(
            id=str(idx),
            timestamp=datetime.utcnow(),
            content=f"memory-{idx}",
            event_class="Foundation",
            scores={"error_severity": 6, "novelty": 6, "foundation_weight": 6, "rlhf_weight": 6},
            importance=0.8,
            novelty=0.4,
            embedding=[1.0, 0.0, 0.0],
        )

    def setUp(self) -> None:
        self.pfc = DummyPFC()
        self.hippocampus = DummyHippocampus()
        self.parietal = DummyParietal()

        evid = Evidence(source="unit-test")
        self.parietal.upsert_triplet("Alice", "WORKS_ON", "Atlas", evid)

        event = MemoryEvent(
            id="1",
            timestamp=datetime.utcnow(),
            content="Atlas overview",
            event_class="Foundation",
            scores={"error_severity": 6, "novelty": 6, "foundation_weight": 6, "rlhf_weight": 6},
            importance=0.8,
            novelty=0.4,
            embedding=[1.0, 0.0, 0.0],
        )
        self.hippocampus.consolidate(event)
        self.pfc.push(event)

    def test_full_composition_includes_all_sources(self):
        composer = ContextComposer(
            pfc=self.pfc,
            hippocampus=self.hippocampus,
            parietal=self.parietal,
            config=ComposerConfig(),
        )
        chunks = composer.compose(query_embedding=[1.0, 0.0, 0.0], entity_hint="Alice")
        sources = {chunk.source for chunk in chunks}
        self.assertEqual(sources, {"pfc", "kg", "ltm"})

    def test_no_ltm_flag_prevents_recall(self):
        composer = ContextComposer(
            pfc=self.pfc,
            hippocampus=self.hippocampus,
            parietal=self.parietal,
            config=ComposerConfig.from_flags(no_ltm=True),
        )
        composer.compose(query_embedding=[1.0, 0.0, 0.0], entity_hint="Alice")
        self.assertEqual(self.hippocampus.calls, 0)

    def test_no_kg_flag_prevents_graph_lookup(self):
        composer = ContextComposer(
            pfc=self.pfc,
            hippocampus=self.hippocampus,
            parietal=self.parietal,
            config=ComposerConfig.from_flags(no_kg=True),
        )
        composer.compose(query_embedding=[1.0, 0.0, 0.0], entity_hint="Alice")
        self.assertEqual(self.parietal.calls, 0)

    def test_only_pfc_flag_short_circuits_other_sources(self):
        composer = ContextComposer(
            pfc=self.pfc,
            hippocampus=self.hippocampus,
            parietal=self.parietal,
            config=ComposerConfig.from_flags(only_pfc=True),
        )
        chunks = composer.compose(query_embedding=[1.0, 0.0, 0.0], entity_hint="Alice")
        self.assertTrue(all(chunk.source == "pfc" for chunk in chunks))
        self.assertEqual(self.parietal.calls, 0)
        self.assertEqual(self.hippocampus.calls, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

