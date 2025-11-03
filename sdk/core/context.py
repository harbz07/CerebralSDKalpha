"""Context composition utilities for the Cerebral SDK memory triad."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .hippocampus import Hippocampus
from .parietal import Fact, ParietalGraph
from .pfc import PrefrontalCache
from .types import ComposerConfig, MemoryEvent


@dataclass(slots=True)
class ContextChunk:
    """A lightweight view of content returned by the composer."""

    source: str
    content: str
    metadata: dict[str, object]


class ContextComposer:
    """Aggregate content from the three memory subsystems.

    The class exposes toggles corresponding to the RFC ablation flags.  PFC
    contributions always lead, followed by KG facts and finally LTM recall.  The
    public API intentionally keeps types simple so that scripts and tests can
    exercise the logic without heavy dependencies.
    """

    def __init__(
        self,
        *,
        pfc: Optional[PrefrontalCache] = None,
        hippocampus: Optional[Hippocampus] = None,
        parietal: Optional[ParietalGraph] = None,
        config: Optional[ComposerConfig] = None,
    ) -> None:
        self.pfc = pfc or PrefrontalCache()
        self.hippocampus = hippocampus or Hippocampus()
        self.parietal = parietal or ParietalGraph()
        self.config = config or ComposerConfig()

    def compose(
        self,
        *,
        query_embedding: Optional[Iterable[float]] = None,
        entity_hint: Optional[str] = None,
        pfc_budget: int = 8,
        kg_budget: int = 6,
        ltm_budget: int = 5,
    ) -> List[ContextChunk]:
        """Return ordered context chunks with provenance metadata."""

        chunks: List[ContextChunk] = []

        if self.config.use_pfc:
            for content in self.pfc.sample_for_context(budget=pfc_budget):
                chunks.append(
                    ContextChunk(
                        source="pfc",
                        content=content,
                        metadata={"budget": pfc_budget},
                    )
                )

        if self.config.use_kg and entity_hint:
            facts = self.parietal.nearest_facts(entity_hint, k=kg_budget)
            for fact in facts:
                snippet = f"{fact.h} -[{fact.r}]-> {fact.t}"
                evidence = fact.evidence.__dict__ if fact.evidence else {}
                chunks.append(
                    ContextChunk(
                        source="kg",
                        content=snippet,
                        metadata={"evidence": evidence},
                    )
                )

        if self.config.use_ltm and query_embedding is not None:
            memories = self.hippocampus.recall(query_embedding, top_k=ltm_budget)
            for memory in memories:
                chunks.append(
                    ContextChunk(
                        source="ltm",
                        content=memory.content,
                        metadata={"id": memory.id},
                    )
                )

        return chunks

    def consolidate(self, event: MemoryEvent) -> bool:
        """Proxy convenience method for long-term memory consolidation."""

        return self.hippocampus.consolidate(event)

