from __future__ import annotations

"""Context composer orchestrating the three memory subsystems."""

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .hippocampus import Hippocampus
from .parietal import ParietalGraph
from .pfc import PrefrontalCache
from .types import AblationConfig, Retrieval


@dataclass(slots=True)
class ComposerResult:
    query: str
    segments: Dict[str, Retrieval]

    def merged_context(self) -> List[str]:
        ordered = ["pfc", "kg", "ltm"]
        parts: List[str] = []
        for key in ordered:
            retrieval = self.segments.get(key)
            if retrieval:
                parts.extend(retrieval.items)
        return parts


class ContextComposer:
    def __init__(
        self,
        *,
        pfc: PrefrontalCache,
        hippocampus: Hippocampus,
        parietal: ParietalGraph,
        ablations: AblationConfig | None = None,
    ) -> None:
        self.pfc = pfc
        self.hippocampus = hippocampus
        self.parietal = parietal
        self.ablations = ablations or AblationConfig()

    def compose(
        self,
        *,
        query: str,
        query_embedding: Sequence[float] | None,
        budget: int = 8,
        kg_k: int = 5,
        ltm_k: int = 5,
    ) -> ComposerResult:
        segments: Dict[str, Retrieval] = {}

        pfc_items = self.pfc.sample_for_context(budget)
        segments["pfc"] = Retrieval(source="pfc", items=pfc_items, metadata={"budget": budget})

        if not self.ablations.only_pfc and not self.ablations.no_kg:
            kg_facts = [
                f"{fact.h} {fact.r} {fact.t}" for fact in self.parietal.nearest_facts(query, kg_k)
            ]
            segments["kg"] = Retrieval(source="kg", items=kg_facts, metadata={"k": kg_k})

        if not self.ablations.only_pfc and not self.ablations.no_ltm and query_embedding is not None:
            events = self.hippocampus.recall(query_embedding, top_k=ltm_k)
            segments["ltm"] = Retrieval(
                source="ltm",
                items=[e.content for e in events],
                metadata={"k": ltm_k},
            )

        return ComposerResult(query=query, segments=segments)
