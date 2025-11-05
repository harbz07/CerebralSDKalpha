"""
Cerebral Memory System - Integration layer for PFC, Parietal, and Hippocampus.

This module provides a unified interface to orchestrate the three core memory subsystems:
- PFC (Prefrontal Cache): Working memory for recent events
- Parietal Graph: Knowledge graph for structured facts and relationships
- Hippocampus: Long-term vector memory with semantic recall

Usage example:
    from sdk.core.cerebral import CerebralMemory
    from sdk.core.types import MemoryEvent

    memory = CerebralMemory()
    memory.ingest(event)  # Automatically routes to appropriate subsystems
    context = memory.retrieve_context(query_embedding)
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from .types import MemoryEvent
from .pfc import PrefrontalCache
from .parietal import ParietalGraph, Evidence, Fact
from .hippocampus import Hippocampus


class CerebralMemory:
    """Unified memory system integrating PFC, Parietal Graph, and Hippocampus.

    This class provides:
    - Automatic event routing to appropriate subsystems
    - Hybrid retrieval combining all three memory types
    - Cross-module communication and synchronization
    - Unified statistics and maintenance
    """

    def __init__(
        self,
        pfc_capacity: int = 1000,
        pfc_decay_halflife: Optional[timedelta] = None,
        hippocampus_threshold: float = 0.80,
        hippocampus_top_k: int = 5,
        hippocampus_temporal_decay: bool = False,
        parietal_dedupe_threshold: float = 0.90
    ):
        """Initialize the Cerebral Memory System.

        Args:
            pfc_capacity: Maximum events in working memory
            pfc_decay_halflife: Temporal decay for PFC (None = no decay)
            hippocampus_threshold: Similarity threshold for LTM recall
            hippocampus_top_k: Default number of LTM results
            hippocampus_temporal_decay: Enable temporal decay for LTM
            parietal_dedupe_threshold: Entity canonicalization threshold
        """
        self.pfc = PrefrontalCache(
            max_events=pfc_capacity,
            decay_halflife=pfc_decay_halflife
        )

        self.hippocampus = Hippocampus(
            similarity_threshold=hippocampus_threshold,
            top_k=hippocampus_top_k,
            temporal_decay=hippocampus_temporal_decay
        )

        self.parietal = ParietalGraph(
            dedupe_threshold=parietal_dedupe_threshold
        )

    def ingest(
        self,
        event: MemoryEvent,
        extract_facts: bool = True,
        fact_extractor: Optional[callable] = None
    ) -> Dict[str, bool]:
        """Ingest an event into the memory system.

        This automatically:
        1. Pushes to PFC (working memory)
        2. Attempts consolidation to Hippocampus (long-term memory)
        3. Optionally extracts facts for Parietal graph

        Args:
            event: MemoryEvent to ingest
            extract_facts: Whether to extract facts for knowledge graph
            fact_extractor: Custom function to extract (h, r, t, evidence) tuples

        Returns:
            Dict indicating which subsystems accepted the event
        """
        results = {
            "pfc": False,
            "hippocampus": False,
            "parietal": False
        }

        # Always push to PFC (working memory)
        self.pfc.push(event)
        results["pfc"] = True

        # Attempt consolidation to Hippocampus (LTM)
        try:
            if self.hippocampus.consolidate(event):
                results["hippocampus"] = True
        except ValueError:
            # Event lacks embedding, skip hippocampus
            pass

        # Extract facts for Parietal graph if requested
        if extract_facts:
            if fact_extractor:
                facts = fact_extractor(event)
                for h, r, t, ev in facts:
                    self.parietal.upsert_triplet(h, r, t, ev)
                results["parietal"] = len(facts) > 0
            else:
                # Default: extract from metadata if available
                if "facts" in event.metadata:
                    for fact_data in event.metadata["facts"]:
                        h = fact_data.get("h")
                        r = fact_data.get("r")
                        t = fact_data.get("t")
                        if h and r and t:
                            evidence = Evidence(
                                source=event.id,
                                timestamp=event.timestamp.isoformat(),
                                confidence=fact_data.get("confidence", 1.0),
                                snippet=event.content[:100]
                            )
                            self.parietal.upsert_triplet(h, r, t, evidence)
                            results["parietal"] = True

        return results

    def ingest_batch(
        self,
        events: List[MemoryEvent],
        extract_facts: bool = True
    ) -> Dict[str, int]:
        """Ingest multiple events efficiently.

        Returns:
            Statistics about ingestion (counts per subsystem)
        """
        stats = {"pfc": 0, "hippocampus": 0, "parietal": 0}

        for event in events:
            results = self.ingest(event, extract_facts=extract_facts)
            for subsys, accepted in results.items():
                if accepted:
                    stats[subsys] += 1

        return stats

    def retrieve_context(
        self,
        query_embedding: Optional[List[float] | np.ndarray] = None,
        query_text: Optional[str] = None,
        pfc_budget: int = 5,
        ltm_top_k: int = 3,
        kg_facts_k: int = 5,
        filter_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Retrieve context from all three memory subsystems.

        Args:
            query_embedding: Vector for semantic search (required for LTM)
            query_text: Text for knowledge graph entity matching
            pfc_budget: Number of working memory events to retrieve
            ltm_top_k: Number of long-term memories to retrieve
            kg_facts_k: Number of knowledge graph facts to retrieve
            filter_tags: Optional tag filter

        Returns:
            Dict with keys: 'working_memory', 'long_term', 'knowledge_graph'
        """
        context = {
            "working_memory": [],
            "long_term": [],
            "knowledge_graph": []
        }

        # Retrieve from PFC (working memory)
        if filter_tags:
            context["working_memory"] = self.pfc.sample_for_context(
                budget=pfc_budget,
                filter_tags=filter_tags
            )
        else:
            context["working_memory"] = self.pfc.sample_for_context(budget=pfc_budget)

        # Retrieve from Hippocampus (long-term memory)
        if query_embedding is not None:
            ltm_events = self.hippocampus.recall(
                query_embedding=query_embedding,
                top_k=ltm_top_k,
                filter_tags=filter_tags
            )
            context["long_term"] = [ev.content for ev in ltm_events]

        # Retrieve from Parietal (knowledge graph)
        if query_text:
            facts = self.parietal.nearest_facts(query_text, k=kg_facts_k)
            context["knowledge_graph"] = [
                f"{f.h} --[{f.r}]--> {f.t}" for f in facts
            ]

        return context

    def find_reasoning_path(
        self,
        start_entity: str,
        end_entity: str,
        max_length: int = 3
    ) -> List[str]:
        """Find reasoning paths in the knowledge graph.

        Returns:
            List of path descriptions
        """
        paths = self.parietal.find_paths(start_entity, end_entity, max_length=max_length)
        return [str(path) for path in paths]

    def get_entity_context(
        self,
        entity: str,
        include_neighbors: bool = True
    ) -> Dict[str, List[str]]:
        """Get comprehensive context about an entity from knowledge graph.

        Returns:
            Dict with 'outgoing' and 'incoming' fact lists
        """
        outgoing, incoming = self.parietal.get_facts_bidirectional(entity)

        return {
            "outgoing": [f"{f.h} --[{f.r}]--> {f.t}" for f in outgoing],
            "incoming": [f"{f.h} --[{f.r}]--> {f.t}" for f in incoming]
        }

    def consolidate_working_to_longterm(
        self,
        tag_filter: Optional[List[str]] = None,
        min_importance: float = 0.5
    ) -> int:
        """Manually consolidate high-importance events from PFC to Hippocampus.

        Useful for periodic maintenance or before clearing working memory.

        Returns:
            Number of events successfully consolidated
        """
        if tag_filter:
            candidates = self.pfc.by_tags(tag_filter)
        else:
            candidates = list(self.pfc._buf)

        consolidated = 0
        for event in candidates:
            if event.importance >= min_importance and event.embedding is not None:
                try:
                    if self.hippocampus.consolidate(event):
                        consolidated += 1
                except ValueError:
                    pass

        return consolidated

    def maintenance(
        self,
        forget_ltm_older_than: Optional[timedelta] = None,
        forget_pfc_older_than: Optional[timedelta] = None,
        forget_low_importance: Optional[float] = None
    ) -> Dict[str, int]:
        """Perform memory maintenance operations.

        Args:
            forget_ltm_older_than: Remove LTM events older than this
            forget_pfc_older_than: Remove PFC events older than this
            forget_low_importance: Remove LTM events below this importance

        Returns:
            Statistics about removed events
        """
        stats = {"ltm_removed": 0, "pfc_removed": 0}

        if forget_ltm_older_than:
            stats["ltm_removed"] += self.hippocampus.forget_old(forget_ltm_older_than)

        if forget_low_importance:
            stats["ltm_removed"] += self.hippocampus.forget_low_importance(forget_low_importance)

        if forget_pfc_older_than:
            stats["pfc_removed"] = self.pfc.clear_old(forget_pfc_older_than)

        return stats

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all subsystems."""
        return {
            "pfc": self.pfc.stats(),
            "hippocampus": self.hippocampus.stats(),
            "parietal": self.parietal.stats(),
            "total_events": self.pfc.stats()["count"] + self.hippocampus.stats()["count"]
        }

    def export_knowledge_graph(self, format: str = "graphml") -> bytes:
        """Export the knowledge graph to file format.

        Args:
            format: 'graphml' or 'jsonl'

        Returns:
            Serialized graph data
        """
        return self.parietal.export(format)


__all__ = ["CerebralMemory"]
