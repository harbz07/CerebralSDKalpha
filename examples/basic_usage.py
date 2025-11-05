"""
Basic usage example for the Cerebral SDK memory system.

This script demonstrates:
1. Creating MemoryEvent objects
2. Using individual modules (PFC, Parietal, Hippocampus)
3. Using the integrated CerebralMemory system
4. Retrieving context from all subsystems
"""

from datetime import datetime, timedelta
import numpy as np

from sdk.core import (
    MemoryEvent,
    PrefrontalCache,
    ParietalGraph,
    Hippocampus,
    Evidence,
    CerebralMemory
)


def example_1_individual_modules():
    """Example: Using each module independently."""
    print("=" * 60)
    print("EXAMPLE 1: Individual Module Usage")
    print("=" * 60)

    # Create some sample events
    now = datetime.utcnow()

    event1 = MemoryEvent(
        id="evt_001",
        timestamp=now,
        content="User asked about machine learning best practices",
        event_class="Foundation",
        scores={
            "error_severity": 0,
            "novelty": 7,
            "foundation_weight": 8,
            "rlhf_weight": 6
        },
        importance=0.8,
        novelty=0.7,
        tags=["question", "ml"],
        embedding=np.random.rand(128).tolist()
    )

    event2 = MemoryEvent(
        id="evt_002",
        timestamp=now + timedelta(minutes=5),
        content="Explained gradient descent and backpropagation",
        event_class="Foundation",
        scores={
            "error_severity": 0,
            "novelty": 5,
            "foundation_weight": 9,
            "rlhf_weight": 7
        },
        importance=0.9,
        novelty=0.5,
        tags=["answer", "ml", "algorithms"],
        embedding=np.random.rand(128).tolist()
    )

    # ---- PFC (Working Memory) ----
    print("\n--- PFC (Working Memory) ---")
    pfc = PrefrontalCache(max_events=100)
    pfc.push(event1)
    pfc.push(event2)

    recent = pfc.recent(n=2)
    print(f"Recent events: {len(recent)}")
    for ev in recent:
        print(f"  - {ev.content}")

    context = pfc.sample_for_context(budget=2)
    print(f"\nSampled context (top 2): {context}")

    # ---- Hippocampus (Long-term Memory) ----
    print("\n--- Hippocampus (Long-term Memory) ---")
    hippo = Hippocampus(similarity_threshold=0.7, top_k=5)

    # Consolidate events
    consolidated = hippo.consolidate(event1)
    print(f"Event 1 consolidated: {consolidated}")
    consolidated = hippo.consolidate(event2)
    print(f"Event 2 consolidated: {consolidated}")

    # Recall by similarity
    query = np.random.rand(128)
    recalled = hippo.recall(query_embedding=query, top_k=2)
    print(f"\nRecalled {len(recalled)} memories:")
    for mem in recalled:
        print(f"  - {mem.content}")

    print(f"\nHippocampus stats: {hippo.stats()}")

    # ---- Parietal (Knowledge Graph) ----
    print("\n--- Parietal (Knowledge Graph) ---")
    parietal = ParietalGraph()

    # Add some facts
    evidence = Evidence(
        source="evt_001",
        timestamp=now.isoformat(),
        confidence=0.95,
        snippet="User asked about ML"
    )

    parietal.upsert_triplet("User", "ASKED_ABOUT", "Machine Learning", evidence)
    parietal.upsert_triplet("Machine Learning", "REQUIRES", "Gradient Descent", evidence)
    parietal.upsert_triplet("Gradient Descent", "USES", "Backpropagation", evidence)
    parietal.upsert_triplet("User", "INTERESTED_IN", "AI", evidence)

    # Query facts
    facts = parietal.nearest_facts("User", k=5)
    print(f"\nFacts about 'User': {len(facts)}")
    for fact in facts:
        print(f"  - {fact.h} --[{fact.r}]--> {fact.t}")

    # Find reasoning paths
    paths = parietal.find_paths("User", "Backpropagation", max_length=3)
    print(f"\nReasoning paths from User to Backpropagation: {len(paths)}")
    for path in paths:
        print(f"  {path}")

    print(f"\nParietal stats: {parietal.stats()}")


def example_2_integrated_system():
    """Example: Using the integrated CerebralMemory system."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: Integrated CerebralMemory System")
    print("=" * 60)

    # Initialize the integrated system
    memory = CerebralMemory(
        pfc_capacity=100,
        hippocampus_threshold=0.75,
        hippocampus_top_k=5
    )

    # Create events with facts embedded in metadata
    now = datetime.utcnow()

    event1 = MemoryEvent(
        id="evt_101",
        timestamp=now,
        content="Alice works on the Cerebral SDK project",
        event_class="Foundation",
        scores={
            "error_severity": 0,
            "novelty": 6,
            "foundation_weight": 8,
            "rlhf_weight": 5
        },
        importance=0.85,
        novelty=0.6,
        tags=["project", "team"],
        embedding=np.random.rand(128).tolist(),
        metadata={
            "facts": [
                {"h": "Alice", "r": "WORKS_ON", "t": "Cerebral SDK", "confidence": 1.0},
                {"h": "Cerebral SDK", "r": "TYPE", "t": "Memory System", "confidence": 1.0}
            ]
        }
    )

    event2 = MemoryEvent(
        id="evt_102",
        timestamp=now + timedelta(minutes=10),
        content="The Cerebral SDK has three main modules: PFC, Parietal, and Hippocampus",
        event_class="Foundation",
        scores={
            "error_severity": 0,
            "novelty": 7,
            "foundation_weight": 9,
            "rlhf_weight": 6
        },
        importance=0.9,
        novelty=0.7,
        tags=["architecture", "technical"],
        embedding=np.random.rand(128).tolist(),
        metadata={
            "facts": [
                {"h": "Cerebral SDK", "r": "HAS_MODULE", "t": "PFC", "confidence": 1.0},
                {"h": "Cerebral SDK", "r": "HAS_MODULE", "t": "Parietal", "confidence": 1.0},
                {"h": "Cerebral SDK", "r": "HAS_MODULE", "t": "Hippocampus", "confidence": 1.0}
            ]
        }
    )

    # Ingest events (automatically distributed to subsystems)
    print("\n--- Ingesting Events ---")
    result1 = memory.ingest(event1, extract_facts=True)
    print(f"Event 1 ingested: {result1}")

    result2 = memory.ingest(event2, extract_facts=True)
    print(f"Event 2 ingested: {result2}")

    # Retrieve comprehensive context
    print("\n--- Retrieving Context ---")
    query_emb = np.random.rand(128)

    context = memory.retrieve_context(
        query_embedding=query_emb,
        query_text="Cerebral SDK",
        pfc_budget=3,
        ltm_top_k=2,
        kg_facts_k=5
    )

    print(f"\nWorking Memory ({len(context['working_memory'])} items):")
    for item in context['working_memory']:
        print(f"  - {item}")

    print(f"\nLong-term Memory ({len(context['long_term'])} items):")
    for item in context['long_term']:
        print(f"  - {item}")

    print(f"\nKnowledge Graph ({len(context['knowledge_graph'])} facts):")
    for fact in context['knowledge_graph']:
        print(f"  - {fact}")

    # Find reasoning paths
    print("\n--- Reasoning Paths ---")
    paths = memory.find_reasoning_path("Alice", "Hippocampus", max_length=3)
    for path in paths:
        print(f"  {path}")

    # Get entity context
    print("\n--- Entity Context: Alice ---")
    entity_context = memory.get_entity_context("Alice")
    print(f"Outgoing relations ({len(entity_context['outgoing'])}):")
    for rel in entity_context['outgoing']:
        print(f"  - {rel}")

    # System statistics
    print("\n--- System Statistics ---")
    stats = memory.stats()
    print(f"Total events: {stats['total_events']}")
    print(f"PFC: {stats['pfc']}")
    print(f"Hippocampus: {stats['hippocampus']}")
    print(f"Parietal: {stats['parietal']}")


def example_3_advanced_features():
    """Example: Advanced features (filtering, forgetting, maintenance)."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 3: Advanced Features")
    print("=" * 60)

    memory = CerebralMemory()
    now = datetime.utcnow()

    # Create diverse events
    events = []
    for i in range(10):
        event = MemoryEvent(
            id=f"evt_{i:03d}",
            timestamp=now + timedelta(minutes=i),
            content=f"Event {i} content",
            event_class=["Chaos", "Foundation", "Glow"][i % 3],
            scores={
                "error_severity": i % 10,
                "novelty": (i * 2) % 10,
                "foundation_weight": (i * 3) % 10,
                "rlhf_weight": (i * 4) % 10
            },
            importance=0.3 + (i * 0.05),
            novelty=0.2 + (i * 0.04),
            tags=["important"] if i % 3 == 0 else ["routine"],
            embedding=np.random.rand(128).tolist()
        )
        events.append(event)

    # Batch ingestion
    print("\n--- Batch Ingestion ---")
    stats = memory.ingest_batch(events, extract_facts=False)
    print(f"Ingestion stats: {stats}")

    # Tag-based filtering
    print("\n--- Tag-based Retrieval ---")
    important = memory.pfc.by_tags(["important"])
    print(f"Important events in PFC: {len(important)}")

    # Class-based filtering
    print("\n--- Class-based Retrieval ---")
    foundation = memory.hippocampus.get_by_class("Foundation")
    print(f"Foundation events in LTM: {len(foundation)}")

    # Maintenance operations
    print("\n--- Memory Maintenance ---")
    cleanup_stats = memory.maintenance(
        forget_ltm_older_than=timedelta(days=365),
        forget_low_importance=0.3
    )
    print(f"Cleanup stats: {cleanup_stats}")

    # Final statistics
    print("\n--- Final System State ---")
    final_stats = memory.stats()
    print(f"Total events remaining: {final_stats['total_events']}")
    print(f"Hippocampus avg importance: {final_stats['hippocampus'].get('avg_importance', 0):.2f}")


if __name__ == "__main__":
    # Run all examples
    example_1_individual_modules()
    example_2_integrated_system()
    example_3_advanced_features()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
