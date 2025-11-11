# Cerebral SDK

A neuromorphic memory engine implementing a "memory triad" architecture that mimics human brain memory systems for AI applications.

## Overview

Cerebral SDK provides three complementary memory subsystems that work together to help AI systems maintain context and make better decisions:

- **PrefrontalCache (PFC)**: Working memory buffer with recency-based sampling
- **Hippocampus**: Long-term semantic memory with vector similarity recall
- **ParietalGraph**: Knowledge graph for structured relational facts

## Architecture

The SDK implements a "Catch-22" scoring system where memory events are evaluated across four canonical dimensions:

1. `error_severity`: Importance of errors or failures
2. `novelty`: Uniqueness of the information
3. `foundation_weight`: Core knowledge importance
4. `rlhf_weight`: Reinforcement learning feedback weight

Events with a total score ≥ 22 (or tagged with "pin") are consolidated into long-term memory.

## Installation

### Development Installation

```bash
pip install -e ".[dev]"
```

This installs the SDK along with development dependencies (pytest, mypy, ruff).

### Basic Installation

```bash
pip install -e .
```

## Quick Start

```python
from sdk.core import MemoryEvent, PrefrontalCache, Hippocampus, ParietalGraph
from datetime import datetime

# Create a memory event
event = MemoryEvent(
    id="evt_001",
    timestamp=datetime.now(),
    content="User asked about Python decorators",
    event_class="Foundation",
    scores={
        "error_severity": 0.0,
        "novelty": 8.0,
        "foundation_weight": 10.0,
        "rlhf_weight": 5.0
    },
    importance=0.8,
    novelty=0.7,
    tags=["learning", "python"],
    embedding=[0.1, 0.2, 0.3, ...]  # Your embedding vector
)

# Working memory (PFC)
pfc = PrefrontalCache(capacity=100)
pfc.push(event)
recent_events = pfc.recent(n=5)
context = pfc.sample_for_context(budget=10)

# Long-term memory (Hippocampus)
hippocampus = Hippocampus(similarity_threshold=0.80, top_k=5)
if hippocampus.consolidate(event):  # Only if score ≥ 22
    print("Event consolidated to LTM")

query_embedding = [0.1, 0.2, 0.3, ...]
recalled = hippocampus.recall(query_embedding, top_k=3)

# Knowledge graph (ParietalGraph)
kg = ParietalGraph(dedupe_threshold=0.90)
from sdk.core.parietal import Evidence

evidence = Evidence(
    source="documentation",
    url="https://example.com/docs",
    confidence=0.95
)

kg.upsert_triplet(
    h="Python",
    r="has_feature",
    t="decorators",
    evidence=evidence
)

facts = kg.nearest_facts("Python", k=5)
graphml_export = kg.export(fmt="graphml")
```

## Memory Event Structure

Each `MemoryEvent` contains:

- **id**: Unique identifier
- **timestamp**: When the event occurred
- **content**: The actual content/text
- **event_class**: Classification (e.g., "Foundation", "Glow", "Chaos")
- **scores**: Dict with the 4 canonical dimensions
- **importance**: Float rating (0.0-1.0)
- **novelty**: Float rating (0.0-1.0)
- **tags**: List of metadata tags
- **metadata**: Arbitrary additional data
- **embedding**: Optional vector for similarity search

## Memory Subsystems

### PrefrontalCache (Working Memory)

Circular buffer that maintains recent events with automatic eviction. Supports sampling by recency × importance weighting.

**Key Features:**
- O(1) push/pop operations
- Configurable capacity
- Context sampling with scoring

### Hippocampus (Long-Term Memory)

Vector-based semantic memory with cosine similarity recall and Catch-22 gating.

**Key Features:**
- Catch-22 threshold (sum of scores ≥ 22)
- Class bias tuning (Foundation: +2, Glow: +1, Chaos: +0)
- Cosine similarity search
- "pin" tag bypass for forced consolidation

### ParietalGraph (Knowledge Graph)

Ontology-aware triplet store with entity canonicalization and evidence tracking.

**Key Features:**
- Entity deduplication (fuzzy matching at 90% threshold)
- Evidence tracking per fact
- Multiple export formats (GraphML, JSONL)
- Relation-based querying

## Configuration

Configuration files are available in the `configs/` directory:

- `configs/memory.yaml`: Memory subsystem parameters
- `configs/embeddings.yaml`: Embedding model settings

Knowledge resources:

- `knowledge/event_taxonomy.yaml`: Event classification schema
- `knowledge/ontology.parietal.yaml`: Knowledge graph ontology
- `knowledge/prompts.md`: LLM prompts for memory processing

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=sdk --cov-report=html

# Using Make
make test
```

### Type Checking

```bash
# Run mypy
mypy sdk/

# Using Make
make type
```

### Code Linting

```bash
# Run ruff
ruff check sdk/ tests/

# Auto-fix issues
ruff check --fix sdk/ tests/

# Using Make
make lint
```

### Development Workflow

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all checks
make lint type test
```

## Project Structure

```
CerebralSDKalpha/
├── sdk/                      # Main SDK code
│   └── core/                 # Core memory implementations
│       ├── types.py          # MemoryEvent dataclass
│       ├── pfc.py            # PrefrontalCache
│       ├── hippocampus.py    # Hippocampus LTM
│       └── parietal.py       # ParietalGraph KG
├── tests/                    # Unit tests
│   └── test_memory.py        # Comprehensive test suite
├── configs/                  # Configuration files
├── knowledge/                # Knowledge resources
├── scripts/                  # Utility scripts
└── seeds/                    # Seed data for examples
```

## Testing

The SDK includes a comprehensive test suite covering:

- **PFC Tests**: Capacity management, recency ordering, context sampling
- **Hippocampus Tests**: Catch-22 consolidation, cosine recall, threshold filtering
- **ParietalGraph Tests**: Triplet operations, canonicalization, export formats
- **RFC Validation**: Constant and configuration sanity checks

All tests pass with 100% success rate.

## Performance Characteristics

- **PFC**: O(1) push/pop, O(n) sampling where n = capacity
- **Hippocampus**: O(n) recall where n = stored events, O(d) similarity computation where d = embedding dimensions
- **ParietalGraph**: O(n) canonicalization where n = nodes, O(e) retrieval where e = edges

## Integration

Cerebral SDK is designed to integrate with:

- **Cerebral Studio**: Unified web application with React frontend and FastAPI backend
- **MCP Servers**: Model Context Protocol for LLM integration
- **Vector Databases**: For scalable embedding storage
- **Graph Databases**: For advanced knowledge graph queries

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contributing

This is an alpha release. The SDK is under active development.

## Documentation

For the complete assembly manual covering the broader Cerebral Studio application:

- See `Foundational Files/Cerebral Studio PRE-BETA Manual.md`

## Requirements

- Python 3.10+
- numpy >= 1.24
- networkx >= 3.1

Development dependencies:

- pytest >= 7.4.0
- mypy >= 1.10.0
- ruff >= 0.6.5

## Status

**Version**: 0.1.0 (Alpha)

Current status: Core memory modules implemented and fully tested. Type checking passes with strict settings.
