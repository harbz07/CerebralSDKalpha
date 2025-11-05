# Cerebral SDK

A neuroscience-inspired memory system for AI agents, featuring three interconnected subsystems modeled after human memory architecture.

## 🧠 Architecture

The Cerebral SDK implements a **memory triad** inspired by cognitive neuroscience:

### 1. **PFC (Prefrontal Cache)** - Working Memory
Short-term, high-access memory for recent events with:
- **Capacity-bounded buffer** (configurable, default 1000 events)
- **Recency × Importance scoring** for context selection
- **Attention-based boosting** for urgent/important items
- **Temporal decay** mechanisms for aging events
- **Tag-based filtering** and retrieval

### 2. **Parietal Graph** - Structured Knowledge
Knowledge graph for facts and relationships with:
- **Entity canonicalization** (fuzzy deduplication)
- **Evidence-backed triplets** (h, r, t format)
- **Path finding** for multi-hop reasoning
- **Subgraph extraction** for focused context
- **Bidirectional queries** (incoming/outgoing relations)
- **Export capabilities** (GraphML, JSONL)

### 3. **Hippocampus** - Long-term Memory
Vector-based semantic memory with:
- **Catch-22 consolidation gate** (threshold score ≥ 22)
- **Cosine similarity recall** with configurable threshold
- **Class bias system** (Chaos, Foundation, Glow)
- **Temporal decay** for aging memories
- **Forgetting mechanisms** (by age, importance, access frequency)
- **Importance-weighted retrieval**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/harbz07/CerebralSDKalpha.git
cd CerebralSDKalpha

# Install dependencies
pip install numpy networkx
```

### Basic Usage

```python
from datetime import datetime
import numpy as np
from sdk.core import MemoryEvent, CerebralMemory

# Initialize the integrated memory system
memory = CerebralMemory()

# Create a memory event
event = MemoryEvent(
    id="evt_001",
    timestamp=datetime.utcnow(),
    content="User asked about machine learning",
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
    embedding=np.random.rand(128).tolist(),
    metadata={
        "facts": [
            {"h": "User", "r": "ASKED_ABOUT", "t": "Machine Learning"}
        ]
    }
)

# Ingest the event (automatically distributed to all subsystems)
result = memory.ingest(event, extract_facts=True)
print(result)  # {'pfc': True, 'hippocampus': True, 'parietal': True}

# Retrieve context from all subsystems
context = memory.retrieve_context(
    query_embedding=np.random.rand(128),
    query_text="Machine Learning",
    pfc_budget=5,
    ltm_top_k=3,
    kg_facts_k=5
)

print(context)
# {
#   'working_memory': [...],
#   'long_term': [...],
#   'knowledge_graph': [...]
# }
```

## 📚 Key Features

### PFC Enhancements
- ✅ **Temporal decay** with configurable half-life
- ✅ **Attention boosting** for tagged events (attention, urgent)
- ✅ **Custom scoring functions** for flexible prioritization
- ✅ **Tag filtering** (include/exclude)
- ✅ **Old event cleanup** by age threshold
- ✅ **Statistics** by event class

### Parietal Enhancements
- ✅ **Multi-hop reasoning** via path finding
- ✅ **Bidirectional fact retrieval** (incoming + outgoing)
- ✅ **Subgraph extraction** with depth control
- ✅ **Relation type queries**
- ✅ **Entity degree ranking** (most connected)
- ✅ **Confidence-weighted paths**

### Hippocampus Enhancements
- ✅ **Batch consolidation** for efficiency
- ✅ **Temporal decay** during recall
- ✅ **Importance-weighted retrieval**
- ✅ **Tag and class filtering**
- ✅ **Multiple forgetting mechanisms**:
  - By age threshold
  - By importance threshold
  - By access frequency
- ✅ **Access tracking** for memory reinforcement

### Integration Layer (CerebralMemory)
- ✅ **Unified ingestion** to all subsystems
- ✅ **Hybrid retrieval** combining all memory types
- ✅ **Cross-module queries**
- ✅ **Maintenance operations**
- ✅ **Comprehensive statistics**

## 🔬 Event Taxonomy

### Event Classes
- **Chaos**: Errors/corrections worth remembering (bias: 0.0)
- **Foundation**: Core knowledge & invariants (bias: +2.0)
- **Glow**: Insights & identity-defining moments (bias: +2.5)

### Score Dimensions (0-10 scale)
1. **error_severity**: How impactful was the failure?
2. **novelty**: How unusual/rare is the information?
3. **foundation_weight**: Does it support core knowledge?
4. **rlhf_weight**: Does it improve aligned behavior?

**Catch-22 Rule**: Events consolidate to long-term memory when:
- `sum(scores) + class_bias >= 22`, OR
- Event has `"pin"` tag (bypass threshold)

## 📖 Examples

### Individual Module Usage

```python
from sdk.core import PrefrontalCache, ParietalGraph, Hippocampus

# PFC: Working Memory
pfc = PrefrontalCache(max_events=100)
pfc.push(event)
recent = pfc.recent(n=5)
context = pfc.sample_for_context(budget=8, filter_tags=["important"])

# Parietal: Knowledge Graph
parietal = ParietalGraph()
parietal.upsert_triplet("Alice", "WORKS_ON", "Project X", evidence)
facts = parietal.nearest_facts("Alice", k=10)
paths = parietal.find_paths("Alice", "Python", max_length=3)

# Hippocampus: Long-term Memory
hippo = Hippocampus(similarity_threshold=0.80, temporal_decay=True)
consolidated = hippo.consolidate(event)
memories = hippo.recall(query_embedding, top_k=5, filter_class="Foundation")
removed = hippo.forget_old(older_than=timedelta(days=365))
```

### Advanced Retrieval

```python
# Importance-weighted semantic search
results = memory.hippocampus.recall(
    query_embedding=query_vec,
    top_k=10,
    filter_tags=["critical"],
    exclude_tags=["archived"],
    filter_class="Foundation",
    importance_weight=0.3  # Boost by 30% per importance unit
)

# Multi-hop reasoning in knowledge graph
paths = memory.parietal.find_paths(
    start="User Input",
    end="System Response",
    max_length=4,
    limit=5
)

# Comprehensive entity context
entity_ctx = memory.get_entity_context("Machine Learning")
print(entity_ctx['outgoing'])  # What ML relates to
print(entity_ctx['incoming'])  # What relates to ML
```

### Memory Maintenance

```python
# Periodic cleanup
stats = memory.maintenance(
    forget_ltm_older_than=timedelta(days=365),
    forget_pfc_older_than=timedelta(hours=24),
    forget_low_importance=0.3
)

# Manual consolidation from working memory to long-term
consolidated = memory.consolidate_working_to_longterm(
    tag_filter=["critical"],
    min_importance=0.7
)

# Forget least accessed memories (keep top 1000)
removed = memory.hippocampus.forget_least_accessed(
    keep_top_n=1000,
    keep_pinned=True
)
```

## 🧪 Testing

```bash
# Run the test suite
python -m unittest tests.test_memory -v

# Run the example script
python examples/basic_usage.py
```

## 📊 System Statistics

```python
stats = memory.stats()
# {
#   'pfc': {
#     'count': 487,
#     'capacity': 1000,
#     'by_class': {'Foundation': 201, 'Chaos': 143, 'Glow': 143}
#   },
#   'hippocampus': {
#     'count': 1523,
#     'dims': 128,
#     'by_class': {'Foundation': 892, 'Glow': 431, 'Chaos': 200},
#     'avg_importance': 0.73,
#     'pinned': 45,
#     'total_accesses': 3421
#   },
#   'parietal': {
#     'nodes': 234,
#     'edges': 587,
#     'relations': {'WORKS_ON': 45, 'USES': 123, ...}
#   },
#   'total_events': 2010
# }
```

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      CerebralMemory                         │
│                  (Integration Layer)                        │
└────────────┬─────────────┬─────────────┬───────────────────┘
             │             │             │
    ┌────────▼────────┐   │   ┌─────────▼────────┐
    │  PFC            │   │   │  Hippocampus     │
    │  (Working Mem)  │   │   │  (Long-term Mem) │
    │                 │   │   │                  │
    │ • Recency×Imp   │   │   │ • Catch-22 Gate  │
    │ • Attention     │   │   │ • Vector Recall  │
    │ • Decay         │   │   │ • Forgetting     │
    └─────────────────┘   │   └──────────────────┘
                          │
                  ┌───────▼──────┐
                  │  Parietal    │
                  │  (Knowledge) │
                  │              │
                  │ • Facts (h,r,t) │
                  │ • Reasoning     │
                  │ • Subgraphs     │
                  └─────────────────┘
```

## 🛠️ Configuration

```python
memory = CerebralMemory(
    # PFC settings
    pfc_capacity=1000,
    pfc_decay_halflife=timedelta(hours=6),

    # Hippocampus settings
    hippocampus_threshold=0.80,
    hippocampus_top_k=5,
    hippocampus_temporal_decay=True,

    # Parietal settings
    parietal_dedupe_threshold=0.90
)
```

## 📄 License

MIT License - See LICENSE file for details.

## 👤 Author

Harvey (@harbz07)

## 🤝 Contributing

Contributions welcome! Please check the issues page or create a pull request.

## 📚 References

- Memory consolidation: Catch-22 threshold inspired by dual-process theory
- Knowledge graphs: NetworkX-based implementation with evidence tracking
- Vector memory: Cosine similarity with temporal decay mechanisms
