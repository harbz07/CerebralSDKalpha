from __future__ import annotations
import json, random
from datetime import datetime
import numpy as np

from sdk.core.types import MemoryEvent
from sdk.core.hippocampus import Hippocampus

def recall_at_k(results, answers, k):
    hits = 0
    for res, ans in zip(results, answers):
        hits += int(any(a in res[:k] for a in ans))
    return hits / len(results)

def mrr(results, answers):
    total = 0.0
    for res, ans in zip(results, answers):
        rank = next((i+1 for i, r in enumerate(res) if any(a == r for a in ans)), None)
        if rank: total += 1/rank
    return total / len(results)

def main():
    random.seed(0); np.random.seed(0)
    h = Hippocampus(similarity_threshold=0.80, top_k=5)

    qa = [
        {"query":"alpha", "answers":["alpha"], "emb": np.array([1,0,0,0], dtype=np.float32)},
        {"query":"bravo", "answers":["bravo"], "emb": np.array([0,1,0,0], dtype=np.float32)},
        {"query":"charlie", "answers":["charlie"], "emb": np.array([0,0,1,0], dtype=np.float32)},
        {"query":"delta", "answers":["delta"], "emb": np.array([0,0,0,1], dtype=np.float32)},
    ]

    now = datetime.utcnow()
    for i, item in enumerate(qa):
        ev = MemoryEvent(
            id=str(i), timestamp=now, content=item["answers"][0],
            event_class="Foundation",
            scores={"error_severity":5,"novelty":5,"foundation_weight":5,"rlhf_weight":9},
            importance=0.7, novelty=0.4, embedding=item["emb"].tolist()
        )
        h.consolidate(ev)

    results = []
    answers = [x["answers"] for x in qa]
    for item in qa:
        out = h.recall(query_embedding=item["emb"], top_k=5)
        results.append([e.content for e in out])

    report = {"R@5": recall_at_k(results, answers, 5), "MRR": mrr(results, answers)}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
