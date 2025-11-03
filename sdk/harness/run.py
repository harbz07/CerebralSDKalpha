from __future__ import annotations

"""Entry point for running evaluation suites with ablation flags."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np

from sdk.core.composer import ContextComposer
from sdk.core.hippocampus import Hippocampus
from sdk.core.parietal import ParietalGraph, Evidence
from sdk.core.pfc import PrefrontalCache
from sdk.core.types import AblationConfig, MemoryEvent

from .datasets import load_entities_dataset, load_qa_dataset
from .metrics import mean_reciprocal_rank, recall_at_k
from .reporter import RunSummary, write_json_report
from .synthetic import generate_events


def _build_memory_system(ablations: AblationConfig) -> ContextComposer:
    pfc = PrefrontalCache(max_events=512)
    hippocampus = Hippocampus()
    parietal = ParietalGraph()

    for event in generate_events(32):
        pfc.push(event)

    now = datetime.utcnow()
    for idx, example in enumerate(load_qa_dataset()):
        event = MemoryEvent(
            id=f"qa-{idx}",
            timestamp=now,
            content=example.answers[0],
            event_class="Foundation",
            scores={"error_severity": 6, "novelty": 5, "foundation_weight": 6, "rlhf_weight": 7},
            importance=1.5,
            novelty=0.3,
            tags=["pin"],
            metadata={"query": example.query},
            embedding=example.embedding,
        )
        pfc.push(event)
        hippocampus.consolidate(event)

    for event in generate_events(32):
        pfc.push(event)

    for entity in load_entities_dataset():
        project = entity.get("attributes", {}).get("project")
        if project:
            parietal.upsert_triplet(
                entity["entity"],
                "OWNS",
                project,
                Evidence(source="entities", snippet=f"{entity['entity']} owns {project}", confidence=0.9, timestamp=None),
            )

    return ContextComposer(pfc=pfc, hippocampus=hippocampus, parietal=parietal, ablations=ablations)


def run_suite(ablations: AblationConfig) -> Dict[str, float]:
    composer = _build_memory_system(ablations)
    qa_examples = load_qa_dataset()

    results: List[List[str]] = []
    truths: List[List[str]] = []
    for item in qa_examples:
        emb = np.asarray(item.embedding, dtype=np.float32)
        res = composer.compose(query=item.query, query_embedding=emb, budget=2, kg_k=1, ltm_k=3)
        ltm_items = res.segments.get("ltm")
        results.append(ltm_items.items if ltm_items else [])
        truths.append(item.answers)

    metrics = {
        "Recall@5": recall_at_k(results, truths, 5),
        "MRR": mean_reciprocal_rank(results, truths),
    }
    return metrics


def main(argv: list[str] | None = None) -> RunSummary:
    parser = argparse.ArgumentParser(description="Run Cerebral SDK eval harness")
    parser.add_argument("--no-kg", action="store_true", help="Disable Parietal graph retrieval")
    parser.add_argument("--no-ltm", action="store_true", help="Disable Hippocampus retrieval")
    parser.add_argument("--only-pfc", action="store_true", help="Use only the PFC context")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"), help="Directory for JSON reports")
    args = parser.parse_args(argv)

    ablations = AblationConfig.from_flags(no_kg=args.no_kg, no_ltm=args.no_ltm, only_pfc=args.only_pfc)
    metrics = run_suite(ablations)
    path = write_json_report(metrics, directory=args.report_dir)
    summary = RunSummary(run_id=path.stem, metrics=metrics)
    print(summary.to_markdown())
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
