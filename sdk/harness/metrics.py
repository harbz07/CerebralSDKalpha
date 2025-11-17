from __future__ import annotations

"""Metric utilities for retrieval evaluation."""

from typing import Iterable, List, Sequence


def recall_at_k(results: Sequence[Sequence[str]], answers: Sequence[Sequence[str]], k: int) -> float:
    if not results:
        return 0.0
    hits = 0
    for res, truth in zip(results, answers):
        hits += int(any(item in truth for item in res[:k]))
    return hits / len(results)


def mean_reciprocal_rank(results: Sequence[Sequence[str]], answers: Sequence[Sequence[str]]) -> float:
    if not results:
        return 0.0
    total = 0.0
    for res, truth in zip(results, answers):
        rr = 0.0
        for idx, item in enumerate(res, start=1):
            if item in truth:
                rr = 1.0 / idx
                break
        total += rr
    return total / len(results)
