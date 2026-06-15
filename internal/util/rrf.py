"""Reciprocal Rank Fusion utility."""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


def rrf_merge(ranked_lists: Iterable[list[T]], k: int = 60) -> list[tuple[T, float]]:
    """合并若干个已排序结果列表，返回 [(item, score)] 按融合分数降序。"""
    scores: dict[T, float] = {}
    for rl in ranked_lists:
        for rank, item in enumerate(rl):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
