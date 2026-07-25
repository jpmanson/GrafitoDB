"""Hybrid retrieval: fuse ranked lists with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def rrf_fuse(
    ranked_lists: list[list[T]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
    limit: int | None = None,
    key: Callable[[T], object] | None = None,
) -> list[tuple[T, float]]:
    """Fuse ordered candidate lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each list is best-first (rank 0 = top).
        k: RRF constant (typical 60).
        weights: Per-list weights (default equal 1.0).
        limit: Max results to return (None = all scored).
        key: Optional identity function for items (default: identity).

    Returns:
        List of ``(item, rrf_score)`` sorted by score descending.
    """
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")

    id_fn = key or (lambda x: x)
    scores: dict[object, float] = {}
    items: dict[object, T] = {}

    for w, ranked in zip(weights, ranked_lists):
        for rank, item in enumerate(ranked):
            ident = id_fn(item)
            items[ident] = item
            scores[ident] = scores.get(ident, 0.0) + float(w) / (float(k) + rank + 1)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = [(items[i], s) for i, s in ordered]
    if limit is not None:
        out = out[:limit]
    return out
