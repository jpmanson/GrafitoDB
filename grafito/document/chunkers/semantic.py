"""Breakpoint-based semantic chunker (optional; requires an embedding function)."""

from __future__ import annotations

import re
from typing import Any, Callable

from ..types import ChunkSpec

# Sentence split: keep delimiters rough for offset recovery
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class SemanticBreakpointChunker:
    """Split on large consecutive-sentence embedding distance (breakpoint).

    Contiguous only (no non-adjacent clustering). Enforces ``min_chars`` /
    ``max_chars`` clamps so chunks do not collapse to 1 sentence or grow unbounded.

    Requires an embedder: ``Callable[[list[str]], list[list[float]]]`` (same
    shape as :class:`~grafito.embedding_functions.base.EmbeddingFunction`).
    """

    name = "semantic_breakpoint"

    def __init__(
        self,
        embedder: Callable[[list[str]], list[list[float]]],
        *,
        threshold: float = 0.25,
        min_chars: int = 200,
        max_chars: int = 1200,
        name: str = "semantic_breakpoint",
    ) -> None:
        if threshold < 0 or threshold > 2:
            raise ValueError("threshold should be a cosine distance in a reasonable range")
        if min_chars <= 0 or max_chars < min_chars:
            raise ValueError("invalid min_chars/max_chars")
        self.embedder = embedder
        self.threshold = threshold
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.name = name

    def split(self, text: str, *, meta: dict[str, Any] | None = None) -> list[ChunkSpec]:
        if not text or not text.strip():
            return []
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [
                ChunkSpec(
                    text=text.strip(),
                    ord=0,
                    char_start=0,
                    char_end=len(text),
                    strategy=self.name,
                )
            ]

        sents = [s for s, _, _ in sentences]
        vectors = self.embedder(sents)
        if len(vectors) != len(sentences):
            raise ValueError("embedder must return one vector per sentence")

        # Break after sentence i when distance(i, i+1) > threshold
        breaks: list[int] = []  # indices after which to cut (end exclusive for group)
        for i in range(len(vectors) - 1):
            dist = _cosine_distance(vectors[i], vectors[i + 1])
            if dist >= self.threshold:
                breaks.append(i + 1)

        groups = self._group_by_breaks(len(sentences), breaks)
        # Enforce size clamps by merging small / splitting large groups
        groups = self._clamp_groups(sentences, groups)

        specs: list[ChunkSpec] = []
        for ord_i, (g0, g1) in enumerate(groups):
            char_start = sentences[g0][1]
            char_end = sentences[g1 - 1][2]
            body = text[char_start:char_end].strip()
            if not body:
                continue
            # Recompute strip offsets roughly
            raw = text[char_start:char_end]
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            specs.append(
                ChunkSpec(
                    text=body,
                    ord=ord_i,
                    char_start=char_start + lead,
                    char_end=char_end - trail,
                    strategy=self.name,
                )
            )
        for i, s in enumerate(specs):
            s.ord = i
        return specs

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Return (sentence, char_start, char_end) half-open spans."""
        parts: list[tuple[str, int, int]] = []
        last = 0
        for m in _SENT_SPLIT.finditer(text):
            end = m.start()
            if end > last:
                seg = text[last:end]
                if seg.strip():
                    parts.append((seg.strip(), last, end))
            last = m.end()
        if last < len(text):
            seg = text[last:]
            if seg.strip():
                parts.append((seg.strip(), last, len(text)))
        if not parts:
            parts.append((text.strip(), 0, len(text)))
        return parts

    def _group_by_breaks(self, n: int, breaks: list[int]) -> list[tuple[int, int]]:
        groups: list[tuple[int, int]] = []
        start = 0
        for b in breaks:
            if b > start:
                groups.append((start, b))
                start = b
        if start < n:
            groups.append((start, n))
        return groups or [(0, n)]

    def _clamp_groups(
        self,
        sentences: list[tuple[str, int, int]],
        groups: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        # Merge undersized consecutive groups; split oversized by sentence count
        merged: list[tuple[int, int]] = []
        for g0, g1 in groups:
            char_len = sentences[g1 - 1][2] - sentences[g0][1]
            if merged and char_len < self.min_chars:
                p0, _ = merged[-1]
                merged[-1] = (p0, g1)
            else:
                merged.append((g0, g1))

        final: list[tuple[int, int]] = []
        for g0, g1 in merged:
            char_len = sentences[g1 - 1][2] - sentences[g0][1]
            if char_len <= self.max_chars or g1 - g0 <= 1:
                final.append((g0, g1))
                continue
            # Split roughly by accumulating sentences until max_chars
            start = g0
            acc_start = sentences[g0][1]
            for i in range(g0, g1):
                if i > start and sentences[i][2] - acc_start > self.max_chars:
                    final.append((start, i))
                    start = i
                    acc_start = sentences[i][1]
            if start < g1:
                final.append((start, g1))
        return final


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 1.0
    cos = dot / ((na ** 0.5) * (nb ** 0.5))
    # clamp numerical noise
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos
