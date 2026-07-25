"""Fixed-size window chunker (chars or tokens via pluggable counter)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from ..types import ChunkSpec

TokenCounter = Callable[[str], int]


class FixedChunker:
    """Split text into fixed-size windows with optional overlap.

    Args:
        max_size: Window size in ``unit`` (characters or tokens).
        overlap: Overlap between consecutive windows (same unit).
        unit: ``chars`` (default) or ``tokens`` (requires ``counter``).
        counter: Token counter ``str -> int`` when ``unit="tokens"``.
        name: Strategy name stored on specs.
    """

    def __init__(
        self,
        max_size: int = 1200,
        overlap: int = 0,
        *,
        unit: Literal["chars", "tokens"] = "chars",
        counter: TokenCounter | None = None,
        name: str = "fixed",
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if overlap < 0 or overlap >= max_size:
            raise ValueError("overlap must be >= 0 and < max_size")
        if unit not in ("chars", "tokens"):
            raise ValueError("unit must be 'chars' or 'tokens'")
        if unit == "tokens" and counter is None:
            raise ValueError("counter is required when unit='tokens'")
        self.max_size = max_size
        self.overlap = overlap
        self.unit = unit
        self.counter = counter
        self.name = name

    def split(self, text: str, *, meta: dict[str, Any] | None = None) -> list[ChunkSpec]:
        if not text:
            return []
        if self.unit == "chars":
            return self._split_chars(text)
        return self._split_tokens(text)

    def _split_chars(self, text: str) -> list[ChunkSpec]:
        specs: list[ChunkSpec] = []
        step = max(1, self.max_size - self.overlap)
        start = 0
        ord_i = 0
        n = len(text)
        while start < n:
            end = min(n, start + self.max_size)
            chunk = text[start:end]
            specs.append(
                ChunkSpec(
                    text=chunk,
                    ord=ord_i,
                    char_start=start,
                    char_end=end,
                    strategy=self.name,
                )
            )
            ord_i += 1
            if end >= n:
                break
            start += step
        return specs

    def _split_tokens(self, text: str) -> list[ChunkSpec]:
        # Approximate token windows by walking characters with a sliding estimate.
        # Prefer accurate counters: accumulate until max_size tokens.
        assert self.counter is not None
        specs: list[ChunkSpec] = []
        n = len(text)
        start = 0
        ord_i = 0
        while start < n:
            end = start + 1
            # Grow end until token count exceeds max_size (binary-ish growth)
            lo, hi = start + 1, n
            best = start + 1
            while lo <= hi:
                mid = (lo + hi) // 2
                tc = self.counter(text[start:mid])
                if tc <= self.max_size:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            end = max(best, start + 1)
            chunk = text[start:end]
            tc = self.counter(chunk)
            specs.append(
                ChunkSpec(
                    text=chunk,
                    ord=ord_i,
                    char_start=start,
                    char_end=end,
                    token_count=tc,
                    strategy=self.name,
                )
            )
            ord_i += 1
            if end >= n:
                break
            # Overlap in tokens: walk back from end until ~overlap tokens
            if self.overlap <= 0:
                start = end
            else:
                back = end
                while back > start:
                    if self.counter(text[back:end]) >= self.overlap:
                        break
                    back -= max(1, (end - start) // 20)
                start = min(back, end - 1) if back < end else end
        return specs
