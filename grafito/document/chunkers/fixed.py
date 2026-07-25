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
        boundary: Literal["word", "none"] = "word",
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
        if boundary not in ("word", "none"):
            raise ValueError("boundary must be 'word' or 'none'")
        self.max_size = max_size
        self.overlap = overlap
        self.unit = unit
        self.counter = counter
        self.boundary = boundary
        self.name = name

    def split(self, text: str, *, meta: dict[str, Any] | None = None) -> list[ChunkSpec]:
        if not text:
            return []
        if self.unit == "chars":
            return self._split_chars(text)
        return self._split_tokens(text)

    def _split_chars(self, text: str) -> list[ChunkSpec]:
        specs: list[ChunkSpec] = []
        start = 0
        ord_i = 0
        n = len(text)
        while start < n:
            if self.boundary == "word":
                start = self._skip_ws(text, start, n)
                if start >= n:
                    break
            end = min(n, start + self.max_size)
            if self.boundary == "word":
                end = self._snap_word_end(text, start, end)
            specs.append(
                ChunkSpec(
                    text=text[start:end],
                    ord=ord_i,
                    char_start=start,
                    char_end=end,
                    strategy=self.name,
                )
            )
            ord_i += 1
            if end >= n:
                break
            nxt = end - self.overlap
            if self.boundary == "word" and self.overlap > 0:
                nxt = self._snap_word_start(text, nxt, end)
            start = max(nxt, start + 1)
        return specs

    @staticmethod
    def _skip_ws(text: str, start: int, n: int) -> int:
        """Advance past leading whitespace so a window begins on a word."""
        while start < n and text[start].isspace():
            start += 1
        return start

    def _snap_word_end(self, text: str, start: int, end: int) -> int:
        """Pull ``end`` back to the last whitespace so a word is not cut.

        Falls back to the hard cut when no whitespace is within a lookback
        window (e.g. a single token longer than ``max_size``). Offsets stay
        exact: ``text[start:end]`` is always a real slice.
        """
        if end >= len(text) or text[end].isspace() or text[end - 1].isspace():
            return end  # already at a clean boundary
        lo = max(start + 1, end - max(1, self.max_size // 4))
        cut = max(
            text.rfind(" ", lo, end),
            text.rfind("\n", lo, end),
            text.rfind("\t", lo, end),
        )
        return cut if cut >= lo else end

    @staticmethod
    def _snap_word_start(text: str, pos: int, end: int) -> int:
        """Push an overlap start forward to the next word start (no content gap).

        Returns ``pos`` unchanged when no clean word start is reachable within the
        window (e.g. a token longer than the overlap), keeping the plain overlap
        step instead of skipping content.
        """
        if pos <= 0 or text[pos - 1].isspace():
            return pos  # already a word start
        n = len(text)
        i = pos
        while i < end and not text[i].isspace():  # skip the rest of the partial word
            i += 1
        while i < end and text[i].isspace():       # skip the whitespace run
            i += 1
        # accept only a real word start at/before end (start <= end => no gap)
        if pos < i <= end and (i >= n or not text[i].isspace()) and text[i - 1].isspace():
            return i
        return pos

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
