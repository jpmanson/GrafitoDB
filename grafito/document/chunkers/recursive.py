"""Recursive hierarchical character chunker (LangChain-style).

Mirrors the idea of ``RecursiveCharacterTextSplitter``: try coarse separators
first (paragraphs → lines → words → characters), merge small pieces up to
``max_size``, and recurse on oversized pieces with finer separators.

Differences from LangChain that matter for Grafito:

- Always returns :class:`~grafito.document.types.ChunkSpec` with **exact**
  ``char_start`` / ``char_end`` (required for pack overlap merge).
- Does **not** strip whitespace by default (stripping breaks offsets).
- Length is measured with ``length_function`` (default ``len``); pass a token
  counter for token budgets without depending on tiktoken.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from ..types import ChunkSpec

# Default hierarchy: paragraph → line → word → character (hard cut).
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]

# Small language presets (subset of LangChain's Language enum). Regex flags
# are applied only when ``is_separator_regex=True`` (see ``from_language``).
_LANGUAGE_SEPARATORS: dict[str, list[str]] = {
    "python": [
        "\nclass ",
        "\ndef ",
        "\n\tdef ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "markdown": [
        r"\n#{1,6} ",
        "```\n",
        r"\n\*\*\*+\n",
        r"\n---+\n",
        r"\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "html": [
        "<body",
        "<div",
        "<p",
        "<br",
        "<li",
        "<h1",
        "<h2",
        "<h3",
        "<h4",
        "<h5",
        "<h6",
        "<span",
        "<table",
        "<tr",
        "<td",
        "<th",
        "<ul",
        "<ol",
        "<header",
        "<footer",
        "<nav",
        "<head",
        "<style",
        "<script",
        "<meta",
        "<title",
        "",
    ],
    "javascript": [
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\nclass ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\ndefault ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "typescript": [
        "\nenum ",
        "\ninterface ",
        "\nnamespace ",
        "\ntype ",
        "\nclass ",
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\ndefault ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "go": [
        "\nfunc ",
        "\nvar ",
        "\nconst ",
        "\ntype ",
        "\nif ",
        "\nfor ",
        "\nswitch ",
        "\ncase ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "rust": [
        "\nfn ",
        "\nconst ",
        "\nlet ",
        "\nif ",
        "\nwhile ",
        "\nfor ",
        "\nloop ",
        "\nmatch ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "java": [
        "\nclass ",
        "\npublic ",
        "\nprotected ",
        "\nprivate ",
        "\nstatic ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "latex": [
        r"\n\\chapter{",
        r"\n\\section{",
        r"\n\\subsection{",
        r"\n\\subsubsection{",
        r"\n\\begin{enumerate}",
        r"\n\\begin{itemize}",
        r"\n\\begin{description}",
        r"\n\\begin{list}",
        r"\n\\begin{quote}",
        r"\n\\begin{quotation}",
        r"\n\\begin{verse}",
        r"\n\\begin{verbatim}",
        r"\n\\begin{align}",
        "$$",
        "$",
        " ",
        "",
    ],
}


class RecursiveChunker:
    """Split text by recursively trying a hierarchy of separators.

    Args:
        max_size: Maximum chunk length (units of ``length_function``).
        overlap: Soft overlap retained from the previous piece list when
            flushing a full chunk (same units). Must be ``< max_size``.
        separators: Ordered from coarsest to finest. The empty string ``""``
            means hard character cut (always last by default).
        keep_separator: If true, attach the matched separator to the **start**
            of the following piece (LangChain default for recursive).
        is_separator_regex: Treat separator strings as regex patterns.
        length_function: ``str -> int`` size measure (default ``len``).
        name: Strategy name stored on specs.
    """

    def __init__(
        self,
        max_size: int = 1200,
        overlap: int = 0,
        *,
        separators: Sequence[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_regex: bool = False,
        length_function: Callable[[str], int] = len,
        name: str = "recursive",
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if overlap < 0 or overlap >= max_size:
            raise ValueError("overlap must be >= 0 and < max_size")
        self.max_size = max_size
        self.overlap = overlap
        self.separators = list(separators) if separators is not None else list(DEFAULT_SEPARATORS)
        if not self.separators:
            raise ValueError("separators must be non-empty")
        # Normalize bool keep_separator to start/end (LangChain: True == start).
        if isinstance(keep_separator, bool):
            self.keep_separator: bool | Literal["start", "end"] = (
                "start" if keep_separator else False
            )
        else:
            self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        self.length_function = length_function
        self.name = name

    @classmethod
    def from_language(
        cls,
        language: str,
        max_size: int = 1200,
        overlap: int = 0,
        **kwargs: Any,
    ) -> RecursiveChunker:
        """Build a chunker with language-specific separators.

        Supported: ``python``, ``markdown``, ``html``, ``javascript``,
        ``typescript``, ``go``, ``rust``, ``java``, ``latex``.
        """
        key = language.lower().strip()
        # Common aliases
        aliases = {
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "md": "markdown",
        }
        key = aliases.get(key, key)
        seps = _LANGUAGE_SEPARATORS.get(key)
        if seps is None:
            supported = ", ".join(sorted(_LANGUAGE_SEPARATORS))
            raise ValueError(f"unknown language {language!r}; supported: {supported}")
        # Language presets use regex-shaped patterns for markdown/latex; let an
        # explicit is_separator_regex override, and always pop it so it is not
        # also forwarded via kwargs (which would duplicate the argument).
        use_regex = kwargs.pop("is_separator_regex", key in ("markdown", "latex"))
        return cls(
            max_size=max_size,
            overlap=overlap,
            separators=seps,
            is_separator_regex=use_regex,
            name=kwargs.pop("name", f"recursive:{key}"),
            **kwargs,
        )

    def split(self, text: str, *, meta: dict[str, Any] | None = None) -> list[ChunkSpec]:
        if not text:
            return []
        pieces = self._split_text(text, self.separators)
        return self._to_specs(text, pieces)

    # ------------------------------------------------------------------ internals

    def _len(self, s: str) -> int:
        return self.length_function(s)

    def _split_text(self, text: str, separators: Sequence[str]) -> list[str]:
        """LangChain-style recursive split + merge (returns text pieces)."""
        final_chunks: list[str] = []
        # Pick the first separator that appears in the text (or last).
        separator = separators[-1]
        new_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            pattern = sep if self.is_separator_regex else re.escape(sep)
            if re.search(pattern, text):
                separator = sep
                new_separators = list(separators[i + 1 :])
                break

        splits = self._split_with_separator(text, separator)

        # Merge small splits; recurse on oversized ones.
        good: list[str] = []
        merge_sep = "" if self.keep_separator else separator
        for s in splits:
            if self._len(s) < self.max_size:
                good.append(s)
            else:
                if good:
                    final_chunks.extend(self._merge_splits(good, merge_sep))
                    good = []
                if not new_separators:
                    # No finer separator: hard-cut by characters if still too big.
                    if self._len(s) > self.max_size and separator == "":
                        final_chunks.extend(self._hard_cut(s))
                    else:
                        final_chunks.append(s)
                else:
                    final_chunks.extend(self._split_text(s, new_separators))
        if good:
            final_chunks.extend(self._merge_splits(good, merge_sep))
        return final_chunks

    def _split_with_separator(self, text: str, separator: str) -> list[str]:
        if separator == "":
            return list(text)  # character-level

        pattern = separator if self.is_separator_regex else re.escape(separator)
        keep = self.keep_separator
        if not keep:
            return [s for s in re.split(pattern, text) if s]

        # Keep separators in the result via capturing group.
        parts = re.split(f"({pattern})", text)
        if len(parts) == 1:
            return [p for p in parts if p]

        out: list[str] = []
        if keep == "end":
            # piece + sep pairs, leftover tail
            i = 0
            while i < len(parts):
                if i + 1 < len(parts):
                    piece = parts[i] + parts[i + 1]
                    if piece:
                        out.append(piece)
                    i += 2
                else:
                    if parts[i]:
                        out.append(parts[i])
                    i += 1
        else:
            # keep == "start" (default): first bare piece, then sep+piece pairs
            if parts[0]:
                out.append(parts[0])
            i = 1
            while i < len(parts):
                if i + 1 < len(parts):
                    piece = parts[i] + parts[i + 1]
                    if piece:
                        out.append(piece)
                    i += 2
                else:
                    if parts[i]:
                        out.append(parts[i])
                    i += 1
        return out

    def _merge_splits(self, splits: Sequence[str], separator: str) -> list[str]:
        """Pack small splits into chunks ≤ max_size, with optional overlap."""
        sep_len = self._len(separator) if separator else 0
        docs: list[str] = []
        current: list[str] = []
        total = 0

        for piece in splits:
            plen = self._len(piece)
            extra = sep_len if current else 0
            if total + plen + extra > self.max_size and current:
                joined = self._join(current, separator)
                if joined is not None:
                    docs.append(joined)
                # Forward-progress guard: always drop at least the first piece so the
                # window's left edge advances every flush (prevents a stall when the
                # overlap budget alone would retain the whole window, e.g. a token
                # nearly as large as max_size).
                total -= self._len(current[0]) + (sep_len if len(current) > 1 else 0)
                current = current[1:]
                # Then pop from the front until under overlap budget (or room for piece).
                while current and (
                    total > self.overlap
                    or (
                        total + plen + (sep_len if current else 0) > self.max_size
                        and total > 0
                    )
                ):
                    total -= self._len(current[0]) + (sep_len if len(current) > 1 else 0)
                    current = current[1:]
            current.append(piece)
            total += plen + (sep_len if len(current) > 1 else 0)

        joined = self._join(current, separator)
        if joined is not None:
            docs.append(joined)
        return docs

    @staticmethod
    def _join(parts: Sequence[str], separator: str) -> str | None:
        if not parts:
            return None
        text = separator.join(parts)
        return text if text else None

    def _hard_cut(self, text: str) -> list[str]:
        """Character hard-cut when no separator remains (unit = length_function)."""
        if self._len(text) <= self.max_size:
            return [text]
        # When length_function is len, cut by chars; otherwise walk and grow.
        if self.length_function is len:
            step = max(1, self.max_size - self.overlap)
            out: list[str] = []
            i = 0
            n = len(text)
            while i < n:
                j = min(n, i + self.max_size)
                out.append(text[i:j])
                if j >= n:
                    break
                i = max(i + 1, j - self.overlap)
            return out

        # Generic length_function: binary search end positions.
        out = []
        start = 0
        n = len(text)
        while start < n:
            lo, hi = start + 1, n
            best = start + 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if self._len(text[start:mid]) <= self.max_size:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            end = max(best, start + 1)
            out.append(text[start:end])
            if end >= n:
                break
            if self.overlap <= 0:
                start = end
            else:
                # Walk back ~overlap units
                back = end
                while back > start and self._len(text[back:end]) < self.overlap:
                    back -= max(1, (end - start) // 20)
                start = min(back, end - 1) if back < end else end
        return out

    def _to_specs(self, text: str, pieces: Sequence[str]) -> list[ChunkSpec]:
        """Map piece strings back onto the original text for exact offsets.

        Pieces are located in reading order from a **monotonic** floor: each chunk
        starts strictly after the previous one, so repeated content (identical
        piece strings) maps to successive occurrences instead of collapsing onto
        the same span (which would drop the tail). With overlap a chunk may still
        start before the previous chunk *ends* — just not before it *starts*.
        """
        specs: list[ChunkSpec] = []
        floor = 0  # next search never starts before the previous chunk's start + 1
        for ord_i, piece in enumerate(pieces):
            if not piece:
                continue
            idx = text.find(piece, floor)
            if idx < 0:
                idx = text.find(piece)  # fallback: locate anywhere in reading order
            if idx < 0:
                # Reassembled piece not literally present (e.g. keep_separator=False
                # with a regex separator): place after the floor, best effort.
                idx = min(floor, len(text))
                end = min(len(text), idx + len(piece))
                specs.append(
                    ChunkSpec(text=piece, ord=ord_i, char_start=idx,
                              char_end=end, strategy=self.name)
                )
            else:
                end = idx + len(piece)
                specs.append(
                    ChunkSpec(text=text[idx:end], ord=ord_i, char_start=idx,
                              char_end=end, strategy=self.name)
                )
            floor = idx + 1  # strictly advance so identical strings don't collapse
        # Re-number ord contiguously (empty pieces dropped).
        for i, s in enumerate(specs):
            s.ord = i
        return specs
