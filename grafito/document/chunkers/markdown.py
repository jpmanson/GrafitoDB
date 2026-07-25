"""Markdown heading-aware chunker with fixed overflow for large sections."""

from __future__ import annotations

import re
from typing import Any

from ..types import ChunkSpec
from .fixed import FixedChunker

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownChunker:
    """Split markdown by ATX headings; large sections overflow via FixedChunker.

    Section **preambles** (text under a heading before the next child heading)
    become their own passages so they remain searchable.
    """

    def __init__(
        self,
        max_chars: int = 1200,
        overlap: int = 0,
        *,
        name: str = "markdown",
    ) -> None:
        self.max_chars = max_chars
        self.overlap = overlap
        self.name = name
        self._overflow = FixedChunker(
            max_size=max_chars,
            overlap=overlap,
            unit="chars",
            name=f"{name}+fixed",
        )

    def split(self, text: str, *, meta: dict[str, Any] | None = None) -> list[ChunkSpec]:
        if not text:
            return []
        lines = text.splitlines(keepends=True)
        headings: list[tuple[int, str, int]] = []
        for i, line in enumerate(lines):
            m = _HEADING_RE.match(line.rstrip("\n"))
            if m:
                headings.append((len(m.group(1)), m.group(2).strip(), i))

        if not headings:
            return self._overflow.split(text)

        # Leading text before first heading
        raw_segments: list[tuple[str, str | None, int, int]] = []
        # (body, section_path, char_start, char_end)

        def line_offset(line_idx: int) -> int:
            return sum(len(lines[j]) for j in range(line_idx))

        if headings[0][2] > 0:
            start = 0
            end = headings[0][2]
            body = "".join(lines[start:end])
            if body.strip():
                raw_segments.append((body, None, line_offset(start), line_offset(end)))

        for idx, (level, title, start_line) in enumerate(headings):
            # content after this heading until next heading at same or higher level
            end_line = len(lines)
            for j in range(idx + 1, len(headings)):
                if headings[j][0] <= level:
                    end_line = headings[j][2]
                    break
            # preamble: until first deeper child heading
            child_start = end_line
            for j in range(idx + 1, len(headings)):
                if headings[j][2] >= end_line:
                    break
                if headings[j][0] > level:
                    child_start = headings[j][2]
                    break
            content_start = start_line + 1
            if content_start >= child_start:
                continue
            body = "".join(lines[content_start:child_start])
            if not body.strip():
                continue
            abs_start = line_offset(content_start)
            abs_end = line_offset(child_start)
            # trim strip while keeping offsets approximate for half-open range of stripped text
            lead = len(body) - len(body.lstrip())
            trail = len(body) - len(body.rstrip())
            stripped = body.strip()
            raw_segments.append(
                (
                    stripped,
                    title,
                    abs_start + lead,
                    abs_end - trail,
                )
            )

        specs: list[ChunkSpec] = []
        ord_i = 0
        for body, path, c0, c1 in raw_segments:
            if len(body) <= self.max_chars:
                specs.append(
                    ChunkSpec(
                        text=body,
                        ord=ord_i,
                        char_start=c0,
                        char_end=c1,
                        heading=path,
                        section_path=path,
                        strategy=self.name,
                    )
                )
                ord_i += 1
                continue
            # Overflow with fixed windows; remap offsets relative to segment
            for part in self._overflow.split(body):
                off0 = part.char_start or 0
                off1 = part.char_end or len(part.text)
                specs.append(
                    ChunkSpec(
                        text=part.text,
                        ord=ord_i,
                        char_start=c0 + off0,
                        char_end=c0 + off1,
                        heading=path,
                        section_path=path,
                        strategy=part.strategy or self.name,
                    )
                )
                ord_i += 1
        # Re-number ord to be contiguous
        for i, s in enumerate(specs):
            s.ord = i
        return specs
