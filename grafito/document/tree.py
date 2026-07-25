"""Markdown → hierarchical SectionSpec tree (+ passage chunks per section)."""

from __future__ import annotations

import re
from typing import Any

from .chunkers.fixed import FixedChunker
from .types import ChunkSpec, SectionSpec

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _line_offsets(lines: list[str]) -> list[int]:
    """Start char offset for each line index."""
    offsets = [0]
    for line in lines[:-1]:
        offsets.append(offsets[-1] + len(line))
    return offsets


def build_markdown_tree(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 0,
    strategy: str = "markdown-tree",
) -> list[SectionSpec]:
    """Build a forest of SectionSpecs from ATX markdown.

    Each section may hold:
    - ``chunks``: preamble (and overflow pieces) as ChunkSpecs (indexable)
    - ``children``: nested sections

    Section nodes themselves do not carry full subtree body text.
    """
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    headings: list[tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.rstrip("\n"))
        if m:
            headings.append((len(m.group(1)), m.group(2).strip(), i))

    overflow = FixedChunker(
        max_size=max_chars,
        overlap=overlap,
        unit="chars",
        name=f"{strategy}+fixed",
    )

    if not headings:
        # Synthetic root holding whole document as chunks
        root = SectionSpec(
            title="(document)",
            local_ord=0,
            level=0,
            node_key="0000",
            char_start=0,
            char_end=len(text),
        )
        root.chunks = _chunk_span(text, 0, len(text), None, overflow, strategy)
        return [root]

    roots: list[SectionSpec] = []
    stack: list[SectionSpec] = []
    key_counter = 0

    def next_key() -> str:
        nonlocal key_counter
        key_counter += 1
        return f"{key_counter:04d}"

    # Leading prose before first heading → synthetic root section level 0
    if headings[0][2] > 0:
        start_off = 0
        end_off = offsets[headings[0][2]]
        body = text[start_off:end_off]
        if body.strip():
            lead = SectionSpec(
                title="(preamble)",
                local_ord=0,
                level=0,
                node_key=next_key(),
                char_start=start_off,
                char_end=end_off,
            )
            lead.chunks = _chunk_span(text, start_off, end_off, None, overflow, strategy)
            roots.append(lead)

    for idx, (level, title, start_line) in enumerate(headings):
        # Section span: after this heading until next heading at same/higher level
        end_line = len(lines)
        for j in range(idx + 1, len(headings)):
            if headings[j][0] <= level:
                end_line = headings[j][2]
                break
        # Preamble: until first deeper child
        child_start = end_line
        for j in range(idx + 1, len(headings)):
            if headings[j][2] >= end_line:
                break
            if headings[j][0] > level:
                child_start = headings[j][2]
                break

        content_start = start_line + 1
        sec_start = offsets[start_line]
        sec_end = offsets[end_line] if end_line < len(offsets) else len(text)
        pre_start = offsets[content_start] if content_start < len(offsets) else sec_end
        pre_end = offsets[child_start] if child_start < len(offsets) else sec_end

        section = SectionSpec(
            title=title,
            local_ord=0,
            level=level,
            node_key=next_key(),
            char_start=sec_start,
            char_end=sec_end,
        )
        if content_start < child_start:
            section.chunks = _chunk_span(
                text, pre_start, pre_end, title, overflow, strategy
            )

        # Attach to parent in stack
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            parent = stack[-1]
            section.local_ord = len(parent.children)
            parent.children.append(section)
        else:
            section.local_ord = len(roots)
            roots.append(section)
        stack.append(section)

    _assign_paths(roots, prefix=None)
    _renumber_global_chunk_ords(roots)
    return roots


def flatten_chunks(sections: list[SectionSpec]) -> list[ChunkSpec]:
    """Depth-first passage list with global_seq in ``ord``."""
    out: list[ChunkSpec] = []

    def walk(sec: SectionSpec) -> None:
        for ch in sec.chunks:
            out.append(ch)
        for child in sec.children:
            walk(child)

    for root in sections:
        walk(root)
    for i, c in enumerate(out):
        c.ord = i
    return out


def _chunk_span(
    full: str,
    start: int,
    end: int,
    heading: str | None,
    overflow: FixedChunker,
    strategy: str,
) -> list[ChunkSpec]:
    raw = full[start:end]
    if not raw.strip():
        return []
    lead = len(raw) - len(raw.lstrip())
    trail = len(raw) - len(raw.rstrip())
    stripped = raw.strip()
    c0 = start + lead
    c1 = end - trail
    path = heading
    if len(stripped) <= overflow.max_size:
        return [
            ChunkSpec(
                text=stripped,
                ord=0,
                char_start=c0,
                char_end=c1,
                heading=heading,
                section_path=path,
                strategy=strategy,
            )
        ]
    specs: list[ChunkSpec] = []
    for part in overflow.split(stripped):
        off0 = part.char_start or 0
        off1 = part.char_end or len(part.text)
        specs.append(
            ChunkSpec(
                text=part.text,
                ord=0,
                char_start=c0 + off0,
                char_end=c0 + off1,
                heading=heading,
                section_path=path,
                strategy=part.strategy or strategy,
            )
        )
    return specs


def _assign_paths(sections: list[SectionSpec], prefix: str | None) -> None:
    for sec in sections:
        path = sec.title if prefix is None else f"{prefix} / {sec.title}"
        if sec.title in ("(preamble)", "(document)"):
            path = prefix  # type: ignore[assignment]
        for ch in sec.chunks:
            ch.section_path = path or ch.heading
            ch.heading = sec.title if sec.level > 0 else ch.heading
        for child in sec.children:
            _assign_paths([child], path if sec.level > 0 else prefix)


def _renumber_global_chunk_ords(sections: list[SectionSpec]) -> None:
    flatten_chunks(sections)


def sections_to_toc_dict(sections: list[SectionSpec]) -> list[dict[str, Any]]:
    """JSON-serializable ToC (no passage bodies)."""

    def one(sec: SectionSpec) -> dict[str, Any]:
        n_chunks = len(sec.chunks)
        if sec.metadata and "n_chunks" in sec.metadata:
            n_chunks = int(sec.metadata["n_chunks"])
        return {
            "title": sec.title,
            "node_key": sec.node_key,
            "level": sec.level,
            "local_ord": sec.local_ord,
            "summary": sec.summary,
            "char_start": sec.char_start,
            "char_end": sec.char_end,
            "n_chunks": n_chunks,
            "children": [one(c) for c in sec.children],
        }

    return [one(s) for s in sections]
