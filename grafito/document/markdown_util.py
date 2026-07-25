"""Shared markdown line scanning (headings outside fenced code blocks)."""

from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# CommonMark: optional ≤3 spaces, then 3+ backticks or tildes
_FENCE_OPEN_RE = re.compile(r"^ {0,3}([`~]{3,})(.*)$")


def iter_atx_headings(lines: list[str]) -> list[tuple[int, str, int]]:
    """Return ``(level, title, line_index)`` for ATX headings outside fences.

    Ignores lines inside fenced code blocks (`` ``` `` / ``~~~``), including
    shell comments that look like headings (e.g. ``# apt update`` in a bash fence).
    """
    headings: list[tuple[int, str, int]] = []
    in_fence = False
    fence_char: str | None = None
    fence_len = 0

    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        fence = _FENCE_OPEN_RE.match(stripped)
        if fence:
            marker = fence.group(1)
            ch = marker[0]
            n = len(marker)
            # Info string after opening fence; closing fence must not have content
            # other than optional whitespace (CommonMark-ish).
            info = fence.group(2)
            if not in_fence:
                in_fence = True
                fence_char = ch
                fence_len = n
            elif ch == fence_char and n >= fence_len and info.strip() == "":
                in_fence = False
                fence_char = None
                fence_len = 0
            # Lines that open/close fences are never headings.
            continue

        if in_fence:
            continue

        m = _HEADING_RE.match(stripped)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip(), i))

    return headings
