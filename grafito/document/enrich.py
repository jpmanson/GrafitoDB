"""Optional passage enrichment hooks (context / summary before embed)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import ChunkSpec


@runtime_checkable
class ChunkEnricher(Protocol):
    """Mutate or replace ChunkSpecs before graph write / embed."""

    def enrich(self, specs: list[ChunkSpec], *, document_title: str | None = None) -> list[ChunkSpec]:
        ...


class TitleContextEnricher:
    """Set ``context`` from document title + section heading for contextual retrieval."""

    def enrich(self, specs: list[ChunkSpec], *, document_title: str | None = None) -> list[ChunkSpec]:
        out: list[ChunkSpec] = []
        for s in specs:
            parts = []
            if document_title:
                parts.append(f"Document: {document_title}")
            if s.section_path or s.heading:
                parts.append(f"Section: {s.section_path or s.heading}")
            if parts:
                s.context = ". ".join(parts) + "."
            out.append(s)
        return out
