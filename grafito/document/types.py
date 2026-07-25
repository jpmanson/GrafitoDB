"""Core types for the document chunking helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..models import Node


@dataclass
class ChunkSpec:
    """One contiguous passage produced by a chunker (not yet a graph node)."""

    text: str
    ord: int
    char_start: int | None = None
    char_end: int | None = None
    token_count: int | None = None
    page_number: int | None = None
    heading: str | None = None
    section_path: str | None = None
    strategy: str | None = None
    title: str | None = None
    context: str | None = None
    summary: str | None = None
    keywords: list[str] | None = None
    questions: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def text_for_embedding(self) -> str:
        """Prefer contextual retrieval shape when ``context`` is set."""
        if self.context:
            return f"{self.context}\n\n{self.text}"
        return self.text


@dataclass
class IngestResult:
    """Outcome of :meth:`DocumentIngestor.ingest` / ``replace``."""

    owner_document_id: int
    version_id: int | None
    generation: int
    passage_ids: list[int]
    n_passages: int
    skipped: bool = False
    fingerprint: str | None = None
    document_key: str | None = None


@dataclass
class SearchHit:
    node: Node
    score: float
    owner_document_id: int | None = None
    generation: int | None = None
    global_seq: int | None = None


@dataclass
class ExpandResult:
    center: Node
    passages: list[Node]
    parent: Node | None = None
    version: Node | None = None


@dataclass
class PackedSegment:
    text: str
    node_id: int
    document_id: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_path: str | None = None
    global_seq: int | None = None


@dataclass
class PackedContext:
    """Structured pack result; ``text`` is derived from ``segments``."""

    segments: list[PackedSegment] = field(default_factory=list)
    text: str = ""
    truncated: bool = False
    order: Literal["reading", "score"] = "reading"
