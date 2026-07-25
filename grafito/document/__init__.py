"""Document → graph chunking helper (passages as nodes, 1 node = 1 vector)."""

from .chunkers.fixed import FixedChunker
from .chunkers.markdown import MarkdownChunker
from .ingest import DocumentIngestor
from .tree import build_markdown_tree, flatten_chunks
from .types import (
    ChunkSpec,
    ExpandResult,
    IngestResult,
    PackedContext,
    PackedSegment,
    SearchHit,
    SectionSpec,
)

__all__ = [
    "ChunkSpec",
    "DocumentIngestor",
    "ExpandResult",
    "FixedChunker",
    "IngestResult",
    "MarkdownChunker",
    "PackedContext",
    "PackedSegment",
    "SearchHit",
    "SectionSpec",
    "build_markdown_tree",
    "flatten_chunks",
]
