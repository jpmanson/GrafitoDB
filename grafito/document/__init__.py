"""Document → graph chunking helper (passages as nodes, 1 node = 1 vector)."""

from .chunkers.chonkie_adapter import ChonkieChunker
from .chunkers.fixed import FixedChunker
from .chunkers.markdown import MarkdownChunker
from .chunkers.recursive import DEFAULT_SEPARATORS, RecursiveChunker
from .chunkers.semantic import SemanticBreakpointChunker
from .enrich import TitleContextEnricher
from .hybrid import rrf_fuse
from .ingest import DocumentIngestor
from .tools import DocumentTools
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
    "DEFAULT_SEPARATORS",
    "ChonkieChunker",
    "ChunkSpec",
    "DocumentIngestor",
    "DocumentTools",
    "ExpandResult",
    "FixedChunker",
    "IngestResult",
    "MarkdownChunker",
    "PackedContext",
    "PackedSegment",
    "RecursiveChunker",
    "SearchHit",
    "SectionSpec",
    "SemanticBreakpointChunker",
    "TitleContextEnricher",
    "build_markdown_tree",
    "flatten_chunks",
    "rrf_fuse",
]
