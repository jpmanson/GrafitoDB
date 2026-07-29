"""Phase 3: hybrid RRF, semantic chunker, tree_select, Cypher string vector search."""

import hashlib
import re

import pytest

from grafito import GrafitoDatabase
from grafito.document import (
    DocumentIngestor,
    MarkdownChunker,
    SemanticBreakpointChunker,
    TitleContextEnricher,
    rrf_fuse,
)
from grafito.document.ingest import MANAGED_BY
from grafito.embedding_functions.base import EmbeddingFunction


def _bucket(token: str, dim: int) -> int:
    """Stable token bucket: built-in ``hash()`` is randomized per process."""
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "little") % dim


class ToyEmbedder(EmbeddingFunction):
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        out = []
        for text in input:
            vec = [0.0] * self._dim
            for i, tok in enumerate(re.findall(r"[a-z0-9]+", text.lower())):
                vec[_bucket(tok, self._dim)] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out

    @staticmethod
    def name() -> str:
        return "toy_p3"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    @staticmethod
    def build_from_config(config):
        return ToyEmbedder()

    def get_config(self):
        return {"dim": self._dim}

    @staticmethod
    def validate_config(config):
        return None

    @property
    def dimension(self) -> int:
        return self._dim


def test_rrf_fuse_basic():
    a = ["x", "y", "z"]
    b = ["y", "z", "w"]
    fused = rrf_fuse([a, b], k=60, limit=3)
    ids = [item for item, _ in fused]
    assert ids[0] == "y"  # appears in both lists near top
    assert "x" in ids or "z" in ids


def test_rrf_empty_list():
    assert rrf_fuse([["a"], []])[0][0] == "a"
    assert rrf_fuse([]) == []


def test_semantic_breakpoint_chunker_contiguous():
    emb = ToyEmbedder()
    # Two thematically different paragraphs → likely break
    text = (
        "Apples and oranges grow on trees in orchards. Fruit harvest season is autumn. "
        "Neural networks train on GPUs with backpropagation. Transformers use attention layers."
    )
    chunker = SemanticBreakpointChunker(emb, threshold=0.15, min_chars=20, max_chars=500)
    specs = chunker.split(text)
    assert specs
    # Contiguous: char ranges non-decreasing and non-overlapping interiors
    for i in range(len(specs) - 1):
        assert specs[i].char_end is not None and specs[i + 1].char_start is not None
        assert specs[i].char_end <= specs[i + 1].char_end


def test_title_context_enricher():
    from grafito.document.types import ChunkSpec

    specs = [
        ChunkSpec(text="body", ord=0, heading="OAuth", section_path="Security / OAuth"),
    ]
    out = TitleContextEnricher().enrich(specs, document_title="Runbook")
    assert out[0].context and "Runbook" in out[0].context
    assert "OAuth" in out[0].text_for_embedding()


def test_tree_select_filters_keys():
    db = GrafitoDatabase(":memory:")
    emb = ToyEmbedder()
    db.create_vector_index("docs", dim=emb.dimension, embedding_function=emb)
    ing = DocumentIngestor(db, chunker=MarkdownChunker(), embed_index="docs")
    md = "# A\n\nintro\n\n## B\n\nbody b\n\n## C\n\nbody c\n"
    ing.ingest(md, document_key="t")
    toc = ing.toc("t", as_dict=True)
    keys = []

    def walk(nodes):
        for n in nodes:
            keys.append(n["node_key"])
            walk(n.get("children") or [])

    walk(toc)

    def fake_llm(prompt: str) -> str:
        # Include one valid and one invalid key
        return f'["{keys[0]}", "9999", "{keys[-1]}"]'

    selected = ing.tree_select("t", "query", fake_llm)
    assert keys[0] in selected
    assert "9999" not in selected
    bodies = ing.load_sections("t", selected)
    assert bodies
    db.close()


def test_hybrid_search_rrf_when_fts_configured():
    db = GrafitoDatabase(":memory:")
    if not db.has_fts5():
        pytest.skip("FTS5 not available")
    emb = ToyEmbedder()
    db.create_vector_index("docs", dim=emb.dimension, embedding_function=emb)
    ing = DocumentIngestor(
        db,
        chunker=MarkdownChunker(),
        embed_index="docs",
        configure_fts=True,
    )
    text = (
        "# Runbook\n\n"
        "Fix connection pool timeout by increasing max_connections.\n\n"
        "## Other\n\nUnrelated garden vegetables and soil.\n"
    )
    ing.ingest(text, document_key="hyb", embed=True)
    hits = ing.hybrid_search("connection pool timeout", k=3)
    assert hits
    assert any("timeout" in h.node.properties.get("text", "").lower() for h in hits)
    db.close()


def test_cypher_vector_search_accepts_string_query():
    db = GrafitoDatabase(":memory:")
    emb = ToyEmbedder()
    db.create_vector_index("docs", dim=emb.dimension, embedding_function=emb)
    n = db.create_node(labels=["Chunk"], properties={"text": "hello world"})
    db.upsert_embeddings([n.id], ["hello world"], index="docs")
    rows = db.execute(
        "CALL db.vector.search('docs', $q, 3) YIELD node, score RETURN node, score",
        {"q": "hello"},
    )
    assert rows
    node = rows[0]["node"]
    node_id = node.id if hasattr(node, "id") else node.get("id")
    assert node_id == n.id
    db.close()


def test_hybrid_fallback_without_fts_hits():
    db = GrafitoDatabase(":memory:")
    emb = ToyEmbedder()
    db.create_vector_index("docs", dim=emb.dimension, embedding_function=emb)
    # No FTS config
    ing = DocumentIngestor(db, chunker=MarkdownChunker(), embed_index="docs")
    ing.ingest("# A\n\nalpha beta gamma uniquephrase\n", document_key="nof", embed=True)
    hits = ing.hybrid_search("uniquephrase", k=2)
    assert hits
    db.close()
