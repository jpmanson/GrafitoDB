"""Tests for grafito.document.DocumentTools — the read-only passage tool tier.

These hang on a plain GrafitoDatabase + DocumentIngestor (no OKF): the passage
retrieval tier serves an arbitrary graph, the same way GraphTools does.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from grafito import GrafitoDatabase
from grafito.document import DocumentIngestor, DocumentTools, MarkdownChunker
from grafito.embedding_functions.base import EmbeddingFunction


def _bucket(token: str, dim: int) -> int:
    """Stable token bucket.

    Python's built-in ``hash()`` is randomized per process (PYTHONHASHSEED), so
    using it here would make the toy embedding — and therefore every ranking
    assertion below — differ from run to run.
    """
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "little") % dim


class ToyEmbedder(EmbeddingFunction):
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        out = []
        for text in input:
            vec = [0.0] * self._dim
            for i, tok in enumerate(re.findall(r"[a-z0-9]+", text.lower())):
                vec[_bucket(tok, self._dim)] += 1.0 + (i % 3) * 0.01
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out

    @staticmethod
    def name() -> str:
        return "toy_document_tools"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    @staticmethod
    def build_from_config(config):
        return ToyEmbedder(dim=int(config.get("dim", 16)))

    def get_config(self):
        return {"dim": self._dim}

    @staticmethod
    def validate_config(config):
        return None

    @property
    def dimension(self) -> int:
        return self._dim


_DOC = """# Connection pooling

The pool caps concurrent connections. When it is exhausted, callers wait.

## Timeouts

A connection pool timeout means every pooled connection is checked out. Raise
the pool size or shorten slow queries to recover from timeouts.

## Retries

Retries with backoff smooth over transient exhaustion but do not fix a pool
that is chronically too small.
"""


@pytest.fixture
def tools():
    db = GrafitoDatabase(":memory:")
    emb = ToyEmbedder()
    db.create_vector_index("docs_chunks", dim=emb.dimension, embedding_function=emb)
    ing = DocumentIngestor(
        db,
        chunker=MarkdownChunker(max_chars=200, overlap=20),
        embed_index="docs_chunks",
    )
    result = ing.ingest(text=_DOC, title="Pooling", document_key="pooling")
    dt = DocumentTools(ing)
    dt._doc_id = result.owner_document_id  # test helper
    yield dt
    db.close()


def test_read_only_surface_excludes_write_tools(tools):
    names = tools.names if hasattr(tools, "names") else [
        s["function"]["name"] for s in tools.schemas
    ]
    assert set(names) == {
        "document_context",
        "document_search",
        "document_expand",
        "document_toc",
        "document_load_sections",
    }
    for forbidden in ("document_ingest", "document_replace", "document_delete"):
        assert forbidden not in tools.enabled
        assert forbidden not in names


def test_unknown_tool_is_a_json_error(tools):
    assert "error" in json.loads(tools.call("nope", {}))
    # A write name the tier does not expose must be rejected too, not merely absent.
    assert "error" in json.loads(tools.call("document_ingest", {"text": "x"}))


def test_document_search_returns_ranked_passages(tools):
    hits = json.loads(tools.call("document_search", {"query": "pool timeout", "k": 3}))
    assert hits, "expected at least one passage hit"
    top = hits[0]
    assert "score" in top and "owner_document_id" in top and "global_seq" in top
    assert top["owner_document_id"] == tools._doc_id
    # the timeouts passage should surface for this query
    assert any("timeout" in h["properties"].get("text", "").lower() for h in hits)


def test_document_expand_windows_by_reading_order(tools):
    hits = json.loads(tools.call("document_search", {"query": "timeout", "k": 1}))
    node_id = hits[0]["id"]
    expanded = json.loads(
        tools.call("document_expand", {"node_id": node_id, "window": 1})
    )
    seqs = [p["properties"].get("global_seq") for p in expanded["passages"]]
    assert node_id in [p["id"] for p in expanded["passages"]]
    # passages come back in reading order, contiguous around the centre
    assert seqs == sorted(seqs)
    assert expanded["parent"] is not None


def test_document_context_is_grounded_and_budgeted(tools):
    out = json.loads(
        tools.call("document_context", {"query": "pool timeout", "k": 3, "max_tokens": 400})
    )
    assert out["text"], "expected packed context text"
    assert out["citations"], "expected citations"
    assert tools._doc_id in out["documents"]
    # every citation points at a real passage node with offsets
    for cite in out["citations"]:
        assert "node_id" in cite


def test_document_context_no_hits_is_empty_not_error(tools):
    # empty query string still runs search; a miss must not raise
    out = json.loads(tools.call("document_context", {"query": "zzzz nonexistent xyzzy"}))
    assert "error" not in out
    assert "text" in out and "documents" in out


def test_document_toc_and_load_sections_round_trip(tools):
    toc = json.loads(tools.call("document_toc", {"document_ref": tools._doc_id}))
    assert isinstance(toc, list)
    # accept both int id and string document_key
    toc_by_key = json.loads(tools.call("document_toc", {"document_ref": "pooling"}))
    assert toc_by_key == toc


def test_document_ref_accepts_numeric_string(tools):
    by_int = json.loads(tools.call("document_toc", {"document_ref": tools._doc_id}))
    by_str = json.loads(tools.call("document_toc", {"document_ref": str(tools._doc_id)}))
    assert by_int == by_str


# --- write tier (enable_writes) -----------------------------------------------


@pytest.fixture
def rw_tools(tools):
    # reuse the read fixture's ingestor, but with the write tier enabled
    dt = DocumentTools(tools.ing, enable_writes=True)
    dt._doc_id = tools._doc_id
    return dt


def test_write_tools_absent_unless_enabled(tools, rw_tools):
    read_names = [s["function"]["name"] for s in tools.schemas]
    for w in ("document_ingest", "document_replace", "document_delete"):
        assert w not in read_names
        assert w in rw_tools.enabled
    rw_names = [s["function"]["name"] for s in rw_tools.schemas]
    assert rw_names[:5] == list(tools._READ)  # read tools still come first, same order


def test_document_ingest_creates_a_searchable_document(rw_tools):
    out = json.loads(
        rw_tools.call(
            "document_ingest",
            {"text": "# Backups\n\nNightly backups run at 02:00 UTC and are retained 30 days.",
             "title": "Backups", "document_key": "backups"},
        )
    )
    assert "error" not in out
    assert out["n_passages"] >= 1
    assert out["document_key"] == "backups"
    # the new document is retrievable
    hits = json.loads(rw_tools.call("document_search", {"query": "nightly backup retention", "k": 3}))
    assert any(h["owner_document_id"] == out["owner_document_id"] for h in hits)


def test_document_ingest_is_idempotent_by_key(rw_tools):
    first = json.loads(rw_tools.call("document_ingest", {"text": "hello world", "document_key": "k1"}))
    again = json.loads(rw_tools.call("document_ingest", {"text": "hello world", "document_key": "k1"}))
    assert again["owner_document_id"] == first["owner_document_id"]
    assert again["skipped"] is True


def test_document_replace_bumps_generation(rw_tools):
    created = json.loads(rw_tools.call("document_ingest", {"text": "version one", "document_key": "r1"}))
    replaced = json.loads(
        rw_tools.call("document_replace", {"document_ref": "r1", "text": "version two updated"})
    )
    assert replaced["owner_document_id"] == created["owner_document_id"]
    assert replaced["generation"] > created["generation"]


def test_document_delete_removes_managed_subgraph(rw_tools):
    created = json.loads(rw_tools.call("document_ingest", {"text": "delete me please", "document_key": "d1"}))
    doc_id = created["owner_document_id"]
    out = json.loads(rw_tools.call("document_delete", {"document_ref": "d1"}))
    assert out["deleted"] is True
    # its passages are gone from search
    hits = json.loads(rw_tools.call("document_search", {"query": "delete me please", "k": 5}))
    assert all(h["owner_document_id"] != doc_id for h in hits)


def test_document_tools_do_not_drag_in_okf():
    import subprocess
    import sys

    code = (
        "import grafito.document.tools, sys; "
        "assert not any(m.startswith('grafito.okf') for m in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
