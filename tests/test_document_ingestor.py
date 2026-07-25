"""Tests for grafito.document DocumentIngestor MVP."""

import re

import pytest

from grafito import GrafitoDatabase
from grafito.document import DocumentIngestor, FixedChunker, MarkdownChunker
from grafito.document.ingest import MANAGED_BY
from grafito.embedding_functions.base import EmbeddingFunction


class ToyEmbedder(EmbeddingFunction):
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        out = []
        for text in input:
            vec = [0.0] * self._dim
            for i, tok in enumerate(re.findall(r"[a-z0-9]+", text.lower())):
                vec[hash(tok) % self._dim] += 1.0 + (i % 3) * 0.01
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out

    @staticmethod
    def name() -> str:
        return "toy_document"

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


@pytest.fixture
def db_ing():
    db = GrafitoDatabase(":memory:")
    emb = ToyEmbedder()
    db.create_vector_index("docs_chunks", dim=emb.dimension, embedding_function=emb)
    ing = DocumentIngestor(
        db,
        chunker=MarkdownChunker(max_chars=500, overlap=50),
        embed_index="docs_chunks",
    )
    yield db, ing
    db.close()


def test_fixed_chunker_chars_and_overlap():
    c = FixedChunker(max_size=10, overlap=4)
    specs = c.split("abcdefghijklmnopqrstuvwxyz")
    assert len(specs) >= 3
    assert specs[0].char_start == 0
    assert specs[0].text == "abcdefghij"
    # second window starts at step=6
    assert specs[1].char_start == 6


def test_markdown_preamble_passages():
    md = """# Security

Intro about authentication and session cookies.

## OAuth

OAuth body.

## MFA

MFA body.
"""
    specs = MarkdownChunker(max_chars=1000).split(md)
    texts = [s.text for s in specs]
    assert any("session cookies" in t for t in texts)
    assert any("OAuth body" in t for t in texts)


def test_ingest_search_expand_pack(db_ing):
    db, ing = db_ing
    md = """# Security

Intro about authentication and session cookies.

## OAuth

OAuth details and tokens.

## MFA

Multi factor devices.
"""
    result = ing.ingest(md, document_key="sec-1", title="Security")
    assert result.skipped is False
    assert result.n_passages >= 3
    assert result.generation == 1

    hits = ing.search("session cookies", k=3)
    assert hits
    assert "session cookies" in hits[0].node.properties["text"]

    expanded = ing.expand(hits[0].node, window=1)
    assert len(expanded.passages) >= 1
    seqs = [p.properties["global_seq"] for p in expanded.passages]
    assert seqs == sorted(seqs)

    packed = ing.pack(expanded, max_chars=2000, deduplicate_overlap=True)
    assert "session cookies" in packed.text
    assert packed.segments


def test_ingest_idempotent_skip(db_ing):
    db, ing = db_ing
    text = "Hello world document for fingerprint."
    r1 = ing.ingest(text, document_key="idem", embed=True)
    r2 = ing.ingest(text, document_key="idem", embed=True)
    assert r1.skipped is False
    assert r2.skipped is True
    assert r2.generation == r1.generation


def test_replace_generational_gc(db_ing):
    db, ing = db_ing
    r1 = ing.ingest("version one content alpha", document_key="gen", embed=True)
    r2 = ing.replace("gen", "version two content beta different", embed=True)
    assert r2.generation == r1.generation + 1
    old = db.match_nodes(
        properties={
            "managed_by": MANAGED_BY,
            "generation": r1.generation,
            "role": "passage",
        }
    )
    assert old == []
    new = db.match_nodes(
        properties={
            "managed_by": MANAGED_BY,
            "generation": r2.generation,
            "role": "passage",
        }
    )
    assert new


def test_replace_failure_keeps_previous_active(db_ing, monkeypatch):
    db, ing = db_ing
    r1 = ing.ingest("stable content that stays", document_key="fail", embed=True)

    def boom(*args, **kwargs):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(db, "upsert_embeddings", boom)
    with pytest.raises(RuntimeError):
        ing.replace("fail", "new content that should not activate", embed=True)

    parent = db.match_nodes(properties={"document_key": "fail"}, limit=1)[0]
    assert parent.properties["active_generation"] == r1.generation
    # Previous passages still present
    still = db.match_nodes(
        properties={
            "managed_by": MANAGED_BY,
            "generation": r1.generation,
            "role": "passage",
        }
    )
    assert still


def test_external_parent_not_deleted(db_ing):
    db, ing = db_ing
    external = db.create_node(labels=["Concept"], properties={"concept_id": "x"})
    r = ing.ingest("body text", parent_id=external.id, document_key="ext", embed=True)
    assert r.owner_document_id == external.id
    ing.delete(external.id, delete_owned_parent=True)
    # External parent remains
    assert db.get_node(external.id) is not None
    # Managed passages gone
    leftovers = db.match_nodes(
        properties={"managed_by": MANAGED_BY, "owner_document_id": external.id}
    )
    assert leftovers == []


def test_search_ignores_foreign_and_stale(db_ing):
    db, ing = db_ing
    ing.ingest("alpha beta gamma document", document_key="mine", embed=True)
    # Foreign chunk with similar embedding-ish text on same index
    foreign = db.create_node(
        labels=["Chunk"],
        properties={
            "text": "alpha beta gamma document",
            "managed_by": "someone.else",
            "embed_role": "passage",
            "generation": 99,
            "owner_document_id": 999,
        },
    )
    db.upsert_embeddings([foreign.id], [foreign.properties["text"]], index="docs_chunks")
    hits = ing.search("alpha beta gamma", k=5)
    assert all(h.node.properties.get("managed_by") == MANAGED_BY for h in hits)
    assert all(h.node.id != foreign.id for h in hits)


def test_pack_overlap_dedupe():
    db = GrafitoDatabase(":memory:")
    full = "AAA " + ("shared middle zone. " * 8) + "ZZZ"
    # Manual two overlapping nodes
    n1 = db.create_node(
        labels=["Chunk"],
        properties={
            "text": full[0:40],
            "char_start": 0,
            "char_end": 40,
            "global_seq": 0,
            "owner_document_id": 1,
            "managed_by": MANAGED_BY,
        },
    )
    n2 = db.create_node(
        labels=["Chunk"],
        properties={
            "text": full[20:60],
            "char_start": 20,
            "char_end": 60,
            "global_seq": 1,
            "owner_document_id": 1,
            "managed_by": MANAGED_BY,
        },
    )
    parent = db.create_node(
        labels=["Document"],
        properties={"text": full, "managed_by": MANAGED_BY, "role": "document"},
    )
    # Fix owner ids to parent
    for n in (n1, n2):
        p = dict(n.properties)
        p["owner_document_id"] = parent.id
        db.replace_node_properties(n.id, p)
        n = db.get_node(n.id)
    n1 = db.get_node(n1.id)
    n2 = db.get_node(n2.id)
    ing = DocumentIngestor(db, embed_index=None)
    naive = n1.properties["text"] + n2.properties["text"]
    packed = ing.pack([n1, n2], deduplicate_overlap=True, include_citations=False)
    assert len(packed.text) < len(naive)
    db.close()


def test_delete_owned_document(db_ing):
    db, ing = db_ing
    r = ing.ingest("bye", document_key="bye", embed=True)
    ing.delete("bye", delete_owned_parent=True)
    assert db.get_node(r.owner_document_id) is None
