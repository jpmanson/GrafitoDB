"""Tests for the high-level OKFBundle façade."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.embedding_functions import EmbeddingFunction
from grafito.okf import Concept, Hit, OKFBundle

pytest.importorskip("yaml")

KB = Path("examples") / "okf_knowledge_base"


class _Embedder(EmbeddingFunction):
    """Deterministic offline embedder (md5-hashed bag of words)."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        out = []
        for text in input:
            vec = [0.0] * self._dim
            for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
                vec[int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little") % self._dim] += 1
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    @staticmethod
    def name() -> str:
        return "bundle_test_embedder"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "ip"]

    @staticmethod
    def build_from_config(config: dict) -> "_Embedder":
        return _Embedder()

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def kb() -> OKFBundle:
    bundle = OKFBundle.load(str(KB), embed=_Embedder())
    yield bundle
    bundle.db.close()


# --- lifecycle / basics -----------------------------------------------------


def test_load_and_summary(kb):
    assert kb.summary["nodes"] == 7
    assert len(kb) == 7
    assert isinstance(kb.db, GrafitoDatabase)


def test_load_into_existing_db():
    db = GrafitoDatabase(":memory:")
    bundle = OKFBundle.load(str(KB), db=db, configure_fts=False)
    assert bundle.db is db
    db.close()


def test_iteration_excludes_non_concepts(kb):
    ids = {c.id for c in kb}
    assert "decisions/0001-use-sqlite" in ids
    # Auto Reference nodes and stubs are not concepts.
    assert all(isinstance(c, Concept) for c in kb)
    assert all(not c.node.properties.get("okf_auto") for c in kb)


# --- concept access ---------------------------------------------------------


def test_concept_lookup_and_accessors(kb):
    c = kb.concept("decisions/0001-use-sqlite")
    assert c is not None
    assert c.type == "ADR"
    assert c.title == "Use SQLite as the storage engine"
    assert "storage" in c.tags
    assert "# Context" in c.body


def test_getitem_and_missing(kb):
    assert kb["glossary/cypher"].title == "Cypher"
    assert kb.concept("does/not/exist") is None
    with pytest.raises(KeyError):
        kb["does/not/exist"]


def test_concepts_filtered_by_type_and_layer(kb):
    adrs = kb.concepts(type="ADR")
    assert len(adrs) == 3
    decisions = kb.concepts(layer="decisions")
    assert {c.id for c in adrs} == {c.id for c in decisions}
    terms = kb.concepts(tag="glossary")
    assert all(c.type == "Term" for c in terms)


# --- topology ---------------------------------------------------------------


def test_layers(kb):
    assert kb.layers() == {"decisions": 3, "glossary": 3, "runbooks": 1}


def test_links_and_cites(kb):
    c = kb.concept("decisions/0003-vector-search")
    linked = {x.id for x in c.links()}
    assert "glossary/semantic-search" in linked
    assert "decisions/0001-use-sqlite" in linked

    cited_urls = {entry.get("url") for entry in c.cites()}
    assert "https://arxiv.org/abs/1603.09320" in cited_urls


def test_linked_by(kb):
    # The SQLite ADR is referenced by the other two decisions and the runbook.
    sqlite = kb.concept("decisions/0001-use-sqlite")
    back = {x.id for x in sqlite.linked_by()}
    assert "decisions/0003-vector-search" in back


def test_references(kb):
    refs = kb.references()
    assert refs
    assert all(r["url"].startswith("http") for r in refs)


def test_index_root_lists_subdirs(kb):
    idx = kb.index()
    assert idx["layer"] is None
    assert idx["subdirs"] == {"decisions": 3, "glossary": 3, "runbooks": 1}
    assert idx["concepts"] == []  # no root-level concepts in this bundle


def test_index_layer_lists_concepts_with_descriptions(kb):
    idx = kb.index("decisions")
    assert idx["subdirs"] == {}
    ids = [e["id"] for e in idx["concepts"]]
    assert ids == [
        "decisions/0001-use-sqlite",
        "decisions/0002-cypher-subset",
        "decisions/0003-vector-search",
    ]
    first = idx["concepts"][0]
    assert first["title"] == "Use SQLite as the storage engine"
    assert first["description"]
    assert first["type"] == "ADR"
    # The listing carries no bodies (progressive disclosure).
    assert "body" not in first


def test_index_nested_directories():
    bundle = OKFBundle.load(str(Path("tests") / "res" / "okf_gcp_ga4"), configure_fts=False)
    try:
        assert set(bundle.index()["subdirs"]) == {"datasets", "references", "tables"}
        assert set(bundle.index("references")["subdirs"]) == {"joins", "metrics"}
        assert len(bundle.index("references/metrics")["concepts"]) == 5
    finally:
        bundle.db.close()


# --- search -----------------------------------------------------------------


def test_search_semantic_returns_hits(kb):
    hits = kb.search("how do I make a query run faster", mode="semantic", k=3)
    assert hits and all(isinstance(h, Hit) for h in hits)
    assert all(h.via == "semantic" for h in hits)
    assert "Triaging a slow graph query" in {h.concept.title for h in hits}


def test_search_layer_filter(kb):
    hits = kb.search("make query faster", layer="decisions", k=5)
    assert hits
    assert all(h.concept.id.startswith("decisions/") for h in hits)


def test_search_type_filter(kb):
    hits = kb.search("graph", type="Term", mode="semantic", k=5)
    assert all(h.concept.type == "Term" for h in hits)


def test_search_hybrid(kb):
    hits = kb.search("vector similarity search", mode="hybrid", k=3)
    assert hits and all(h.via == "hybrid" for h in hits)


def test_search_auto_uses_semantic_when_embedded(kb):
    # The fixture imported with an embedder, so auto resolves to semantic.
    hits = kb.search("performance", k=2)
    assert hits and hits[0].via == "semantic"


# --- metadata / escape hatch ------------------------------------------------


def test_okf_version_captured(tmp_path):
    (tmp_path / "index.md").write_text('---\nokf_version: "0.1"\n---\n# Subdirectories\n')
    (tmp_path / "a.md").write_text("---\ntype: Note\ntitle: A\n---\nbody\n")
    bundle = OKFBundle.load(str(tmp_path), configure_fts=False)
    assert bundle.okf_version == "0.1"
    bundle.db.close()


def test_execute_escape_hatch(kb):
    rows = kb.execute("MATCH (n) WHERE n.concept_id = $c RETURN n.title AS t", c="glossary/cypher")
    assert rows[0]["t"] == "Cypher"


def test_save_round_trip(kb, tmp_path):
    summary = kb.save(str(tmp_path))
    assert summary["concepts"] == 7
    reloaded = OKFBundle.load(str(tmp_path), configure_fts=False)
    assert len(reloaded) == 7
    assert reloaded.concept("glossary/cypher").title == "Cypher"
    reloaded.db.close()


def test_save_without_path_uses_source(tmp_path):
    # Build a tiny bundle on disk, load it, mutate via the graph, save back.
    (tmp_path / "a.md").write_text("---\ntype: Note\ntitle: A\n---\nbody\n")
    bundle = OKFBundle.load(str(tmp_path), configure_fts=False)
    summary = bundle.save()  # defaults to the load path
    assert summary["concepts"] == 1
    bundle.db.close()
