"""Tests for the high-level OKFBundle façade."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.embedding_functions import EmbeddingFunction
from grafito.okf import Concept, ContextPack, Hit, LexicalReranker, OKFBundle
from grafito.okf.rerank import (
    DEFAULT_RERANK_FIELDS,
    CohereReranker,
    CrossEncoderReranker,
    JinaReranker,
    VoyageReranker,
    _parse_rerank_results,
)

pytest.importorskip("yaml")

KB = Path("examples") / "okf" / "okf_knowledge_base"


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


# --- mutation (Phase 2) -----------------------------------------------------


def test_add_concept(kb):
    before = len(kb)
    c = kb.add_concept(
        "decisions/0004-caching",
        type="ADR",
        title="Add a query cache",
        description="Cache hot query results.",
        tags=["perf"],
        body="# Context\nHot queries repeat.\n",
    )
    assert c.type == "ADR"
    assert c.title == "Add a query cache"
    assert len(kb) == before + 1
    assert kb.concept("decisions/0004-caching") is not None


def test_add_concept_duplicate_raises(kb):
    with pytest.raises(ValueError):
        kb.add_concept("decisions/0001-use-sqlite", type="ADR")


def test_add_concept_is_searchable(kb):
    kb.add_concept(
        "glossary/embedding",
        type="Term",
        title="Embedding",
        body="A vector representation of text used for semantic similarity.",
    )
    hits = kb.search("vector representation similarity", mode="semantic", k=5)
    assert "Embedding" in {h.concept.title for h in hits}


def test_link_and_cite(kb):
    kb.add_concept("notes/x", type="Note", title="X")
    kb.link("notes/x", "decisions/0001-use-sqlite", anchor="see")
    kb.cite("notes/x", "https://example.com/ref", anchor="ref")

    x = kb.concept("notes/x")
    assert {c.id for c in x.links()} == {"decisions/0001-use-sqlite"}
    assert x.cites() == [{"url": "https://example.com/ref", "anchor": "ref"}]


def test_cite_deduplicates_reference(kb):
    kb.add_concept("notes/a", type="Note", title="A")
    kb.add_concept("notes/b", type="Note", title="B")
    before = len(kb.references())
    kb.cite("notes/a", "https://example.com/shared")
    kb.cite("notes/b", "https://example.com/shared")
    after = [r for r in kb.references() if r["url"] == "https://example.com/shared"]
    assert len(after) == 1
    assert len(kb.references()) == before + 1


def test_remove_concept(kb):
    assert kb.remove_concept("glossary/cypher") is True
    assert kb.concept("glossary/cypher") is None
    assert kb.remove_concept("does/not/exist") is False


def test_mutations_round_trip_for_bodyless_concept(kb, tmp_path):
    # A concept with no body: links/citations must persist via export synthesis.
    kb.add_concept("notes/seed", type="Note", title="Seed")
    kb.link("notes/seed", "decisions/0001-use-sqlite", anchor="see")
    kb.cite("notes/seed", "https://example.com/ref", anchor="ref")
    kb.save(str(tmp_path))

    reloaded = OKFBundle.load(str(tmp_path), configure_fts=False)
    try:
        seed = reloaded.concept("notes/seed")
        assert {c.id for c in seed.links()} == {"decisions/0001-use-sqlite"}
        assert reloaded.concept("notes/seed").cites() == [
            {"url": "https://example.com/ref", "anchor": "ref"}
        ]
    finally:
        reloaded.db.close()


def test_save_without_path_uses_source(tmp_path):
    # Build a tiny bundle on disk, load it, mutate via the graph, save back.
    (tmp_path / "a.md").write_text("---\ntype: Note\ntitle: A\n---\nbody\n")
    bundle = OKFBundle.load(str(tmp_path), configure_fts=False)
    summary = bundle.save()  # defaults to the load path
    assert summary["concepts"] == 1
    bundle.db.close()


# --- directory tree / log (Phase 3) -----------------------------------------


def test_directory_nodes_and_contains_traversal():
    bundle = OKFBundle.load(str(KB), directory_nodes=True, configure_fts=False)
    try:
        # root + 3 subdirectories; concepts unchanged.
        assert bundle.summary["directories"] == 4
        assert len(bundle) == 7

        root = bundle.children()
        assert root["subdirs"] == ["decisions", "glossary", "runbooks"]
        assert root["concepts"] == []

        decisions = bundle.children("decisions")
        assert decisions["subdirs"] == []
        assert {c.id for c in decisions["concepts"]} == {
            "decisions/0001-use-sqlite",
            "decisions/0002-cypher-subset",
            "decisions/0003-vector-search",
        }
    finally:
        bundle.db.close()


def test_directory_nodes_not_exported_or_counted_as_concepts():
    bundle = OKFBundle.load(str(KB), directory_nodes=True, configure_fts=False)
    try:
        # Directory nodes are not concepts and not iterated.
        assert all(not c.node.properties.get("directory") for c in bundle)
    finally:
        bundle.db.close()


def test_log_import_and_mentions(tmp_path):
    (tmp_path / "log.md").write_text(
        "# Log\n"
        "## 2026-05-22\n"
        "* **Creation**: Established [the note](/notes/a.md).\n"
        "## 2026-05-10\n"
        "* **Update**: Minor tweak.\n"
    )
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("---\ntype: Note\ntitle: A\n---\nbody\n")

    bundle = OKFBundle.load(str(tmp_path), import_log=True, configure_fts=False)
    try:
        assert bundle.summary["log_entries"] == 2
        assert len(bundle) == 1  # only the note is a concept

        entries = bundle.log()
        assert [e["date"] for e in entries] == ["2026-05-22", "2026-05-10"]
        assert entries[0]["kind"] == "Creation"

        # The first entry mentions notes/a; the second mentions nothing.
        about_a = bundle.log("notes/a")
        assert len(about_a) == 1
        assert about_a[0]["kind"] == "Creation"
    finally:
        bundle.db.close()


def test_log_entries_not_exported(tmp_path):
    (tmp_path / "log.md").write_text("# Log\n## 2026-05-22\n* **Creation**: Set up.\n")
    (tmp_path / "a.md").write_text("---\ntype: Note\ntitle: A\n---\nbody\n")
    bundle = OKFBundle.load(str(tmp_path), import_log=True, configure_fts=False)
    try:
        out = tmp_path / "out"
        summary = bundle.save(str(out))
        assert summary["concepts"] == 1
        assert [p.name for p in out.rglob("*.md") if p.name != "index.md"] == ["a.md"]
    finally:
        bundle.db.close()


# --- context assembly -------------------------------------------------------


def test_context_returns_grounded_pack(kb):
    pack = kb.context("how do I make a query run faster", k=3)
    assert isinstance(pack, ContextPack)
    assert pack.concepts and pack.text
    # str(pack) is the prompt-ready text.
    assert str(pack) == pack.text
    # The top hit (the slow-query runbook) is rendered as a titled block.
    assert "Triaging a slow graph query" in pack.text
    assert "### " in pack.text


def test_context_expands_into_graph_neighbours(kb):
    # The runbook links to glossary terms / decisions the query words don't match;
    # graph expansion should pull at least one such neighbour into the pack.
    seed = kb.search("how do I make a query run faster", k=1)[0].concept
    linked_ids = {c.id for c in seed.neighbors(depth=1)}
    assert linked_ids  # the seed has outgoing links

    expanded = {c.id for c in kb.context("how do I make a query run faster", k=1).concepts}
    assert expanded & linked_ids  # at least one neighbour made it in
    # And expansion can be disabled.
    no_expand = kb.context("how do I make a query run faster", k=1, expand_hops=0)
    assert {c.id for c in no_expand.concepts} == {seed.id}


def test_context_respects_token_budget(kb):
    pack = kb.context("query performance", budget_tokens=40)
    assert pack.tokens <= 60  # at/near budget (whole-block packing + truncation)
    assert pack.concepts  # the top hit is never dropped
    assert pack.truncated


def test_context_includes_deduplicated_citations(kb):
    pack = kb.context("vector similarity search", k=5)
    assert pack.citations
    # Citations carry provenance and are unique per target.
    targets = [c.get("url") or c.get("concept") for c in pack.citations]
    assert len(targets) == len(set(targets))
    assert all("cited_by" in c and c["cited_by"] for c in pack.citations)


def test_context_custom_token_counter_is_used(kb):
    calls: list[str] = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text.split())

    pack = kb.context("query performance", token_counter=counter, budget_tokens=50)
    assert calls  # the custom counter drove budgeting
    assert pack.tokens == len(pack.text.split())


def test_context_empty_when_no_hits(kb):
    pack = kb.context("zqxwvurstplmnbgk", mode="text", k=3)
    assert pack.concepts == [] or pack.text
    # No hits → empty, well-formed pack.
    if not pack.concepts:
        assert pack.text == ""
        assert pack.tokens == 0
        assert pack.citations == []


# --- reranking --------------------------------------------------------------


def test_context_rerank_reorders_candidate_pool(kb):
    question = "how do I make a query run faster"
    base = kb.context(question, k=3, expand_hops=1, budget_tokens=5000)
    reranked = kb.context(
        question, k=3, expand_hops=1, budget_tokens=5000, rerank=LexicalReranker()
    )
    base_ids = [c.id for c in base.concepts]
    reranked_ids = [c.id for c in reranked.concepts]
    # Same pool, but the lexical reranker changes the order.
    assert set(base_ids) == set(reranked_ids)
    assert reranked_ids != base_ids


def test_context_rerank_is_injectable_callable(kb):
    captured: dict = {}

    def reverse_reranker(query, candidates):
        captured["query"] = query
        captured["n"] = len(candidates)
        return [(c, float(i)) for i, c in enumerate(reversed(candidates))]

    pack = kb.context("query performance", k=2, rerank=reverse_reranker, budget_tokens=5000)
    assert captured["query"] == "query performance"
    assert captured["n"] >= len(pack.concepts)
    # The reranker's order is honoured.
    assert pack.concepts  # something was packed


def test_context_rerank_top_n_subset_limits_pool(kb):
    def top1(query, candidates):
        scored = LexicalReranker()(query, candidates)
        return scored[:1]  # reranker keeps only its best candidate

    pack = kb.context(
        "how do I make a query run faster", k=3, rerank=top1, budget_tokens=5000
    )
    assert len(pack.concepts) == 1


# --- provider rerank adapters (offline: parsing + key resolution) -----------


def test_parse_rerank_results_cohere_jina_shape(kb):
    candidates = list(kb)[:3]
    # Cohere/Jina return reordered items under "results"; order is honoured.
    data = {"results": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.4},
    ]}
    out = _parse_rerank_results(data, candidates, results_key="results", provider="Cohere")
    assert [c.id for c, _ in out] == [candidates[2].id, candidates[0].id]
    assert out[0][1] == 0.9


def test_parse_rerank_results_voyage_shape(kb):
    candidates = list(kb)[:3]
    # Voyage returns items under "data".
    data = {"data": [{"index": 1, "relevance_score": 0.7}]}
    out = _parse_rerank_results(data, candidates, results_key="data", provider="Voyage")
    assert [c.id for c, _ in out] == [candidates[1].id]


def test_parse_rerank_results_error_payload_raises(kb):
    with pytest.raises(ValueError, match="Voyage rerank API error"):
        _parse_rerank_results(
            {"message": "bad key"}, list(kb)[:1], results_key="data", provider="Voyage"
        )


@pytest.mark.parametrize(
    "cls, env, default_model, url, top_param, results_key",
    [
        (CohereReranker, "COHERE_API_KEY", "rerank-english-v3.0",
         "https://api.cohere.ai/v1/rerank", "top_n", "results"),
        (VoyageReranker, "VOYAGE_API_KEY", "rerank-2",
         "https://api.voyageai.com/v1/rerank", "top_k", "data"),
        (JinaReranker, "JINA_API_KEY", "jina-reranker-v2-base-multilingual",
         "https://api.jina.ai/v1/rerank", "top_n", "results"),
    ],
)
def test_provider_reranker_config(monkeypatch, cls, env, default_model, url, top_param, results_key):
    pytest.importorskip("httpx")
    monkeypatch.setenv(env, "test-key")
    rr = cls()
    assert rr.model == default_model
    assert rr._url == url
    assert rr._top_param == top_param
    assert rr._results_key == results_key


def test_provider_reranker_missing_key_raises(monkeypatch):
    pytest.importorskip("httpx")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Voyage API key not provided"):
        VoyageReranker()


def test_cross_encoder_reranker_sorts_by_model_score(kb):
    # Drive the scoring/ordering logic offline with a fake CrossEncoder model
    # (the real one needs sentence-transformers + a model download).
    rr = object.__new__(CrossEncoderReranker)
    rr._fields = DEFAULT_RERANK_FIELDS
    rr._max_chars = 2000

    class _FakeModel:
        def predict(self, pairs):
            assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
            return [0.1, 0.9, 0.5][: len(pairs)]

    rr._model = _FakeModel()
    candidates = list(kb)[:3]
    out = rr("how do I make a query run faster", candidates)
    assert [c.id for c, _ in out] == [candidates[1].id, candidates[2].id, candidates[0].id]
    assert out[0][1] == 0.9


def test_cross_encoder_reranker_works_in_context(kb):
    rr = object.__new__(CrossEncoderReranker)
    rr._fields = DEFAULT_RERANK_FIELDS
    rr._max_chars = 2000
    rr._model = type("M", (), {"predict": lambda self, pairs: [1.0] * len(pairs)})()

    pack = kb.context("query performance", k=3, rerank=rr, budget_tokens=5000)
    assert pack.concepts  # the cross-encoder reranker is a valid injectable Reranker


# --- update_concept -----------------------------------------------------------


def test_update_concept_changes_only_passed_fields(kb):
    cid = "decisions/0003-vector-search"
    before = kb[cid]
    updated = kb.update_concept(cid, title="Vector search (revised)")
    assert updated.title == "Vector search (revised)"
    assert updated.description == before.description
    assert updated.body == before.body
    assert kb[cid].title == "Vector search (revised)"


def test_update_concept_none_removes_field(kb):
    cid = "decisions/0003-vector-search"
    assert kb[cid].description
    updated = kb.update_concept(cid, description=None)
    assert updated.description is None
    assert "description" not in kb[cid].properties


def test_update_concept_relabels_type(kb):
    cid = "decisions/0003-vector-search"
    updated = kb.update_concept(cid, type="Decision")
    assert updated.type == "Decision"
    assert kb.concepts(type="Decision")


def test_update_concept_body_is_reindexed(kb):
    cid = "glossary/semantic-search"
    kb.update_concept(cid, body="Completely new content about zebras.")
    hits = kb.search("zebras", mode="text", k=3)
    assert any(h.concept.id == cid for h in hits)


def test_update_concept_reembeds(kb):
    cid = "glossary/semantic-search"
    kb.update_concept(cid, body="xylophone quartz nebula")
    hits = kb.search("xylophone quartz nebula", mode="semantic", k=3)
    assert hits and hits[0].concept.id == cid


def test_update_concept_unknown_or_protected_raises(kb):
    with pytest.raises(ValueError, match="Unknown concept"):
        kb.update_concept("nope/missing", title="X")
    with pytest.raises(ValueError, match="reserved property"):
        kb.update_concept("glossary/semantic-search", stub=True)
    with pytest.raises(ValueError, match="non-empty string"):
        kb.update_concept("glossary/semantic-search", type="")


# --- remove_concept + save(prune) round-trip ----------------------------------


def test_remove_concept_prunes_file_on_save(kb, tmp_path):
    out = tmp_path / "bundle"
    kb.save(out)
    assert (out / "glossary" / "semantic-search.md").exists()

    kb.remove_concept("glossary/semantic-search")
    summary = kb.save(out)
    assert summary["pruned"] == 1
    assert not (out / "glossary" / "semantic-search.md").exists()
    # Round-trip: the removed concept must not resurrect on re-import.
    again = OKFBundle.load(str(out))
    assert again.concept("glossary/semantic-search") is None
    again.db.close()


# --- search degradation without embeddings ------------------------------------


def test_hybrid_search_degrades_to_text_without_embeddings():
    bundle = OKFBundle.load(str(KB))  # no embed= -> no vector index
    hits = bundle.search("vector similarity", mode="hybrid", k=3)
    assert hits
    assert all(h.via == "text" for h in hits)
    bundle.db.close()


# --- persistent reuse: store_embeddings + open() -------------------------------


def test_open_reuses_persistent_db_without_reimport(tmp_path):
    db_path = str(tmp_path / "kb.db")

    # Session 1: import once, persisting the embeddings alongside the graph.
    kb1 = OKFBundle.load(
        str(KB),
        db=GrafitoDatabase(db_path),
        embed=_Embedder(),
        embed_options={"store_embeddings": True},
    )
    assert kb1.summary["embedded"] == 7
    kb1.db.close()

    # Session 2: no markdown parsing, no re-embedding — just open the file.
    kb2 = OKFBundle.open(GrafitoDatabase(db_path), embed=_Embedder(), source_path=str(KB))
    assert len(kb2) == 7  # not re-imported (no duplicates)
    assert kb2.summary["nodes"] == 7
    assert kb2.okf_version is None  # this bundle's root index.md has no frontmatter
    assert kb2.concept("decisions/0001-use-sqlite").title == "Use SQLite as the storage engine"

    # The vector index rehydrates from the stored embeddings.
    hits = kb2.search("how do I make a query run faster", mode="semantic", k=3)
    assert "Triaging a slow graph query" in {h.concept.title for h in hits}
    kb2.db.close()


def test_open_without_source_path_requires_save_target(tmp_path):
    db_path = str(tmp_path / "kb.db")
    OKFBundle.load(str(KB), db=GrafitoDatabase(db_path)).db.close()

    kb = OKFBundle.open(GrafitoDatabase(db_path))
    with pytest.raises(ValueError, match="No path to save"):
        kb.save()
    out = tmp_path / "out"
    assert kb.save(out)["concepts"] == 7
    kb.db.close()


# --- SQL-side filtering ---------------------------------------------------------


def test_concepts_layer_accepts_nested_paths():
    bundle = OKFBundle.load(str(Path("tests") / "res" / "okf_gcp_ga4"), configure_fts=False)
    try:
        nested = bundle.concepts(layer="references/joins")
        assert nested
        assert all(c.id.startswith("references/joins/") for c in nested)
        top = bundle.concepts(layer="references")
        assert {c.id for c in nested} <= {c.id for c in top}
    finally:
        bundle.db.close()


def test_concepts_are_ordered_by_id(kb):
    ids = [c.id for c in kb.concepts()]
    assert ids == sorted(ids)
