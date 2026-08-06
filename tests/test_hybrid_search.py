"""Fused vector + lexical retrieval, and its subgraph form."""

import hashlib
import re

import pytest

from grafito import GrafitoDatabase
from grafito.embedding_functions import EmbeddingFunction
from grafito.exceptions import DatabaseError


class _Embedder(EmbeddingFunction):
    """Bag-of-words embedder: paraphrase matches, unseen tokens do not."""

    def __init__(self, dim: int = 64) -> None:
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
        return "hybrid_test_embedder"

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


DOCS = [
    {"id": "d1", "text": "graph databases store nodes and edges", "public": True},
    {"id": "d2", "text": "knowledge graphs connect entities", "public": True},
    {"id": "d3", "text": "the error code ENOSPC means disk full", "public": False},
    {"id": "d4", "text": "vector search finds nearest neighbours", "public": True},
    {"id": "d5", "text": "embeddings represent meaning numerically", "public": True},
]


@pytest.fixture
def db() -> GrafitoDatabase:
    database = GrafitoDatabase(':memory:')
    database.create_vector_index("default", dim=64, embedding_function=_Embedder())
    database.index_documents(DOCS, label="Doc", configure_fts=True)
    first = database.match_nodes(labels=["Doc"], properties={"id": "d1"})[0]
    second = database.match_nodes(labels=["Doc"], properties={"id": "d2"})[0]
    database.create_relationship(first.id, second.id, "RELATED")
    yield database
    database.close()


def _ids(hits: list[dict]) -> list[str]:
    return [hit["node"].properties["id"] for hit in hits]


# --- hybrid_search ----------------------------------------------------------


def test_returns_node_and_score(db):
    hits = db.hybrid_search("graph databases", k=3)
    assert hits
    assert all(set(hit) == {"node", "score"} for hit in hits)
    assert all(isinstance(hit["score"], float) for hit in hits)


def test_results_are_ordered(db):
    scores = [hit["score"] for hit in db.hybrid_search("graph databases", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_results(db):
    assert len(db.hybrid_search("graph", k=2)) == 2


def test_finds_an_exact_term_the_embedder_cannot_place(db):
    """The lexical half earns its keep on identifiers and codes."""
    assert _ids(db.hybrid_search("ENOSPC", k=1)) == ["d3"]


def test_finds_a_paraphrase_no_keyword_would_match(db):
    """The vector half earns its keep when no term is shared."""
    assert _ids(db.hybrid_search("meaning vectors", k=1)) == ["d5"]


def test_no_duplicate_nodes(db):
    """A document ranked by both sides must appear once, not twice."""
    hits = db.hybrid_search("graph databases store", k=5)
    ids = _ids(hits)
    assert len(ids) == len(set(ids))


def test_labels_restrict_both_sides(db):
    db.create_node(labels=["Other"], properties={"id": "x", "text": "graph databases"})
    hits = db.hybrid_search("graph databases", k=5, labels=["Doc"])
    assert "x" not in _ids(hits)


def test_filter_props_restricts_the_vector_side(db):
    hits = db.hybrid_search("graph databases", k=5, filter_props={"public": True})
    assert "d3" not in _ids(hits)


def test_weights_reach_the_fused_score(db):
    """Not a proof of quality — only that the knob is wired through.

    Asserted on scores rather than order: on a small corpus both sides often
    agree on the ranking even when weighted differently.
    """
    balanced = db.hybrid_search("graph databases", k=3)
    text_led = db.hybrid_search("graph databases", k=3, vector_weight=0.0)
    vector_led = db.hybrid_search("graph databases", k=3, text_weight=0.0)

    assert [h["score"] for h in balanced] != [h["score"] for h in text_led]
    assert [h["score"] for h in balanced] != [h["score"] for h in vector_led]
    # A zeroed side contributes nothing, so the surviving scores are lower.
    assert balanced[0]["score"] > text_led[0]["score"]


def test_candidate_pools_are_configurable(db):
    assert db.hybrid_search("graph", k=2, vector_k=2, text_k=2)


# --- degradation ------------------------------------------------------------


def test_without_a_text_index_the_vector_side_carries_it():
    database = GrafitoDatabase(':memory:')
    database.create_vector_index("default", dim=64, embedding_function=_Embedder())
    database.index_documents(DOCS, label="Doc")  # no configure_fts

    hits = database.hybrid_search("graph databases", k=2)
    assert _ids(hits) == _ids(database.semantic_search("graph databases", k=2))
    database.close()


def test_without_a_vector_index_the_text_side_carries_it():
    database = GrafitoDatabase(':memory:')
    database.create_text_index("node", "Doc", ["text"])
    for row in DOCS:
        database.create_node(labels=["Doc"], properties=row)

    hits = database.hybrid_search("graph databases", k=2)
    assert "d1" in _ids(hits)
    database.close()


def test_with_neither_index_it_returns_nothing():
    database = GrafitoDatabase(':memory:')
    database.create_node(labels=["Doc"], properties={"id": "d1", "text": "graph"})
    assert database.hybrid_search("graph", k=2) == []
    database.close()


def test_an_unrelated_query_still_returns_top_k(db):
    """There is no relevance floor: this is top-k, inherited from both sides.

    The lexical side finds nothing, but vector search always returns its k
    nearest — at score 0.0 here. Filter on the score if you need a cutoff.
    """
    hits = db.hybrid_search("zzzz nonexistent qqqq", k=3)
    assert len(hits) == 3
    assert db.text_search("zzzz nonexistent qqqq", k=3) == []


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_invalid_queries_are_rejected(db, bad):
    with pytest.raises(DatabaseError, match="non-empty query string"):
        db.hybrid_search(bad, k=3)


def test_invalid_k_is_rejected(db):
    with pytest.raises(DatabaseError, match="k must be"):
        db.hybrid_search("graph", k=0)


# --- hybrid_subgraph --------------------------------------------------------


def test_hybrid_subgraph_seeds_from_the_fused_ranking(db):
    sub = db.hybrid_subgraph("graph databases", k=2, expand=0)
    assert {node.properties["id"] for node in sub.nodes} == set(
        _ids(db.hybrid_search("graph databases", k=2))
    )
    assert all(sub.hops[node_id] == 0 for node_id in sub.seed_ids())


def test_hybrid_subgraph_expands_and_keeps_provenance(db):
    sub = db.hybrid_subgraph("ENOSPC", k=1, expand=0)
    assert {node.properties["id"] for node in sub.nodes} == {"d3"}
    assert set(sub.scores) == set(sub.seed_ids())


def test_hybrid_subgraph_returns_edges_among_seeds(db):
    sub = db.hybrid_subgraph("graph databases entities", k=5, expand=0)
    assert any(rel.type == "RELATED" for rel in sub.relationships)


def test_hybrid_subgraph_forwards_search_options(db):
    sub = db.hybrid_subgraph(
        "graph databases", k=5, expand=0, filter_props={"public": True}
    )
    assert "d3" not in {node.properties["id"] for node in sub.nodes}


def test_hybrid_subgraph_separates_retrieval_from_expansion(db):
    other = db.create_node(labels=["Other"], properties={"id": "x"})
    first = db.match_nodes(labels=["Doc"], properties={"id": "d1"})[0]
    db.create_relationship(first.id, other.id, "RELATED")

    sub = db.hybrid_subgraph(
        "graph databases", k=1, expand=1, search_labels=["Doc"], labels=["Doc"]
    )
    assert "x" not in {node.properties["id"] for node in sub.nodes}


def test_hybrid_subgraph_feeds_centrality(db):
    sub = db.hybrid_subgraph("graph databases", k=5, expand=1)
    assert db.centrality("pagerank", graph=sub.to_networkx(), limit=1)
