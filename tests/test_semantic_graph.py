"""Materialising the vector index as navigable relationships."""

import hashlib
import re

import pytest

from grafito import GrafitoDatabase, SemanticGraphReport
from grafito.embedding_functions import EmbeddingFunction
from grafito.exceptions import DatabaseError


class _Embedder(EmbeddingFunction):
    """Deterministic offline embedder (md5-hashed bag of words)."""

    def __init__(self, dim: int = 32) -> None:
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
        return "semantic_graph_test_embedder"

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
    {"id": "d1", "text": "graph databases store nodes edges"},
    {"id": "d2", "text": "graph databases query nodes"},
    {"id": "d3", "text": "vector search nearest neighbors"},
    {"id": "d4", "text": "vector search embeddings neighbors"},
    {"id": "d5", "text": "cooking pasta recipe italian"},
]


@pytest.fixture
def db() -> GrafitoDatabase:
    database = GrafitoDatabase(':memory:')
    database.create_vector_index(
        "default",
        dim=32,
        embedding_function=_Embedder(),
        options={"store_embeddings": True},
    )
    database.index_documents(DOCS, label="Doc")
    yield database
    database.close()


def _generated_count(db: GrafitoDatabase, rel_type: str = "SEMANTIC_SIMILAR") -> int:
    rows = db.execute(
        f"MATCH ()-[r:{rel_type}]->() WHERE r.generated_by IS NOT NULL RETURN count(r) AS c"
    )
    return rows[0]["c"]


# --- building --------------------------------------------------------------


def test_creates_edges_between_similar_nodes(db):
    report = db.create_semantic_graph(k=2, min_score=0.1)
    assert isinstance(report, SemanticGraphReport)
    assert report.edges_created > 0
    assert report.nodes_processed == 5
    assert _generated_count(db) == report.edges_created


def test_edges_carry_provenance(db):
    db.create_semantic_graph(k=2, min_score=0.1)
    rows = db.execute("""
        MATCH ()-[r:SEMANTIC_SIMILAR]->()
        RETURN r.score AS score, r.index AS index, r.generated_by AS by,
               r.generated_at AS at
        LIMIT 1
    """)
    edge = rows[0]
    assert 0.0 <= edge["score"] <= 1.0
    assert edge["index"] == "default"
    assert edge["by"] == "create_semantic_graph"
    assert edge["at"]


def test_neighbours_are_the_semantically_close_ones(db):
    """d3 and d4 are near-duplicates; the recipe is not near either."""
    db.create_semantic_graph(k=1, min_score=0.1)
    rows = db.execute("""
        MATCH (a:Doc {id: 'd3'})-[:SEMANTIC_SIMILAR]-(b:Doc)
        RETURN b.id AS id
    """)
    assert "d4" in {row["id"] for row in rows}


def test_min_score_controls_the_edge_count(db):
    permissive = db.create_semantic_graph(k=4, min_score=0.0)
    strict = db.create_semantic_graph(k=4, min_score=0.5)
    assert strict.edges_created < permissive.edges_created


def test_k_controls_the_edge_count(db):
    small = db.create_semantic_graph(k=1, min_score=0.0)
    large = db.create_semantic_graph(k=4, min_score=0.0)
    assert large.edges_created > small.edges_created


def test_a_node_is_never_linked_to_itself(db):
    db.create_semantic_graph(k=5, min_score=0.0)
    rows = db.execute(
        "MATCH (a)-[r:SEMANTIC_SIMILAR]->(b) WHERE id(a) = id(b) RETURN count(r) AS c"
    )
    assert rows[0]["c"] == 0


def test_undirected_emits_one_edge_per_pair(db):
    undirected = db.create_semantic_graph(k=2, min_score=0.1, undirected=True)
    directed = db.create_semantic_graph(k=2, min_score=0.1, undirected=False)
    assert directed.edges_created > undirected.edges_created


def test_custom_relationship_type(db):
    db.create_semantic_graph(k=2, min_score=0.1, rel_type="NEAR")
    assert _generated_count(db, "NEAR") > 0
    assert _generated_count(db, "SEMANTIC_SIMILAR") == 0


def test_labels_restrict_which_nodes_are_linked(db):
    other = db.create_node(labels=["Other"], properties={"id": "x"})
    db.upsert_embedding(other.id, _Embedder()(["graph databases store nodes edges"])[0])
    db.create_semantic_graph(k=5, min_score=0.0, labels=["Doc"])
    rows = db.execute(
        "MATCH (n:Other)-[r:SEMANTIC_SIMILAR]-() RETURN count(r) AS c"
    )
    assert rows[0]["c"] == 0


def test_max_edges_truncates_and_reports_it(db):
    report = db.create_semantic_graph(k=4, min_score=0.0, max_edges=3)
    assert report.truncated is True
    assert report.edges_created == 3


def test_nodes_without_embeddings_are_ignored(db):
    db.create_node(labels=["Doc"], properties={"id": "no-vector"})
    report = db.create_semantic_graph(k=2, min_score=0.1)
    assert report.nodes_processed == 5


# --- rebuilding and incremental updates ------------------------------------


def test_rebuild_is_idempotent(db):
    first = db.create_semantic_graph(k=2, min_score=0.1)
    second = db.create_semantic_graph(k=2, min_score=0.1)
    assert second.edges_removed == first.edges_created
    assert second.edges_created == first.edges_created
    assert _generated_count(db) == first.edges_created


def test_replace_spares_hand_made_edges_of_the_same_type(db):
    """A manual SEMANTIC_SIMILAR edge is not this method's to delete."""
    a = db.match_nodes(labels=["Doc"], properties={"id": "d1"})[0]
    b = db.match_nodes(labels=["Doc"], properties={"id": "d5"})[0]
    db.create_relationship(a.id, b.id, "SEMANTIC_SIMILAR", {"manual": True})

    db.create_semantic_graph(k=2, min_score=0.1)
    db.create_semantic_graph(k=2, min_score=0.1)

    rows = db.execute(
        "MATCH ()-[r:SEMANTIC_SIMILAR]->() WHERE r.manual = true RETURN count(r) AS c"
    )
    assert rows[0]["c"] == 1


def test_replace_false_appends(db):
    first = db.create_semantic_graph(k=2, min_score=0.1)
    db.create_semantic_graph(k=2, min_score=0.1, replace=False)
    assert _generated_count(db) == first.edges_created * 2


def test_approximate_skips_already_linked_nodes(db):
    db.create_semantic_graph(k=2, min_score=0.1)
    report = db.create_semantic_graph(k=2, min_score=0.1, approximate=True, replace=False)
    assert report.nodes_skipped == 5
    assert report.nodes_processed == 0
    assert report.edges_created == 0


def test_refresh_links_only_the_new_nodes(db):
    db.create_semantic_graph(k=2, min_score=0.1)
    before = _generated_count(db)

    db.index_documents([{"id": "d6", "text": "cooking pasta italian food"}], label="Doc")
    report = db.refresh_semantic_graph(k=2, min_score=0.1)

    assert report.nodes_processed == 1
    assert report.nodes_skipped == 5
    assert report.edges_created > 0
    assert _generated_count(db) == before + report.edges_created


# --- dropping --------------------------------------------------------------


def test_drop_removes_generated_edges_only(db):
    a = db.match_nodes(labels=["Doc"], properties={"id": "d1"})[0]
    b = db.match_nodes(labels=["Doc"], properties={"id": "d5"})[0]
    db.create_relationship(a.id, b.id, "SEMANTIC_SIMILAR", {"manual": True})
    generated = db.create_semantic_graph(k=2, min_score=0.1).edges_created

    assert db.drop_semantic_graph() == generated
    rows = db.execute("MATCH ()-[r:SEMANTIC_SIMILAR]->() RETURN count(r) AS c")
    assert rows[0]["c"] == 1


def test_drop_on_a_clean_database_returns_zero(db):
    assert db.drop_semantic_graph() == 0


# --- validation ------------------------------------------------------------


def test_invalid_arguments_are_rejected(db):
    with pytest.raises(DatabaseError, match="k must be"):
        db.create_semantic_graph(k=0)
    with pytest.raises(DatabaseError, match="max_edges must be"):
        db.create_semantic_graph(max_edges=0)
    with pytest.raises(DatabaseError, match="rel_type must be"):
        db.create_semantic_graph(rel_type="")


def test_unknown_index_is_rejected(db):
    with pytest.raises(DatabaseError, match="does not exist"):
        db.create_semantic_graph(index="nope")


# --- composition -----------------------------------------------------------


def test_semantic_graph_is_traversable_from_cypher(db):
    """The point of materialising: multi-hop patterns over similarity."""
    db.create_semantic_graph(k=2, min_score=0.1)
    rows = db.execute("""
        MATCH p=(a:Doc {id: 'd1'})-[:SEMANTIC_SIMILAR*1..2]-(b:Doc)
        RETURN DISTINCT b.id AS id
    """)
    assert len(rows) > 1


def test_communities_over_the_semantic_graph(db):
    """The one thing the ANN index alone cannot do: cluster by similarity."""
    db.create_semantic_graph(k=2, min_score=0.1)
    communities = db.communities(
        "louvain", rel_types=["SEMANTIC_SIMILAR"], weight_property="score", seed=42
    )
    assert communities
    covered = {node.properties["id"] for c in communities for node in c.nodes}
    assert covered == {"d1", "d2", "d3", "d4", "d5"}


def test_generated_edges_can_be_excluded_from_analysis(db):
    """The escape hatch the warning in the docs depends on."""
    a = db.match_nodes(labels=["Doc"], properties={"id": "d1"})[0]
    b = db.match_nodes(labels=["Doc"], properties={"id": "d2"})[0]
    db.create_relationship(a.id, b.id, "CITES")
    db.create_semantic_graph(k=4, min_score=0.0)

    polluted = db.to_analysis_graph()
    clean = db.to_analysis_graph(exclude_rel_types=["SEMANTIC_SIMILAR"])
    assert clean.number_of_edges() == 1
    assert polluted.number_of_edges() > clean.number_of_edges()


def test_report_str_is_readable(db):
    report = db.create_semantic_graph(k=2, min_score=0.1)
    assert "edges" in str(report)
    assert "nodes" in str(report)
