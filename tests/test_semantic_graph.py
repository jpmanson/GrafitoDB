"""Materialising the vector index as navigable relationships."""

import hashlib
import re
import sqlite3

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
    """The cap is enforced between nodes, so it overshoots by at most k-1."""
    report = db.create_semantic_graph(k=4, min_score=0.0, max_edges=3)
    assert report.truncated is True
    assert 3 <= report.edges_created < 3 + 4
    assert report.nodes_processed < 5


def test_capped_refreshes_eventually_build_the_whole_graph(db):
    """Every node the cap lets through is complete, so progress never stalls."""
    db.create_semantic_graph(k=2, min_score=0.0, max_edges=2)
    for _ in range(10):
        db.refresh_semantic_graph(k=2, min_score=0.0, max_edges=2)
    capped = _generated_count(db)

    db.create_semantic_graph(k=2, min_score=0.0)
    assert _generated_count(db) == capped


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


def test_replace_false_adds_without_duplicating(db):
    """Appending must not stack a second copy of edges that already exist."""
    first = db.create_semantic_graph(k=2, min_score=0.1)
    db.create_semantic_graph(k=2, min_score=0.1, replace=False)
    assert _generated_count(db) == first.edges_created


def test_replace_false_adds_edges_a_wider_k_discovers(db):
    narrow = db.create_semantic_graph(k=1, min_score=0.0)
    db.create_semantic_graph(k=3, min_score=0.0, replace=False)
    assert _generated_count(db) > narrow.edges_created


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


# --- atomicity -------------------------------------------------------------


def test_failed_rebuild_leaves_the_previous_graph_intact(db):
    """The slow neighbour search runs before anything is deleted."""
    db.create_semantic_graph(k=2, min_score=0.1)
    before = _generated_count(db)
    assert before > 0

    index = db._get_vector_index("default")
    original = index.search
    calls = {"n": 0}

    def flaky(vector, k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("vector backend exploded")
        return original(vector, k)

    index.search = flaky
    try:
        with pytest.raises(RuntimeError, match="exploded"):
            db.create_semantic_graph(k=2, min_score=0.1)
    finally:
        index.search = original

    assert _generated_count(db) == before


def test_a_failed_insert_rolls_the_delete_back(db):
    """The delete and the insert must land together or not at all."""
    db.create_semantic_graph(k=2, min_score=0.1)
    before = _generated_count(db)

    class _FailingInsert:
        """Delegates everything to the real connection except the bulk insert."""

        def __init__(self, conn):
            self._conn = conn

        def executemany(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        def __getattr__(self, name):
            return getattr(self._conn, name)

    real_conn = db.conn
    db.conn = _FailingInsert(real_conn)
    try:
        with pytest.raises(DatabaseError, match="Failed to create semantic graph"):
            db.create_semantic_graph(k=2, min_score=0.1)
    finally:
        db.conn = real_conn

    assert _generated_count(db) == before


def test_rebuild_inside_an_explicit_transaction_is_rolled_back(db):
    db.create_semantic_graph(k=2, min_score=0.1)
    before = _generated_count(db)

    db.begin_transaction()
    db.create_semantic_graph(k=3, min_score=0.0)
    db.rollback()

    assert _generated_count(db) == before


# --- approximate scoping ---------------------------------------------------


def test_approximate_ignores_hand_made_edges(db):
    """A manual edge must not exclude its endpoints from ever being linked."""
    a = db.match_nodes(labels=["Doc"], properties={"id": "d1"})[0]
    b = db.match_nodes(labels=["Doc"], properties={"id": "d5"})[0]
    db.create_relationship(a.id, b.id, "SEMANTIC_SIMILAR", {"manual": True})

    report = db.refresh_semantic_graph(k=2, min_score=0.1)
    assert report.nodes_skipped == 0
    assert report.nodes_processed == 5

    linked = db.execute("""
        MATCH (n:Doc)-[r:SEMANTIC_SIMILAR]->()
        WHERE r.generated_by IS NOT NULL
        RETURN DISTINCT n.id AS id
    """)
    assert {"d1", "d5"} & {row["id"] for row in linked}


def test_approximate_ignores_edges_from_another_index(db):
    """Edges generated from a different index say nothing about this one."""
    db.create_vector_index("other", dim=32, embedding_function=_Embedder(),
                           options={"store_embeddings": True})
    embedder = _Embedder()
    for node in db.match_nodes(labels=["Doc"]):
        db.upsert_embedding(node.id, embedder([node.properties["text"]])[0], index="other")
    db.create_semantic_graph(index="other", k=2, min_score=0.1)

    report = db.refresh_semantic_graph(index="default", k=2, min_score=0.1)
    assert report.nodes_skipped == 0
    assert report.nodes_processed == 5


def test_approximate_still_skips_its_own_edges(db):
    db.create_semantic_graph(k=2, min_score=0.1)
    report = db.refresh_semantic_graph(k=2, min_score=0.1)
    assert report.nodes_processed == 0
    assert report.nodes_skipped == 5


# --- per-index scoping -----------------------------------------------------


def _seed_second_index(db) -> None:
    db.create_vector_index("other", dim=32, embedding_function=_Embedder(),
                           options={"store_embeddings": True})
    embedder = _Embedder()
    for node in db.match_nodes(labels=["Doc"]):
        db.upsert_embedding(node.id, embedder([node.properties["text"]])[0], index="other")


def test_rebuilding_one_index_spares_another(db):
    _seed_second_index(db)
    db.create_semantic_graph(index="default", k=2, min_score=0.1)
    after_first = _generated_count(db)
    db.create_semantic_graph(index="other", k=2, min_score=0.1)
    assert _generated_count(db) > after_first

    db.create_semantic_graph(index="default", k=2, min_score=0.1)
    assert _generated_count(db) > after_first


def test_drop_can_target_one_index(db):
    _seed_second_index(db)
    db.create_semantic_graph(index="default", k=2, min_score=0.1)
    default_edges = _generated_count(db)
    db.create_semantic_graph(index="other", k=2, min_score=0.1)

    dropped = db.drop_semantic_graph(index="other")
    assert dropped > 0
    assert _generated_count(db) == default_edges


# --- min_score defaults ----------------------------------------------------


def test_default_min_score_is_not_permissive(db):
    """0.0 would admit orthogonal neighbours; the default must not."""
    from grafito.database import DEFAULT_SEMANTIC_GRAPH_MIN_SCORE

    default = db.create_semantic_graph(k=4)
    permissive = db.create_semantic_graph(k=4, min_score=0.0)
    assert DEFAULT_SEMANTIC_GRAPH_MIN_SCORE > 0.0
    assert default.edges_created < permissive.edges_created


def test_l2_index_requires_an_explicit_min_score():
    """l2 scores are negative, so a cosine-shaped default builds nothing."""
    database = GrafitoDatabase(':memory:')
    database.create_vector_index(
        "default", dim=32, embedding_function=_Embedder(),
        options={"store_embeddings": True, "metric": "l2"},
    )
    database.index_documents(DOCS, label="Doc")

    with pytest.raises(DatabaseError, match="explicit min_score"):
        database.create_semantic_graph(k=2)

    report = database.create_semantic_graph(k=2, min_score=-2.0)
    assert report.edges_created > 0
    database.close()


def test_explicit_min_score_of_zero_is_honoured(db):
    """None means 'pick a default'; 0.0 means 0.0."""
    assert db.create_semantic_graph(k=4, min_score=0.0).edges_created > 0


# --- min_score validation --------------------------------------------------


@pytest.mark.parametrize("bad", [True, False, "0.5", [0.5], object()])
def test_min_score_rejects_non_numbers(db, bad):
    """bool is a subclass of int: float(True) would silently mean 1.0."""
    with pytest.raises(DatabaseError, match="min_score must be a number"):
        db.create_semantic_graph(k=2, min_score=bad)


def test_min_score_accepts_ints(db):
    assert db.create_semantic_graph(k=2, min_score=0).edges_created > 0


# --- processed vs merely linked --------------------------------------------


def test_a_truncated_build_does_not_strand_nodes(db):
    """A node that only *received* an edge was never searched; it must not be
    treated as done, or a truncated build would exclude it forever."""
    truncated = db.create_semantic_graph(k=2, min_score=0.0, max_edges=3)
    assert truncated.truncated is True

    sources = {
        row["id"]
        for row in db.execute("""
            MATCH (a:Doc)-[r:SEMANTIC_SIMILAR]->()
            WHERE r.generated_by IS NOT NULL RETURN DISTINCT a.id AS id
        """)
    }
    targets = {
        row["id"]
        for row in db.execute("""
            MATCH (a:Doc)<-[r:SEMANTIC_SIMILAR]-()
            WHERE r.generated_by IS NOT NULL RETURN DISTINCT a.id AS id
        """)
    }
    receivers_only = targets - sources
    assert receivers_only, "fixture no longer produces the case under test"

    db.refresh_semantic_graph(k=2, min_score=0.0)

    for doc_id in receivers_only:
        reachable = db.execute(
            "MATCH (a:Doc {id: $id})-[:SEMANTIC_SIMILAR]-(b:Doc) "
            "RETURN count(DISTINCT b.id) AS c",
            {"id": doc_id},
        )
        assert reachable[0]["c"] > 0


def test_refresh_is_idempotent(db):
    """Re-running must converge, not accumulate."""
    db.create_semantic_graph(k=2, min_score=0.0)
    db.refresh_semantic_graph(k=2, min_score=0.0)
    settled = _generated_count(db)
    db.refresh_semantic_graph(k=2, min_score=0.0)
    db.refresh_semantic_graph(k=2, min_score=0.0)
    assert _generated_count(db) == settled


def test_directed_mode_also_deduplicates_against_the_database(db):
    first = db.create_semantic_graph(k=2, min_score=0.1, undirected=False)
    db.create_semantic_graph(k=2, min_score=0.1, undirected=False, replace=False)
    assert _generated_count(db) == first.edges_created
