"""SIMILAR() / VECTOR_SCORE(): pointwise vector predicates inside Cypher."""

import pytest

from grafito import GrafitoDatabase
from grafito.embedding_functions import EmbeddingFunction


class _AxisEmbedder(EmbeddingFunction):
    """Maps a query to a basis vector, so scores are exactly 1.0 or 0.0.

    "x" -> [1, 0], "y" -> [0, 1], anything else -> [0, 0].
    """

    calls: int = 0

    def __call__(self, input: list[str]) -> list[list[float]]:
        type(self).calls += len(input)
        return [
            [1.0, 0.0] if text == "x" else [0.0, 1.0] if text == "y" else [0.0, 0.0]
            for text in input
        ]

    @staticmethod
    def name() -> str:
        return "axis_test_embedder"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "ip"]

    @staticmethod
    def build_from_config(config: dict) -> "_AxisEmbedder":
        return _AxisEmbedder()

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    @property
    def dimension(self) -> int:
        return 2


def _db(metric: str = "cosine", *, embedder: bool = True) -> GrafitoDatabase:
    """Alice on the x axis, Bob on the y axis, Carol with no embedding at all."""
    db = GrafitoDatabase(':memory:')
    db.create_vector_index(
        "vec",
        dim=2,
        options={"store_embeddings": True, "metric": metric},
        embedding_function=_AxisEmbedder() if embedder else None,
    )
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.create_node(labels=["Person"], properties={"name": "Carol"})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="vec")
    db.upsert_embedding(bob.id, [0.0, 1.0], index="vec")
    db.create_relationship(alice.id, bob.id, "KNOWS")
    return db


def _names(rows: list[dict]) -> list[str]:
    return sorted(row["name"] for row in rows)


# --- VECTOR_SCORE ----------------------------------------------------------


def test_vector_score_returns_metric_score():
    db = _db()
    rows = db.execute("""
        MATCH (n:Person)
        RETURN n.name AS name, VECTOR_SCORE(n, [1.0, 0.0], {index: 'vec'}) AS s
    """)
    assert {row["name"]: row["s"] for row in rows} == {
        "Alice": 1.0,
        "Bob": 0.0,
        "Carol": None,  # no embedding: unknown, not distant
    }
    db.close()


def test_vector_score_accepts_query_string_via_embedder():
    db = _db()
    rows = db.execute("""
        MATCH (n:Person) WHERE n.name = 'Alice'
        RETURN VECTOR_SCORE(n, 'x', {index: 'vec'}) AS s
    """)
    assert rows == [{"s": 1.0}]
    db.close()


def test_vector_score_is_usable_for_ordering():
    db = _db()
    rows = db.execute("""
        MATCH (n:Person)
        WHERE VECTOR_SCORE(n, 'x', {index: 'vec'}) IS NOT NULL
        RETURN n.name AS name
        ORDER BY VECTOR_SCORE(n, 'x', {index: 'vec'}) DESC
    """)
    assert [row["name"] for row in rows] == ["Alice", "Bob"]
    db.close()


def test_vector_score_rejects_bare_threshold():
    """A number means min_score, which VECTOR_SCORE has no use for."""
    db = _db()
    with pytest.raises(Exception, match="must be an options map"):
        db.execute("MATCH (n:Person) RETURN VECTOR_SCORE(n, 'x', 0.5)")
    db.close()


# --- SIMILAR ---------------------------------------------------------------


def test_similar_uses_default_threshold():
    db = _db()
    rows = db.execute("""
        MATCH (n:Person)
        WHERE SIMILAR(n, 'x', {index: 'vec'})
        RETURN n.name AS name
    """)
    assert _names(rows) == ["Alice"]
    db.close()


def test_similar_with_explicit_threshold_widens_the_match():
    db = _db()
    rows = db.execute("""
        MATCH (n:Person)
        WHERE SIMILAR(n, 'x', {index: 'vec', min_score: 0.0})
        RETURN n.name AS name
    """)
    assert _names(rows) == ["Alice", "Bob"]
    db.close()


def test_similar_bare_number_is_min_score():
    """The shorthand form carries a threshold but no index, so it uses `default`."""
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("default", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(alice.id, [1.0, 0.0])
    db.upsert_embedding(bob.id, [0.0, 1.0])

    wide = db.execute("""
        MATCH (n:Person) WHERE SIMILAR(n, [1.0, 0.0], 0.0) RETURN n.name AS name
    """)
    narrow = db.execute("""
        MATCH (n:Person) WHERE SIMILAR(n, [1.0, 0.0], 0.9) RETURN n.name AS name
    """)
    assert _names(wide) == ["Alice", "Bob"]
    assert _names(narrow) == ["Alice"]
    db.close()


def test_similar_is_false_for_nodes_without_embedding():
    """False, not NULL — an unindexed node must be dropped, not propagate unknown."""
    db = _db()
    rows = db.execute("""
        MATCH (n:Person)
        WHERE SIMILAR(n, [1.0, 0.0], {index: 'vec', min_score: -1.0})
        RETURN n.name AS name
    """)
    assert "Carol" not in _names(rows)
    db.close()


def test_similar_accepts_query_parameter():
    db = _db()
    rows = db.execute(
        "MATCH (n:Person) WHERE SIMILAR(n, $q, {index: 'vec'}) RETURN n.name AS name",
        {"q": "y"},
    )
    assert _names(rows) == ["Bob"]
    db.close()


def test_similar_constrains_the_far_end_of_a_path():
    """The intended shape: ANN seeds the pattern, SIMILAR filters the far end."""
    db = _db()
    rows = db.execute("""
        CALL db.vector.search('vec', [1.0, 0.0], 1) YIELD node AS a
        MATCH (a)-[:KNOWS]->(b)
        WHERE SIMILAR(b, 'y', {index: 'vec'})
        RETURN a.name AS ini, b.name AS fin
    """)
    assert rows == [{"ini": "Alice", "fin": "Bob"}]
    db.close()


def test_query_embedding_is_computed_once_per_execution():
    """The embedder must not be re-run for every candidate row."""
    db = _db()
    _AxisEmbedder.calls = 0
    db.execute("""
        MATCH (n:Person)
        WHERE SIMILAR(n, 'x', {index: 'vec'}) OR SIMILAR(n, 'x', {index: 'vec'})
        RETURN n.name AS name
    """)
    assert _AxisEmbedder.calls == 1
    db.close()


# --- errors and edge cases -------------------------------------------------


def test_similar_requires_explicit_threshold_on_l2_index():
    """l2 scores are <= 0, so the cosine-calibrated default would match nothing."""
    db = _db(metric="l2")
    with pytest.raises(Exception, match="explicit min_score"):
        db.execute("MATCH (n:Person) WHERE SIMILAR(n, [1.0, 0.0], {index: 'vec'}) RETURN n")

    rows = db.execute("""
        MATCH (n:Person)
        WHERE SIMILAR(n, [1.0, 0.0], {index: 'vec', min_score: -0.5})
        RETURN n.name AS name
    """)
    assert _names(rows) == ["Alice"]
    db.close()


def test_similar_rejects_unknown_options():
    db = _db()
    with pytest.raises(Exception, match="unknown options"):
        db.execute("MATCH (n:Person) WHERE SIMILAR(n, 'x', {indice: 'vec'}) RETURN n")
    db.close()


def test_similar_rejects_non_node_first_argument():
    db = _db()
    with pytest.raises(Exception, match="expects a node"):
        db.execute("MATCH (n:Person) WHERE SIMILAR(n.name, 'x') RETURN n")
    db.close()


def test_similar_rejects_bad_arity():
    db = _db()
    with pytest.raises(Exception, match="expects 2 or 3 arguments"):
        db.execute("MATCH (n:Person) WHERE SIMILAR(n) RETURN n")
    db.close()


def test_similar_on_null_node_is_null():
    """OPTIONAL MATCH leaves NULL; that is unknown similarity, so the row drops."""
    db = _db()
    rows = db.execute("""
        MATCH (n:Person) WHERE n.name = 'Carol'
        OPTIONAL MATCH (n)-[:KNOWS]->(m)
        RETURN n.name AS name, SIMILAR(m, 'x', {index: 'vec'}) AS sim
    """)
    assert rows == [{"name": "Carol", "sim": None}]
    db.close()


def test_query_string_without_embedder_is_rejected():
    db = _db(embedder=False)
    with pytest.raises(Exception, match="no embedding function"):
        db.execute("MATCH (n:Person) WHERE SIMILAR(n, 'x', {index: 'vec'}) RETURN n")
    db.close()


def test_vector_score_defaults_to_the_default_index():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("default", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    db.upsert_embedding(alice.id, [1.0, 0.0])
    rows = db.execute("MATCH (n:Person) RETURN VECTOR_SCORE(n, [1.0, 0.0]) AS s")
    assert rows == [{"s": 1.0}]
    db.close()
