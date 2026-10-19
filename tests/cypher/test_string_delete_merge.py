"""Regression tests: string predicates, DETACH DELETE and RETURN after a leading MERGE."""

import pytest
from grafito import GrafitoDatabase
from grafito.cypher.exceptions import CypherExecutionError


@pytest.fixture
def db():
    database = GrafitoDatabase(":memory:")
    database.execute(
        "CREATE (a:C {nombre: 'Ana'}), (b:C {nombre: 'Bea'}), (c:C {nombre: 'Ceci'}), "
        "(a)-[:R]->(b), (c)-[:R]->(b)"
    )
    yield database
    database.close()


def names(rows):
    return sorted(row["n"] for row in rows)


class TestStringPredicates:
    def test_contains(self, db):
        assert names(db.execute("MATCH (x:C) WHERE x.nombre CONTAINS 'e' RETURN x.nombre AS n")) == ["Bea", "Ceci"]

    def test_starts_with(self, db):
        assert names(db.execute("MATCH (x:C) WHERE x.nombre STARTS WITH 'A' RETURN x.nombre AS n")) == ["Ana"]

    def test_ends_with(self, db):
        assert names(db.execute("MATCH (x:C) WHERE x.nombre ENDS WITH 'a' RETURN x.nombre AS n")) == ["Ana", "Bea"]

    def test_regex_is_full_match(self, db):
        assert names(db.execute("MATCH (x:C) WHERE x.nombre =~ '(?i)c.*' RETURN x.nombre AS n")) == ["Ceci"]
        assert db.execute("MATCH (x:C) WHERE x.nombre =~ 'e' RETURN x.nombre AS n") == []

    def test_not_contains(self, db):
        assert names(db.execute("MATCH (x:C) WHERE NOT x.nombre CONTAINS 'e' RETURN x.nombre AS n")) == ["Ana"]

    def test_non_string_is_null(self, db):
        assert db.execute("RETURN 3 CONTAINS 'a' AS v") == [{"v": None}]

    def test_contextual_words_still_usable_as_names(self, db):
        db.execute("CREATE (:K {contains: 1, starts: 2})")
        assert db.execute("MATCH (k:K) RETURN k.contains AS c, k.starts AS s") == [{"c": 1, "s": 2}]


class TestDelete:
    def test_delete_connected_node_requires_detach(self, db):
        with pytest.raises(CypherExecutionError, match="DETACH DELETE"):
            db.execute("MATCH (x:C {nombre: 'Bea'}) DELETE x")

    def test_detach_delete_counts_relationships(self, db):
        assert db.execute("MATCH (x:C {nombre: 'Bea'}) DETACH DELETE x") == [
            {"deleted_nodes": 1, "deleted_relationships": 2}
        ]
        assert db.execute("MATCH ()-[r]->() RETURN count(r) AS c") == [{"c": 0}]

    def test_delete_relationship_and_node_together(self, db):
        assert db.execute("MATCH (x:C {nombre: 'Ana'})-[r]->() DELETE r, x") == [
            {"deleted_nodes": 1, "deleted_relationships": 1}
        ]


class TestLeadingMerge:
    def test_return_after_merge_on_match(self, db):
        rows = db.execute(
            "MERGE (z:C {nombre: 'Ana'}) ON CREATE SET z.k = 1 ON MATCH SET z.k = 2 "
            "RETURN z.nombre AS n, z.k AS k"
        )
        assert rows == [{"n": "Ana", "k": 2}]

    def test_return_after_merge_on_create(self, db):
        rows = db.execute("MERGE (z:C {nombre: 'Dora'}) ON CREATE SET z.k = 1 RETURN z.nombre AS n, z.k AS k")
        assert rows == [{"n": "Dora", "k": 1}]
