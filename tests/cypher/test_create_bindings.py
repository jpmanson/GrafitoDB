"""Regression tests: CREATE shares bindings across patterns, supports chains, and
accepts a parameter as the whole property map."""

import pytest
from grafito import GrafitoDatabase
from grafito.cypher.exceptions import CypherExecutionError


def counts(db):
    nodes = db.execute("MATCH (n) RETURN count(n) AS c")[0]["c"]
    rels = db.execute("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    return nodes, rels


@pytest.fixture
def db():
    database = GrafitoDatabase(":memory:")
    yield database
    database.close()


class TestCreateSharedBindings:
    def test_comma_patterns_reuse_variables(self, db):
        db.execute("CREATE (a:P {n: 'a'}), (b:P {n: 'b'}), (a)-[:R]->(b)")
        assert counts(db) == (2, 1)
        rows = db.execute("MATCH (x)-[:R]->(y) RETURN x.n AS x, y.n AS y")
        assert rows == [{"x": "a", "y": "b"}]

    def test_two_create_clauses_reuse_variables(self, db):
        db.execute("CREATE (a:P {n: 'a'}), (b:P {n: 'b'}) CREATE (a)-[:R]->(b)")
        assert counts(db) == (2, 1)

    def test_chain_with_both_directions(self, db):
        db.execute("CREATE (a:P {n: 'a'})-[:R]->(b:P {n: 'b'})<-[:S {k: 1}]-(c:P {n: 'c'})")
        assert counts(db) == (3, 2)
        assert db.execute("MATCH (x)-[r:S]->(y) RETURN x.n AS x, r.k AS k, y.n AS y") == [
            {"x": "c", "k": 1, "y": "b"}
        ]

    def test_return_reads_created_properties_in_one_row(self, db):
        rows = db.execute(
            "CREATE (a:P {n: 'a'}), (b:P {n: 'b'}), (a)-[r:R]->(b) "
            "RETURN a.n AS a, b.n AS b, type(r) AS t"
        )
        assert rows == [{"a": "a", "b": "b", "t": "R"}]

    def test_create_after_empty_match_creates_nothing(self, db):
        assert db.execute("MATCH (x:Missing) CREATE (y:Q) RETURN y") == []
        assert counts(db) == (0, 0)

    def test_single_clause_result_shape_unchanged(self, db):
        rows = db.execute("CREATE (n:Person {name: 'Alice'})")
        assert rows[0]["n"]["labels"] == ["Person"]
        rows = db.execute("CREATE (:Person)")
        assert "created" in rows[0]


class TestPropertyMapParameter:
    def test_node_property_map(self, db):
        db.execute("CREATE (:P $props)", {"props": {"n": "a", "x": 1}})
        assert db.execute("MATCH (p:P) RETURN p.n AS n, p.x AS x") == [{"n": "a", "x": 1}]

    def test_relationship_property_map(self, db):
        db.execute("CREATE (:P {n: 'a'}), (:P {n: 'b'})")
        db.execute(
            "MATCH (a:P {n: $a}), (b:P {n: $b}) CREATE (a)-[:R $props]->(b)",
            {"a": "a", "b": "b", "props": {"year": 1812}},
        )
        assert db.execute("MATCH ()-[r:R]->() RETURN r.year AS y") == [{"y": 1812}]

    def test_non_map_parameter_is_rejected(self, db):
        with pytest.raises(CypherExecutionError):
            db.execute("CREATE (:P $props)", {"props": 3})
