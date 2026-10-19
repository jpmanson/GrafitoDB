"""Regression tests: relationship type alternatives [:A|B]."""

import pytest
from grafito import GrafitoDatabase
from grafito.cypher.exceptions import CypherExecutionError


@pytest.fixture
def db():
    database = GrafitoDatabase(":memory:")
    database.execute("CREATE (a:N {n: 'a'})-[:X]->(b:N {n: 'b'})-[:Y]->(c:N {n: 'c'})-[:Z]->(d:N {n: 'd'})")
    yield database
    database.close()


def test_single_hop_alternatives(db):
    rows = db.execute("MATCH (a:N)-[:X|Y]->(b) RETURN a.n AS a, b.n AS b ORDER BY a")
    assert rows == [{"a": "a", "b": "b"}, {"a": "b", "b": "c"}]


def test_colon_form(db):
    rows = db.execute("MATCH (a:N)-[:X|:Z]->(b) RETURN a.n AS a ORDER BY a")
    assert rows == [{"a": "a"}, {"a": "c"}]


def test_undirected_with_type(db):
    rows = db.execute("MATCH (b {n: 'b'})-[r:X|Y]-(o) RETURN o.n AS o, type(r) AS t ORDER BY o")
    assert rows == [{"o": "a", "t": "X"}, {"o": "c", "t": "Y"}]


def test_variable_length(db):
    rows = db.execute("MATCH (a {n: 'a'})-[:X|Y*]->(b) RETURN b.n AS b ORDER BY b")
    assert rows == [{"b": "b"}, {"b": "c"}]


def test_chain_and_predicate(db):
    assert db.execute("MATCH (a)-[:X|Y]->()-[:Y|Z]->(c) RETURN a.n AS a ORDER BY a") == [{"a": "a"}, {"a": "b"}]
    assert db.execute("MATCH (n:N) WHERE (n)-[:Y|Z]->() RETURN n.n AS n ORDER BY n") == [{"n": "b"}, {"n": "c"}]


def test_create_rejects_alternatives(db):
    with pytest.raises(CypherExecutionError):
        db.execute("CREATE (q)-[:X|Y]->(w)")
