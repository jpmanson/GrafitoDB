"""Tests for the graph tool tiers (grafito.tools): GraphTools, CypherTools.

These hang on a plain GrafitoDatabase, not on OKF — the tests use only the core,
which is the point: a graph deployment must not need OKF.
"""

from __future__ import annotations

import json

import pytest

from grafito import CypherTools, GrafitoDatabase, GraphTools


@pytest.fixture
def db():
    database = GrafitoDatabase(":memory:")
    alice = database.create_node(["Person"], {"name": "Alice"}).id
    bob = database.create_node(["Person", "Employee"], {"name": "Bob"}).id
    acme = database.create_node(["Company"], {"name": "Acme"}).id
    database.create_relationship(alice, bob, "KNOWS", {})
    database.create_relationship(bob, acme, "WORKS_AT", {})
    database._ids = {"alice": alice, "bob": bob, "acme": acme}
    yield database
    database.close()


def test_graph_tools_do_not_drag_in_okf():
    """Importing the graph tools must not pull the OKF package."""
    import subprocess
    import sys

    code = "import grafito.tools, sys; assert not any(m.startswith('grafito.okf') for m in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_graph_schema_reports_labels_types_and_counts(db):
    schema = json.loads(GraphTools(db).call("graph_schema", {}))
    assert schema["labels"] == ["Company", "Employee", "Person"]
    assert schema["relationship_types"] == ["KNOWS", "WORKS_AT"]
    assert schema["node_count"] == 3
    assert schema["indexes"] == []


def test_graph_neighbors_traverses_by_id(db):
    out = json.loads(
        GraphTools(db).call("graph_neighbors", {"node_id": db._ids["bob"], "direction": "both"})
    )
    names = {node["properties"]["name"] for node in out}
    assert names == {"Alice", "Acme"}

    outgoing = json.loads(
        GraphTools(db).call(
            "graph_neighbors", {"node_id": db._ids["bob"], "direction": "outgoing"}
        )
    )
    assert {n["properties"]["name"] for n in outgoing} == {"Acme"}


def test_graph_tools_unknown_name_is_a_json_error(db):
    assert "error" in json.loads(GraphTools(db).call("nope", {}))


def test_cypher_query_returns_rows(db):
    result = json.loads(CypherTools(db).call("graph_query", {"query": "MATCH (n) RETURN count(n) AS c"}))
    assert result["rows"] == [{"c": 3}]
    assert result["truncated"] is False


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) DELETE n",
        "CREATE (n:X)",
        "MATCH (n) SET n.x = 1",
        "MATCH (n) DETACH DELETE n",
        "MERGE (n:X {id: 1})",
        "MATCH (n) REMOVE n.x",
    ],
)
def test_cypher_query_refuses_mutations(db, query):
    result = json.loads(CypherTools(db).call("graph_query", {"query": query}))
    assert "error" in result and "Read-only" in result["error"]


def test_cypher_query_caps_rows(db):
    result = json.loads(
        CypherTools(db, max_rows=2).call("graph_query", {"query": "MATCH (n) RETURN n.name AS name"})
    )
    assert len(result["rows"]) == 2
    assert result["count"] == 3 and result["truncated"] is True


def test_cypher_tools_raise_errors_propagates(db):
    with pytest.raises(ValueError):
        CypherTools(db, raise_errors=True).call("graph_query", {"query": "CREATE (n:X)"})
