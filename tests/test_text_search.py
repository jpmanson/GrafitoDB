import sqlite3

import pytest

from grafito import GrafitoDatabase
from grafito.exceptions import DatabaseError
from grafito.text_index.sqlite_fts import SQLiteFTSIndex


def _fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE fts5_check USING fts5(content)")
        conn.execute("DROP TABLE fts5_check")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _fts5_available(), reason="FTS5 not available")


@pytest.fixture
def db():
    instance = GrafitoDatabase(":memory:")
    try:
        yield instance
    finally:
        instance.close()


def test_text_index_create_list_drop(db):
    db.create_text_index("node", "Person", ["name", "bio"])

    indexes = db.list_text_indexes()
    props = [row["property"] for row in indexes if row["label_or_type"] == "Person"]
    assert set(props) == {"name", "bio"}

    db.drop_text_index("node", "Person", "bio")
    indexes = db.list_text_indexes()
    props = [row["property"] for row in indexes if row["label_or_type"] == "Person"]
    assert props == ["name"]


def test_text_search_nodes_with_label_filter(db):
    db.create_text_index("node", None, ["name"])

    alice = db.create_node(labels=["Person", "Employee"], properties={"name": "Alice"})
    db.create_node(labels=["Person"], properties={"name": "Bob"})

    results = db.text_search("Alice", labels=["Employee"])
    assert len(results) == 1
    assert results[0]["entity"].id == alice.id
    assert results[0]["entity_type"] == "node"


def test_text_search_relationships(db):
    db.create_text_index("relationship", "WORKS_AT", ["role"])

    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    acme = db.create_node(labels=["Company"], properties={"name": "Acme"})
    rel = db.create_relationship(alice.id, acme.id, "WORKS_AT", {"role": "Engineer"})

    results = db.text_search("Engineer", rel_types=["WORKS_AT"])
    assert len(results) == 1
    assert results[0]["entity"].id == rel.id
    assert results[0]["entity_type"] == "relationship"


def test_text_search_rebuild_index(db):
    node = db.create_node(labels=["Doc"], properties={"title": "Graph Databases"})

    db.create_text_index("node", "Doc", ["title"])
    results = db.text_search("Graph", labels=["Doc"])
    assert results == []

    db.rebuild_text_index()
    results = db.text_search("Graph", labels=["Doc"])
    assert len(results) == 1
    assert results[0]["entity"].id == node.id


@pytest.mark.parametrize("query", ["graph databases?", "graph, databases", '"unbalanced'])
def test_text_search_reports_invalid_fts_syntax_as_a_database_error(db, query):
    """text_search takes FTS5 syntax, so bad syntax fails — but as our error type.

    Free-text callers want :meth:`hybrid_search`, which retries these as literals.
    """
    db.create_text_index("node", "Doc", ["title"])
    db.create_node(labels=["Doc"], properties={"title": "Graph Databases"})

    with pytest.raises(DatabaseError, match="Failed to search text index"):
        db.text_search(query, labels=["Doc"])


def test_text_search_still_honours_fts_operators(db):
    """The wrapping must not swallow the query language the docs promise."""
    db.create_text_index("node", "Doc", ["title"])
    db.create_node(labels=["Doc"], properties={"title": "Graph Databases"})

    assert len(db.text_search("graph AND databases", labels=["Doc"])) == 1
    assert len(db.text_search("graph OR nothing", labels=["Doc"])) == 1
    assert len(db.text_search("dat*", labels=["Doc"])) == 1


def test_custom_fts_backend_reports_invalid_syntax_as_a_database_error(db):
    index = SQLiteFTSIndex(db.conn, "custom")
    index.add([1], ["graph databases"])

    with pytest.raises(DatabaseError, match="Failed to search text index"):
        index.search("graph databases?", k=2)
