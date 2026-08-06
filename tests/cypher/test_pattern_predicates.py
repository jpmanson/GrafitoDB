"""Patterns used as boolean expressions: (a)-[:R]->(b) in WHERE and RETURN."""

import pytest

from grafito import GrafitoDatabase


@pytest.fixture
def db() -> GrafitoDatabase:
    """a -LINKS-> b -CITES-> c, plus an isolated node."""
    database = GrafitoDatabase(':memory:')
    a = database.create_node(labels=["Doc"], properties={"id": "a"})
    b = database.create_node(labels=["Doc"], properties={"id": "b"})
    c = database.create_node(labels=["Doc"], properties={"id": "c"})
    database.create_node(labels=["Doc"], properties={"id": "lonely"})
    database.create_relationship(a.id, b.id, "LINKS", {"weight": 2})
    database.create_relationship(b.id, c.id, "CITES")
    yield database
    database.close()


def _ids(rows: list[dict]) -> list[str]:
    return sorted(row["id"] for row in rows)


# --- as a filter ------------------------------------------------------------


def test_outgoing_any_type(db):
    rows = db.execute("MATCH (n:Doc) WHERE (n)-->() RETURN n.id AS id")
    assert _ids(rows) == ["a", "b"]


def test_negated_finds_the_sinks(db):
    """The canonical use: nodes with no outgoing relationship."""
    rows = db.execute("MATCH (n:Doc) WHERE NOT (n)-->() RETURN n.id AS id")
    assert _ids(rows) == ["c", "lonely"]


def test_typed_relationship(db):
    rows = db.execute("MATCH (n:Doc) WHERE (n)-[:LINKS]->() RETURN n.id AS id")
    assert _ids(rows) == ["a"]


def test_negated_typed_relationship(db):
    rows = db.execute("MATCH (n:Doc) WHERE NOT (n)-[:CITES]->() RETURN n.id AS id")
    assert _ids(rows) == ["a", "c", "lonely"]


def test_incoming_direction(db):
    rows = db.execute("MATCH (n:Doc) WHERE ()-->(n) RETURN n.id AS id")
    assert _ids(rows) == ["b", "c"]


def test_undirected(db):
    rows = db.execute("MATCH (n:Doc) WHERE (n)--() RETURN n.id AS id")
    assert _ids(rows) == ["a", "b", "c"]


def test_multi_hop(db):
    rows = db.execute("MATCH (n:Doc) WHERE (n)-->()-->() RETURN n.id AS id")
    assert _ids(rows) == ["a"]


def test_variable_length(db):
    rows = db.execute("MATCH (n:Doc) WHERE (n)-[*1..2]->() RETURN n.id AS id")
    assert _ids(rows) == ["a", "b"]


def test_inline_properties_on_the_far_node(db):
    rows = db.execute("MATCH (n:Doc) WHERE (n)-->({id: 'b'}) RETURN n.id AS id")
    assert _ids(rows) == ["a"]


def test_inline_properties_on_the_relationship(db):
    rows = db.execute("MATCH (n:Doc) WHERE (n)-[:LINKS {weight: 2}]->() RETURN n.id AS id")
    assert _ids(rows) == ["a"]


def test_label_on_the_far_node(db):
    other = db.create_node(labels=["Other"], properties={"id": "other"})
    a = db.match_nodes(labels=["Doc"], properties={"id": "a"})[0]
    db.create_relationship(a.id, other.id, "LINKS")

    rows = db.execute("MATCH (n:Doc) WHERE (n)-->(:Other) RETURN n.id AS id")
    assert _ids(rows) == ["a"]


# --- bindings from the surrounding scope ------------------------------------


def test_both_endpoints_bound(db):
    """A pattern over two bound variables asks about *those* nodes."""
    rows = db.execute("""
        MATCH (n:Doc), (m:Doc {id: 'c'})
        WHERE (n)-->(m)
        RETURN n.id AS id
    """)
    assert _ids(rows) == ["b"]


def test_bound_variable_is_not_rebound(db):
    """`n` in the predicate is the matched node, not a fresh one."""
    rows = db.execute("""
        MATCH (n:Doc {id: 'lonely'})
        WHERE (n)-->()
        RETURN n.id AS id
    """)
    assert rows == []


# --- composition ------------------------------------------------------------


def test_combines_with_other_predicates(db):
    rows = db.execute("""
        MATCH (n:Doc) WHERE (n)-->() AND n.id <> 'a' RETURN n.id AS id
    """)
    assert _ids(rows) == ["b"]


def test_or(db):
    rows = db.execute("""
        MATCH (n:Doc) WHERE (n)-[:CITES]->() OR n.id = 'a' RETURN n.id AS id
    """)
    assert _ids(rows) == ["a", "b"]


def test_usable_as_a_returned_value(db):
    rows = db.execute("MATCH (n:Doc) RETURN n.id AS id, (n)-->() AS has_out ORDER BY n.id")
    assert rows == [
        {"id": "a", "has_out": True},
        {"id": "b", "has_out": True},
        {"id": "c", "has_out": False},
        {"id": "lonely", "has_out": False},
    ]


def test_usable_in_with(db):
    """Projecting through WITH.

    Note: filtering on the projected alias inside the same WITH
    (``WITH ... AS h WHERE h``) is unsupported for *any* expression, not just
    patterns — ``WITH n, n.id AS x WHERE x = 'a'`` fails the same way. Filter in
    a later clause instead.
    """
    rows = db.execute("""
        MATCH (n:Doc)
        WITH n, (n)-->() AS has_out
        RETURN n.id AS id, has_out
        ORDER BY n.id
    """)
    assert [row["has_out"] for row in rows] == [True, True, False, False]


def test_filtering_after_a_with_projection(db):
    rows = db.execute("""
        MATCH (n:Doc)
        WITH n
        WHERE (n)-->()
        RETURN n.id AS id
    """)
    assert _ids(rows) == ["a", "b"]


# --- exists() ---------------------------------------------------------------


def test_exists_over_a_pattern(db):
    rows = db.execute("MATCH (n:Doc) WHERE exists((n)-[:LINKS]->()) RETURN n.id AS id")
    assert _ids(rows) == ["a"]


def test_not_exists_over_a_pattern(db):
    rows = db.execute("MATCH (n:Doc) WHERE NOT exists((n)-->()) RETURN n.id AS id")
    assert _ids(rows) == ["c", "lonely"]


def test_exists_over_a_pattern_returns_false_not_truthy(db):
    """exists(<bool>) would be True for both; the pattern form must not be."""
    rows = db.execute("MATCH (n:Doc) RETURN n.id AS id, exists((n)-->()) AS e ORDER BY n.id")
    assert [row["e"] for row in rows] == [True, True, False, False]


def test_exists_over_a_property_still_works(db):
    """The pattern special case must not shadow the original meaning."""
    rows = db.execute("MATCH (n:Doc) WHERE exists(n.id) RETURN n.id AS id")
    assert _ids(rows) == ["a", "b", "c", "lonely"]

    rows = db.execute("MATCH (n:Doc) WHERE exists(n.missing) RETURN n.id AS id")
    assert rows == []


# --- parenthesised expressions must keep working ----------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("(1 + 2) * 3", 9),
        ("((1 + 2))", 3),
        ("(1 + 2) - (3 - 1)", 1),
        ("(true)", True),
        ("('a' + 'b')", "ab"),
    ],
)
def test_grouping_is_not_read_as_a_pattern(expression, expected):
    """Both start with '('; only a relationship arrow distinguishes them."""
    db = GrafitoDatabase(':memory:')
    assert db.execute(f"RETURN {expression} AS v") == [{"v": expected}]
    db.close()


def test_single_node_parenthesis_is_not_a_predicate(db):
    """`(n)` alone is a grouped variable, not a pattern to match."""
    rows = db.execute("MATCH (n:Doc {id: 'a'}) RETURN (n).id AS id")
    assert rows == [{"id": "a"}]


def test_pattern_comprehension_still_parses(db):
    """The comprehension form shares the pattern parser; it must not regress."""
    rows = db.execute(
        "MATCH (n:Doc) RETURN n.id AS id, size([(n)-->(x) | x]) AS n_out ORDER BY n.id"
    )
    assert [row["n_out"] for row in rows] == [1, 1, 0, 0]
