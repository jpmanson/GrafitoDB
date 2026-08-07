"""What a WITH clause's own WHERE can see."""

import pytest

from grafito import GrafitoDatabase


@pytest.fixture
def db() -> GrafitoDatabase:
    """alice(30) knows bob(20) and carol(40); carol knows nobody."""
    database = GrafitoDatabase(':memory:')
    alice = database.create_node(labels=["P"], properties={"name": "alice", "age": 30})
    bob = database.create_node(labels=["P"], properties={"name": "bob", "age": 20})
    carol = database.create_node(labels=["P"], properties={"name": "carol", "age": 40})
    database.create_relationship(alice.id, bob.id, "KNOWS", {"weight": 9})
    database.create_relationship(alice.id, carol.id, "KNOWS", {"weight": 1})
    yield database
    database.close()


def _names(rows: list[dict]) -> list[str]:
    return sorted(row["r"] for row in rows)


# --- aliases are in scope for the clause's own WHERE -------------------------


def test_property_alias(db):
    rows = db.execute("MATCH (n:P) WITH n, n.age AS age WHERE age > 25 RETURN n.name AS r")
    assert _names(rows) == ["alice", "carol"]


def test_expression_alias(db):
    rows = db.execute(
        "MATCH (n:P) WITH n, (n.age * 2) AS doubled WHERE doubled > 50 RETURN n.name AS r"
    )
    assert _names(rows) == ["alice", "carol"]


def test_renamed_variable(db):
    """`WITH n AS person` must make the predicate see the new name."""
    rows = db.execute(
        "MATCH (n:P) WITH n AS person WHERE person.age > 25 RETURN person.name AS r"
    )
    assert _names(rows) == ["alice", "carol"]


def test_alias_shadows_an_incoming_binding(db):
    """When a projected name reuses an incoming one, the projection wins."""
    rows = db.execute(
        "MATCH (n:P) WITH n.name AS n WHERE n = 'alice' RETURN n AS r"
    )
    assert _names(rows) == ["alice"]


def test_boolean_alias(db):
    rows = db.execute(
        "MATCH (n:P) WITH n, n.age > 25 AS senior WHERE senior RETURN n.name AS r"
    )
    assert _names(rows) == ["alice", "carol"]


def test_pattern_predicate_alias(db):
    rows = db.execute(
        "MATCH (n:P) WITH n, (n)-[:KNOWS]->() AS connected WHERE connected "
        "RETURN n.name AS r"
    )
    assert _names(rows) == ["alice"]


def test_aggregate_alias_still_works(db):
    """The aggregating path already filtered after grouping; keep it that way."""
    rows = db.execute(
        "MATCH (n:P)-[:KNOWS]->(m) WITH n, count(m) AS friends WHERE friends > 1 "
        "RETURN n.name AS r"
    )
    assert _names(rows) == ["alice"]


# --- incoming bindings stay visible -----------------------------------------


def test_filtering_on_an_unprojected_variable(db):
    """Cypher proper drops these from scope; tightening it would break queries."""
    rows = db.execute(
        "MATCH (n:P)-[rel:KNOWS]->(m) WITH n WHERE rel.weight > 5 RETURN n.name AS r"
    )
    assert _names(rows) == ["alice"]


def test_filtering_on_the_original_variable(db):
    rows = db.execute(
        "MATCH (n:P) WITH n, n.age AS age WHERE n.age > 25 RETURN n.name AS r"
    )
    assert _names(rows) == ["alice", "carol"]


def test_mixing_alias_and_incoming_variable(db):
    rows = db.execute(
        "MATCH (n:P) WITH n, n.age AS age WHERE age > 25 AND n.name <> 'carol' "
        "RETURN n.name AS r"
    )
    assert _names(rows) == ["alice"]


# --- projection is unaffected -----------------------------------------------


def test_where_does_not_alter_what_is_projected(db):
    rows = db.execute(
        "MATCH (n:P) WITH n, n.age AS age WHERE age > 25 RETURN n.name AS r, age "
        "ORDER BY age"
    )
    assert [(row["r"], row["age"]) for row in rows] == [("alice", 30), ("carol", 40)]


def test_order_by_and_limit_apply_after_the_filter(db):
    rows = db.execute(
        "MATCH (n:P) WITH n, n.age AS age WHERE age > 19 "
        "RETURN n.name AS r ORDER BY age DESC LIMIT 2"
    )
    assert [row["r"] for row in rows] == ["carol", "alice"]


def test_a_false_predicate_drops_every_row(db):
    assert db.execute(
        "MATCH (n:P) WITH n, n.age AS age WHERE age > 999 RETURN n.name AS r"
    ) == []


def test_null_predicate_drops_the_row(db):
    db.create_node(labels=["P"], properties={"name": "dave"})  # no age
    rows = db.execute(
        "MATCH (n:P) WITH n, n.age AS age WHERE age > 25 RETURN n.name AS r"
    )
    assert "dave" not in _names(rows)


def test_chained_with_clauses(db):
    rows = db.execute(
        "MATCH (n:P) WITH n, n.age AS age WHERE age > 19 "
        "WITH n, age * 10 AS scaled WHERE scaled > 250 "
        "RETURN n.name AS r"
    )
    assert _names(rows) == ["alice", "carol"]


# --- WITH accepts the same expressions RETURN does --------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("n.age * 2", 60),
        ("n.age > 25", True),
        ("n.age + 1", 31),
        ("n.age >= 30 AND n.name = 'alice'", True),
        ("toUpper(n.name)", "ALICE"),
        ("n.age / 2", 15),
    ],
)
def test_with_parses_full_expressions(db, expression, expected):
    """WITH used to read only primary expressions, so `n.age * 2 AS x` failed
    while the identical RETURN item parsed."""
    rows = db.execute(
        f"MATCH (n:P) WHERE n.name = 'alice' WITH {expression} AS value RETURN value AS r"
    )
    assert rows == [{"r": expected}]


def test_with_expression_matches_return_expression(db):
    """The two clauses should accept the same thing."""
    expression = "n.age * 2"
    via_with = db.execute(
        f"MATCH (n:P) WITH n.name AS name, {expression} AS v RETURN name AS r, v ORDER BY name"
    )
    via_return = db.execute(
        f"MATCH (n:P) RETURN n.name AS r, {expression} AS v ORDER BY n.name"
    )
    assert via_with == via_return


def test_expression_alias_is_filterable_without_parentheses(db):
    """The two fixes together: full expression parsed, alias visible to WHERE."""
    rows = db.execute(
        "MATCH (n:P) WITH n, n.age * 2 AS doubled WHERE doubled > 50 RETURN n.name AS r"
    )
    assert _names(rows) == ["alice", "carol"]


# --- WITH * and RETURN * ----------------------------------------------------


def test_with_star_carries_every_variable(db):
    rows = db.execute("MATCH (n:P) WITH * RETURN n.name AS r")
    assert _names(rows) == ["alice", "bob", "carol"]


def test_with_star_carries_all_of_a_multi_variable_match(db):
    rows = db.execute("""
        MATCH (n:P)-[rel:KNOWS]->(m:P)
        WITH *
        RETURN n.name AS r, m.name AS other, rel.weight AS weight
    """)
    assert sorted(
        (row["r"], row["other"], row["weight"]) for row in rows
    ) == [("alice", "bob", 9), ("alice", "carol", 1)]


def test_with_star_plus_an_alias(db):
    rows = db.execute(
        "MATCH (n:P) WITH *, n.age AS age WHERE age > 25 RETURN n.name AS r, age"
    )
    assert [(row["r"], row["age"]) for row in sorted(rows, key=lambda x: x["r"])] == [
        ("alice", 30),
        ("carol", 40),
    ]


def test_with_star_then_where(db):
    rows = db.execute("MATCH (n:P) WITH * WHERE n.age > 25 RETURN n.name AS r")
    assert _names(rows) == ["alice", "carol"]


def test_with_distinct_star(db):
    rows = db.execute("MATCH (n:P)-[:KNOWS]->() WITH DISTINCT * RETURN n.name AS r")
    assert _names(rows) == ["alice"]


def test_with_star_then_order_and_limit(db):
    rows = db.execute(
        "MATCH (n:P) WITH * ORDER BY n.age DESC LIMIT 2 RETURN n.name AS r"
    )
    assert [row["r"] for row in rows] == ["carol", "alice"]


def test_chained_with_star(db):
    rows = db.execute("MATCH (n:P) WITH * WITH * WHERE n.age > 25 RETURN n.name AS r")
    assert _names(rows) == ["alice", "carol"]


def test_return_star(db):
    rows = db.execute("MATCH (n:P) WHERE n.name = 'alice' RETURN *")
    assert list(rows[0]) == ["n"]
    assert rows[0]["n"].properties["name"] == "alice"


def test_return_star_with_several_variables(db):
    rows = db.execute("MATCH (n:P)-[rel:KNOWS]->(m:P) WHERE m.name = 'bob' RETURN *")
    assert sorted(rows[0]) == ["m", "n", "rel"]


def test_star_does_not_carry_projected_columns(db):
    """`*` means variables, not the result columns an earlier stage produced."""
    rows = db.execute("MATCH (n:P) WITH n, n.age AS age WITH * RETURN *")
    assert sorted(rows[0]) == ["age", "n"]


def test_star_after_an_aggregating_with(db):
    rows = db.execute("""
        MATCH (n:P)-[:KNOWS]->(m)
        WITH n, count(m) AS friends
        WITH *
        WHERE friends > 1
        RETURN n.name AS r
    """)
    assert _names(rows) == ["alice"]


def test_star_alongside_an_aggregate_groups_by_every_variable(db):
    """`*` contributes every bound variable to the implicit GROUP BY.

    `m` is in scope, so grouping is per (n, m) — two rows of one — not the
    single row of two that `WITH n, count(m)` produces.
    """
    with_star = db.execute("""
        MATCH (n:P)-[:KNOWS]->(m)
        WITH *, count(m) AS friends
        RETURN n.name AS r, friends
    """)
    assert sorted((row["r"], row["friends"]) for row in with_star) == [
        ("alice", 1),
        ("alice", 1),
    ]

    without_star = db.execute("""
        MATCH (n:P)-[:KNOWS]->(m)
        WITH n, count(m) AS friends
        RETURN n.name AS r, friends
    """)
    assert [(row["r"], row["friends"]) for row in without_star] == [("alice", 2)]


def test_multiplication_still_parses(db):
    """`*` is also the multiplication operator; the star form must not shadow it."""
    rows = db.execute("MATCH (n:P) RETURN n.age * 2 AS r ORDER BY r")
    assert [row["r"] for row in rows] == [40, 60, 80]

    # Ordered by the WITH alias, not the RETURN alias: after a WITH, ORDER BY
    # is applied before the final projection, so a name introduced by that
    # RETURN is not yet in scope. Separate pre-existing defect.
    rows = db.execute(
        "MATCH (n:P) WITH n.age * 2 AS doubled RETURN doubled AS r ORDER BY doubled"
    )
    assert [row["r"] for row in rows] == [40, 60, 80]
