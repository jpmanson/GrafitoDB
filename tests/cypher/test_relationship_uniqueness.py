"""Regression test: a MATCH pattern never traverses the same relationship twice."""

from grafito import GrafitoDatabase


def test_shared_neighbor_excludes_self():
    db = GrafitoDatabase(":memory:")
    db.execute(
        "CREATE (r:C {n: 'Rosario'}), (p:C {n: 'Parana'}), (x:R {n: 'rio'}), "
        "(r)-[:A]->(x), (p)-[:A]->(x)"
    )
    rows = db.execute("MATCH (:C {n: 'Rosario'})-[:A]->(:R)<-[:A]-(c:C) RETURN c.n AS c")
    assert rows == [{"c": "Parana"}]
    rows = db.execute("MATCH (a:C)-[:A]-(x)-[:A]-(b) RETURN a.n AS a, b.n AS b ORDER BY a")
    assert rows == [{"a": "Parana", "b": "Rosario"}, {"a": "Rosario", "b": "Parana"}]
    db.close()
