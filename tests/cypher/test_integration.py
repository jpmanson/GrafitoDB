"""Integration tests for Cypher query execution."""

import os
import pathlib
import pytest
from grafito import GrafitoDatabase
from grafito.cypher.exceptions import CypherExecutionError, CypherSyntaxError
from grafito.exceptions import ConstraintError


def _tests_res_uri(filename: str) -> str:
    base = pathlib.Path(__file__).resolve().parents[1] / "res"
    return (base / filename).as_uri()


class TestCypherCreate:
    """Test end-to-end CREATE queries."""

    def test_create_simple_node(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("CREATE (n:Person)")
        assert len(results) == 1
        assert 'n' in results[0]
        assert results[0]['n']['labels'] == ['Person']
        db.close()

    def test_create_node_with_properties(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        assert len(results) == 1
        node = results[0]['n']
        assert node['properties']['name'] == 'Alice'
        assert node['properties']['age'] == 30
        db.close()

    def test_create_node_with_list_and_map_properties(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("CREATE (n:Person {tags: ['a', 'b'], meta: {score: 1}})")
        assert len(results) == 1
        node = results[0]['n']
        assert node['properties']['tags'] == ['a', 'b']
        assert node['properties']['meta'] == {'score': 1}
        db.close()

    def test_create_anonymous_node(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("CREATE (:Person {name: 'Bob'})")
        assert len(results) == 1
        assert 'created' in results[0]
        db.close()

    def test_create_node_persists(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        results = db.execute("MATCH (n:Person) RETURN n")
        assert len(results) == 1
        assert results[0]['n']['properties']['name'] == 'Alice'
        db.close()

    def test_create_relationship_simple(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})")
        assert len(results) == 1
        result = results[0]
        assert 'a' in result
        assert 'b' in result
        assert 'r' in result
        assert result['a']['properties']['name'] == 'Alice'
        assert result['b']['properties']['name'] == 'Bob'
        assert result['r']['type'] == 'KNOWS'
        db.close()

    def test_create_relationship_with_properties(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            CREATE (a:Person {name: 'Alice'})-[r:KNOWS {since: 2020, strength: 0.9}]->(b:Person {name: 'Bob'})
        """)
        result = results[0]
        assert result['r']['properties']['since'] == 2020
        assert result['r']['properties']['strength'] == 0.9
        db.close()

    def test_create_relationship_persists(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})")

        # Verify nodes exist
        nodes = db.execute("MATCH (n:Person) RETURN n")
        assert len(nodes) == 2

        # Verify relationship exists
        rels = db.execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b")
        assert len(rels) == 1
        assert rels[0]['a']['properties']['name'] == 'Alice'
        assert rels[0]['b']['properties']['name'] == 'Bob'
        db.close()

    def test_create_relationship_anonymous_nodes(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("CREATE (:Person)-[r:KNOWS]->(:Person)")
        assert len(results) == 1
        assert 'r' in results[0]
        db.close()

    def test_create_relationship_multiple_labels(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            CREATE (a:Person:Employee {name: 'Alice'})-[r:WORKS_WITH]->(b:Person:Employee {name: 'Bob'})
        """)
        result = results[0]
        assert set(result['a']['labels']) == {'Person', 'Employee'}
        assert set(result['b']['labels']) == {'Person', 'Employee'}
        db.close()


class TestCypherMatch:
    """Test end-to-end MATCH queries."""

    def test_match_all_nodes(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")
        results = db.execute("MATCH (n) RETURN n")
        assert len(results) == 2
        db.close()

    def test_match_by_label(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Company {name: 'TechCorp'})")
        results = db.execute("MATCH (n:Person) RETURN n")
        assert len(results) == 1
        assert results[0]['n']['properties']['name'] == 'Alice'
        db.close()

    def test_match_with_properties(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        results = db.execute("MATCH (n:Person {age: 30}) RETURN n")
        assert len(results) == 1
        assert results[0]['n']['properties']['name'] == 'Alice'
        db.close()


class TestCypherWhere:
    """Test end-to-end MATCH with WHERE queries."""

    def test_where_simple_comparison(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        results = db.execute("MATCH (n:Person) WHERE n.age > 26 RETURN n.name")
        assert len(results) == 1
        assert results[0]['n.name'] == 'Alice'
        db.close()

    def test_where_equality(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', city: 'NYC'})")
        db.execute("CREATE (n:Person {name: 'Bob', city: 'LA'})")
        results = db.execute("MATCH (n:Person) WHERE n.city = 'NYC' RETURN n.name")
        assert len(results) == 1
        assert results[0]['n.name'] == 'Alice'
        db.close()

    def test_where_and(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30, city: 'NYC'})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25, city: 'NYC'})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35, city: 'LA'})")
        results = db.execute("""
            MATCH (n:Person)
            WHERE n.age > 26 AND n.city = 'NYC'
            RETURN n.name
        """)
        assert len(results) == 1
        assert results[0]['n.name'] == 'Alice'
        db.close()

    def test_where_or(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 20})")
        results = db.execute("""
            MATCH (n:Person)
            WHERE n.age > 28 OR n.age < 22
            RETURN n.name
        """)
        assert len(results) == 2  # Alice and Charlie
        names = {r['n.name'] for r in results}
        assert 'Alice' in names
        assert 'Charlie' in names
        db.close()

    def test_where_not(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', active: true})")
        db.execute("CREATE (n:Person {name: 'Bob', active: false})")
        results = db.execute("""
            MATCH (n:Person)
            WHERE NOT n.active
            RETURN n.name
        """)
        assert len(results) == 1
        assert results[0]['n.name'] == 'Bob'
        db.close()

    def test_where_is_null(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 30})")
        results = db.execute("""
            MATCH (n:Person)
            WHERE n.age IS NULL
            RETURN n.name
        """)
        assert results == [{'n.name': 'Alice'}]
        db.close()

    def test_where_is_not_null(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 30})")
        results = db.execute("""
            MATCH (n:Person)
            WHERE n.age IS NOT NULL
            RETURN n.name
        """)
        assert results == [{'n.name': 'Bob'}]
        db.close()

    def test_where_in_list(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")
        db.execute("CREATE (n:Person {name: 'Charlie'})")
        results = db.execute("""
            MATCH (n:Person)
            WHERE n.name IN ['Alice', 'Charlie']
            RETURN n.name
        """)
        names = {row['n.name'] for row in results}
        assert names == {'Alice', 'Charlie'}
        db.close()

    def test_where_any_predicate(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH [1,2,3] AS xs
            RETURN ANY(x IN xs WHERE x > 2) AS any
        """)
        assert results == [{'any': True}]
        db.close()

    def test_where_temporal_comparison(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            WHERE date('2024-01-01') < date('2024-01-02')
            RETURN date('2024-01-01') AS d
        """)
        assert results[0]['d'].isoformat() == '2024-01-01'
        db.close()

    def test_where_predicates_without_where(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH [true, false, false] AS xs
            RETURN ANY(x IN xs) AS any,
                   ALL(x IN xs) AS all,
                   NONE(x IN xs) AS none,
                   SINGLE(x IN xs) AS single
        """)
        assert results == [{
            'any': True,
            'all': False,
            'none': False,
            'single': True,
        }]
        db.close()

    def test_where_null_comparisons(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            WHERE date('2024-01-01') = null
            RETURN 1 AS ok
        """)
        assert results == []
        db.close()

    def test_where_in_with_nulls(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            WHERE 2 IN [1, null]
            RETURN 1 AS ok
        """)
        assert results == []
        results = db.execute("""
            WITH 1 AS dummy
            WHERE 1 IN [1, null]
            RETURN 1 AS ok
        """)
        assert results == [{'ok': 1}]
        db.close()

    def test_where_list_predicates_with_nulls(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH [null, false] AS xs
            RETURN ANY(x IN xs) AS any,
                   ALL(x IN xs) AS all,
                   NONE(x IN xs) AS none,
                   SINGLE(x IN xs) AS single
        """)
        assert results == [{
            'any': None,
            'all': False,
            'none': None,
            'single': None,
        }]
        db.close()

    def test_where_list_predicates_with_nulls_in_where(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH [null, true] AS xs
            RETURN ALL(x IN xs WHERE x) AS all,
                   NONE(x IN xs WHERE x) AS none
        """)
        assert results == [{'all': None, 'none': False}]
        db.close()

    def test_where_predicates_over_relationship_list_without_where(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2019})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2021})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*1..2]->(b:Person)
            RETURN ANY(rel IN r) AS any,
                   ALL(rel IN r) AS all,
                   NONE(rel IN r) AS none,
                   SINGLE(rel IN r) AS single
        """)
        assert results == [
            {'any': True, 'all': True, 'none': False, 'single': True},
            {'any': True, 'all': True, 'none': False, 'single': False},
        ]
        db.close()

    def test_where_predicates_over_list_variable(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH [1,2,3] AS xs
            RETURN ANY(x IN xs WHERE x = 2) AS any,
                   ALL(x IN xs WHERE x > 0) AS all,
                   NONE(x IN xs WHERE x < 0) AS none,
                   SINGLE(x IN xs WHERE x = 3) AS single
        """)
        assert results == [{
            'any': True,
            'all': True,
            'none': True,
            'single': True,
        }]
        db.close()

    def test_where_predicates_over_node_list_variable(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (b:Person {name: 'Bob', age: 20})")
        db.execute("CREATE (c:Person {name: 'Cara', age: 40})")

        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            MATCH (b:Person {name: 'Bob'})
            MATCH (c:Person {name: 'Cara'})
            WITH [a,b,c] AS people
            RETURN ANY(n IN people WHERE n.age > 35) AS any,
                   ALL(n IN people WHERE n.age >= 20) AS all,
                   NONE(n IN people WHERE n.age < 18) AS none,
                   SINGLE(n IN people WHERE n.age = 20) AS single
        """)
        assert results == [{
            'any': True,
            'all': True,
            'none': True,
            'single': True,
        }]
        db.close()

    def test_where_predicates_over_relationship_list_variable(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2019})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2021})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*1..2]->(b:Person)
            RETURN ANY(rel IN r WHERE rel.since >= 2021) AS any,
                   ALL(rel IN r WHERE rel.since >= 2019) AS all,
                   NONE(rel IN r WHERE rel.since < 2018) AS none,
                   SINGLE(rel IN r WHERE rel.since >= 2021) AS single
        """)
        assert results == [
            {
                'any': False,
                'all': True,
                'none': True,
                'single': False,
            },
            {
                'any': True,
                'all': True,
                'none': True,
                'single': True,
            },
        ]
        db.close()

    def test_where_predicates_over_unbounded_relationship_list(self):
        db = GrafitoDatabase(':memory:', cypher_max_hops=3)
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        d = db.create_node(labels=['Person'], properties={'name': 'D'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2018})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2020})
        db.create_relationship(c.id, d.id, 'KNOWS', {'since': 2022})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*..]->(b:Person)
            RETURN ANY(rel IN r WHERE rel.since >= 2022) AS any,
                   ALL(rel IN r WHERE rel.since >= 2018) AS all,
                   NONE(rel IN r WHERE rel.since < 2018) AS none
        """)
        assert results == [
            {'any': False, 'all': True, 'none': True},
            {'any': False, 'all': True, 'none': True},
            {'any': True, 'all': True, 'none': True},
        ]
        db.close()

    def test_where_predicates_over_unbounded_relationship_list_mixed_direction(self):
        db = GrafitoDatabase(':memory:', cypher_max_hops=3)
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2018})
        db.create_relationship(c.id, b.id, 'KNOWS', {'since': 2022})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*..]-(b:Person)
            RETURN ANY(rel IN r WHERE rel.since >= 2022) AS any,
                   ALL(rel IN r WHERE rel.since >= 2018) AS all,
                   NONE(rel IN r WHERE rel.since < 2018) AS none
        """)
        assert results == [
            {'any': False, 'all': True, 'none': True},
            {'any': True, 'all': True, 'none': True},
        ]
        db.close()

    def test_where_all_none_single_over_unbounded_relationship_list_mixed_direction(self):
        db = GrafitoDatabase(':memory:', cypher_max_hops=3)
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2018})
        db.create_relationship(c.id, b.id, 'KNOWS', {'since': 2022})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*..]-(b:Person)
            RETURN ALL(rel IN r WHERE rel.since >= 2018) AS all,
                   NONE(rel IN r WHERE rel.since < 2018) AS none,
                   SINGLE(rel IN r WHERE rel.since >= 2022) AS single
        """)
        assert results == [
            {'all': True, 'none': True, 'single': False},
            {'all': True, 'none': True, 'single': True},
        ]
        db.close()

    def test_where_any_predicate_over_node_list(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (b:Person {name: 'Bob', age: 20})")
        db.execute("CREATE (c:Person {name: 'Cara', age: 40})")

        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            MATCH (b:Person {name: 'Bob'})
            MATCH (c:Person {name: 'Cara'})
            WITH [a,b,c] AS people
            RETURN ANY(n IN people WHERE n.age > 35) AS any
        """)
        assert results == [{'any': True}]
        db.close()

    def test_where_all_predicate_over_node_list(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (b:Person {name: 'Bob', age: 20})")

        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            MATCH (b:Person {name: 'Bob'})
            WITH [a,b] AS people
            RETURN ALL(n IN people WHERE n.age >= 20) AS all
        """)
        assert results == [{'all': True}]
        db.close()

    def test_where_none_predicate_over_node_list(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (b:Person {name: 'Bob', age: 20})")

        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            MATCH (b:Person {name: 'Bob'})
            WITH [a,b] AS people
            RETURN NONE(n IN people WHERE n.age < 18) AS none
        """)
        assert results == [{'none': True}]
        db.close()

    def test_where_single_predicate_over_node_list(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (b:Person {name: 'Bob', age: 20})")
        db.execute("CREATE (c:Person {name: 'Cara', age: 40})")

        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            MATCH (b:Person {name: 'Bob'})
            MATCH (c:Person {name: 'Cara'})
            WITH [a,b,c] AS people
            RETURN SINGLE(n IN people WHERE n.age >= 40) AS single
        """)
        assert results == [{'single': True}]
        db.close()

    def test_where_any_predicate_over_relationship_list(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2019})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2021})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*1..2]->(b:Person)
            WHERE ANY(rel IN r WHERE rel.since >= 2021)
            RETURN b.name
        """)
        assert results == [{'b.name': 'C'}]
        db.close()

    def test_where_all_predicate_over_relationship_list(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2020})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2021})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*1..2]->(b:Person)
            WHERE ALL(rel IN r WHERE rel.since >= 2020)
            RETURN b.name
        """)
        assert results == [{'b.name': 'B'}, {'b.name': 'C'}]
        db.close()

    def test_where_none_predicate_over_relationship_list(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2018})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2021})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*1..2]->(b:Person)
            WHERE NONE(rel IN r WHERE rel.since < 2018)
            RETURN b.name
        """)
        assert results == [{'b.name': 'B'}, {'b.name': 'C'}]
        db.close()

    def test_where_single_predicate_over_relationship_list(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2020})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2020})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*1..2]->(b:Person)
            WHERE SINGLE(rel IN r WHERE rel.since >= 2020)
            RETURN b.name
        """)
        assert results == [{'b.name': 'B'}]
        db.close()


class TestCypherReturn:
    """Test end-to-end RETURN projections."""

    def test_return_variable(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        results = db.execute("MATCH (n:Person) RETURN n")
        assert len(results) == 1
        assert 'n' in results[0]
        assert results[0]['n']['properties']['name'] == 'Alice'
        db.close()

    def test_return_property(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        results = db.execute("MATCH (n:Person) RETURN n.name")
        assert len(results) == 1
        assert results[0] == {'n.name': 'Alice'}
        db.close()

    def test_return_multiple_properties(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30, city: 'NYC'})")
        results = db.execute("MATCH (n:Person) RETURN n.name, n.age")
        assert len(results) == 1
        assert results[0] == {'n.name': 'Alice', 'n.age': 30}
        db.close()

    def test_return_list_comprehension(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN [x IN [1,2,3] WHERE x > 1 | x] AS nums
        """)
        assert results == [{'nums': [2, 3]}]
        db.close()

    def test_return_apoc_load_jsonarray_local(self):
        db = GrafitoDatabase(':memory:')
        file_uri = _tests_res_uri("apoc_sample.json")
        results = db.execute(f"""
            CALL apoc.load.jsonArray('{file_uri}') YIELD value
            RETURN value.name AS name
            ORDER BY name
        """)
        assert results == [{'name': 'Arya'}, {'name': 'Jon'}]
        db.close()

    def test_return_apoc_load_json_array_local(self):
        db = GrafitoDatabase(':memory:')
        file_uri = _tests_res_uri("apoc_sample.json")
        results = db.execute(f"""
            CALL apoc.load.json('{file_uri}') YIELD value
            RETURN value.name AS name
            ORDER BY name
        """)
        assert results == [{'name': 'Arya'}, {'name': 'Jon'}]
        db.close()

    def test_return_apoc_load_json_object_local(self):
        db = GrafitoDatabase(':memory:')
        file_uri = _tests_res_uri("apoc_object.json")
        results = db.execute(f"""
            CALL apoc.load.json('{file_uri}') YIELD value
            RETURN value.name AS name, value.house AS house
        """)
        assert results == [{'name': 'Tyrion', 'house': 'Lannister'}]
        db.close()

    def test_return_list_comprehension_relationships(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2020})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2019})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[r:KNOWS*2..2]->(b:Person)
            RETURN [rel IN r WHERE rel.since >= 2020 | rel.since] AS since
        """)
        assert results == [{'since': [2020]}]
        db.close()

    def test_return_list_comprehension_multi_hop_relationships(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        acme = db.create_node(labels=['Company'], properties={'name': 'Acme'})
        beta = db.create_node(labels=['Company'], properties={'name': 'Beta'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, acme.id, 'WORKS_AT', {'since': 2020})
        db.create_relationship(acme.id, beta.id, 'WORKS_AT', {'since': 2015})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})-[:KNOWS]->(b:Person)
            MATCH (b:Person)-[r:WORKS_AT*1..2]->(c:Company {name: 'Beta'})
            RETURN [rel IN r WHERE rel.since >= 2020 | rel.since] AS since, c.name
        """)
        assert results == [{'since': [2020], 'c.name': 'Beta'}]
        db.close()

    def test_return_list_comprehension_nodes(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (b:Person {name: 'Bob', age: 20})")

        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            WITH a
            MATCH (b:Person {name: 'Bob'})
            WITH [a,b] AS nodes
            RETURN [n IN nodes WHERE n.age > 25 | n.name] AS names
        """)
        assert results == [{'names': ['Alice']}]
        db.close()

    def test_return_pattern_comprehension(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        b = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        c = db.create_node(labels=['Person'], properties={'name': 'Cara'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(a.id, c.id, 'KNOWS')

        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            RETURN [(a)-[:KNOWS]->(b) WHERE b.name = 'Bob' | b.name] AS friends
        """)
        assert results == [{'friends': ['Bob']}]
        db.close()

    def test_return_pattern_comprehension_multi_hop(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        b = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        acme = db.create_node(labels=['Company'], properties={'name': 'Acme'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, acme.id, 'WORKS_AT')

        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            RETURN [(a)-[:KNOWS]->(b)-[:WORKS_AT]->(c) WHERE c.name = 'Acme' | b.name] AS friends
        """)
        assert results == [{'friends': ['Bob']}]
        db.close()

    def test_return_pattern_comprehension_variable_length_filter(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2020})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2019})

        results = db.execute("""
            MATCH (a:Person {name: 'A'})
            RETURN [(a)-[r:KNOWS*1..2]->(b)
                    WHERE ALL(rel IN r WHERE rel.since >= 2020)
                    | b.name] AS names
        """)
        assert results == [{'names': ['B']}]
        db.close()

    def test_return_list_index_and_concat(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN [1,2,3][1] AS second, [1,2] + [3] AS combined
        """)
        assert results == [{'second': 2, 'combined': [1, 2, 3]}]
        db.close()

    def test_return_list_with_negative_literal(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN [-1.0, 0.5] AS values
        """)
        assert results == [{'values': [-1.0, 0.5]}]
        db.close()

    def test_return_list_with_exponent_literals(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN [1e-3, -2.5e2] AS values
        """)
        assert results == [{'values': [0.001, -250.0]}]
        db.close()


class TestCypherLoadCsv:
    """Test LOAD CSV ingestion."""

    def test_load_csv_with_headers_set_merge(self):
        import os
        import uuid

        base_dir = os.path.join(os.getcwd(), ".grafito", "tmp")
        os.makedirs(base_dir, exist_ok=True)
        csv_path = os.path.join(base_dir, f"products-{uuid.uuid4().hex}.csv")
        try:
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "productID,unitPrice,discontinued\n"
                    "1,3.5,0\n"
                    "2,4.0,1\n"
                )
            path = csv_path.replace("\\", "/")
            db = GrafitoDatabase(':memory:')

            results = db.execute(f"""
                LOAD CSV WITH HEADERS FROM "{path}" AS row
                MERGE (n:Product {{productID: row.productID}})
                SET n += row,
                    n.unitPrice = toFloat(row.unitPrice),
                    n.discontinued = (row.discontinued <> "0")
                RETURN n.productID, n.unitPrice, n.discontinued
                ORDER BY n.productID
            """)
            assert results == [
                {"n.productID": "1", "n.unitPrice": 3.5, "n.discontinued": False},
                {"n.productID": "2", "n.unitPrice": 4.0, "n.discontinued": True},
            ]
            db.close()
        finally:
            if os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                except PermissionError:
                    pass

    def test_load_csv_relationship_set_map(self):
        import os
        import uuid

        base_dir = os.path.join(os.getcwd(), ".grafito", "tmp")
        os.makedirs(base_dir, exist_ok=True)
        csv_path = os.path.join(base_dir, f"details-{uuid.uuid4().hex}.csv")
        try:
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "orderID,productID,quantity\n"
                    "10,1,5\n"
                )
            path = csv_path.replace("\\", "/")
            db = GrafitoDatabase(':memory:')
            db.execute("CREATE (:Product {productID: '1'})")
            db.execute("CREATE (:Order {orderID: '10'})")

            results = db.execute(f"""
                LOAD CSV WITH HEADERS FROM "{path}" AS row
                MATCH (p:Product {{productID: row.productID}}), (o:Order {{orderID: row.orderID}})
                MERGE (o)-[details:ORDERS]->(p)
                SET details = row,
                    details.quantity = toInteger(row.quantity)
                RETURN details.quantity AS quantity, details.orderID AS orderID
            """)
            assert results == [{"quantity": 5, "orderID": "10"}]
            db.close()
        finally:
            if os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                except PermissionError:
                    pass

    def test_return_list_functions(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN size([1,2,3]) AS count,
                   head([1,2,3]) AS first,
                   tail([1,2,3]) AS rest,
                   last([1,2,3]) AS last,
                   reverse([1,2,3]) AS reversed,
                   range(1,3) AS seq
        """)
        assert results == [{
            'count': 3,
            'first': 1,
            'rest': [2, 3],
            'last': 3,
            'reversed': [3, 2, 1],
            'seq': [1, 2, 3],
        }]
        db.close()

    def test_return_distinct(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Alice'})")
        results = db.execute("MATCH (n:Person) RETURN DISTINCT n.name")
        assert results == [{'n.name': 'Alice'}]
        db.close()

    def test_return_collect(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")
        results = db.execute("MATCH (n:Person) RETURN COLLECT(n.name) AS names")
        assert results == [{'names': ['Alice', 'Bob']}]
        db.close()

    def test_return_collect_distinct(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Alice'})")
        results = db.execute("MATCH (n:Person) RETURN COLLECT(DISTINCT n.name) AS names")
        assert results == [{'names': ['Alice']}]
        db.close()

    def test_return_count_distinct(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")
        results = db.execute("MATCH (n:Person) RETURN COUNT(DISTINCT n.name) AS count")
        assert results == [{'count': 2}]
        db.close()

    def test_return_list_slice(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH [1,2,3,4] AS xs
            RETURN xs[1..3] AS slice
        """)
        assert results == [{'slice': [2, 3]}]
        db.close()

    def test_return_case_expression(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 20})")
        results = db.execute("""
            MATCH (n:Person)
            RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END AS status
        """)
        assert len(results) == 1
        assert results[0]['status'] == 'adult'
        db.close()

    def test_return_temporal_functions(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN date('2024-01-02') AS d,
                   time('10:11:12') AS t,
                   datetime('2024-01-02T10:11:12') AS dt,
                   duration('P1DT2H') AS dur
        """)
        row = results[0]
        assert row['d'].isoformat() == '2024-01-02'
        assert row['t'].isoformat() == '10:11:12'
        assert row['dt'].isoformat() == '2024-01-02T10:11:12'
        assert row['dur'].total_seconds() == 93600
        db.close()

    def test_return_temporal_arithmetic(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN date('2024-01-01') + duration('P2D') AS d,
                   datetime('2024-01-01T00:00:00') - duration('PT1H') AS dt
        """)
        row = results[0]
        assert row['d'].isoformat() == '2024-01-03'
        assert row['dt'].isoformat() == '2023-12-31T23:00:00'
        db.close()

    def test_return_temporal_truncate_between(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN date.truncate('month', date('2024-02-20')) AS d,
                   duration.between(date('2024-01-01'), date('2024-01-03')) AS diff
        """)
        row = results[0]
        assert row['d'].isoformat() == '2024-02-01'
        assert row['diff'].total_seconds() == 172800
        db.close()

    def test_return_temporal_truncate_week(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN date.truncate('week', date('2024-02-14')) AS d
        """)
        assert results[0]['d'].isoformat() == '2024-02-12'
        db.close()

    def test_return_temporal_timezone_comparison(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN datetime('2024-01-01T00:00:00Z') = datetime('2024-01-01T01:00:00+01:00') AS eq,
                   datetime('2024-01-01 01:00:00+01:00') > datetime('2024-01-01T00:00:00Z') AS gt
        """)
        assert results == [{'eq': True, 'gt': False}]
        db.close()

    def test_return_map_literal_access(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH {a: 1, b: 2} AS m
            RETURN m.a AS a, m.b AS b
        """)
        assert results == [{'a': 1, 'b': 2}]
        db.close()

    def test_return_keys_values(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        results = db.execute("""
            MATCH (n:Person {name: 'Alice'})
            WITH n, {a: 1, b: 2} AS m
            RETURN keys(n) AS nkeys, values(n) AS nvalues,
                   keys(m) AS mkeys, values(m) AS mvalues
        """)
        row = results[0]
        assert sorted(row['nkeys']) == ['age', 'name']
        assert set(row['nvalues']) == {30, 'Alice'}
        assert row['mkeys'] == ['a', 'b']
        assert row['mvalues'] == [1, 2]
        db.close()

    def test_return_apoc_functions(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH [1,1,2] AS xs, {a: 1} AS m
            RETURN apoc.text.join(['a','b'], ',') AS joined,
                   apoc.text.split('a,b,c', ',') AS parts,
                   apoc.text.replace('a-b-c', '-', '_') AS replaced,
                   apoc.map.merge(m, {b: 2}) AS merged,
                   apoc.map.removeKey({a: 1, b: 2}, 'a') AS removed,
                   apoc.map.get(m, 'a', 9) AS value,
                   apoc.map.get(m, 'missing', 9) AS fallback,
                   apoc.coll.contains(xs, 2) AS has_two,
                   apoc.coll.toSet(xs) AS set
        """)
        assert results == [{
            'joined': 'a,b',
            'parts': ['a', 'b', 'c'],
            'replaced': 'a_b_c',
            'merged': {'a': 1, 'b': 2},
            'removed': {'b': 2},
            'value': 1,
            'fallback': 9,
            'has_two': True,
            'set': [1, 2],
        }]
        db.close()

    def test_return_apoc_invalid_types(self):
        db = GrafitoDatabase(':memory:')
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN keys(1) AS k")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN apoc.text.join('a', ',') AS joined")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN apoc.text.split('a', '[') AS parts")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN apoc.text.replace(1, '-', '_') AS out")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN apoc.map.merge(1, 2) AS merged")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN apoc.map.removeKey(1, 'a') AS removed")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN apoc.coll.contains(1, 2) AS has")
        db.close()

    def test_return_casting_functions(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN toInteger('42') AS i,
                   toFloat('3.5') AS f,
                   toString(123) AS s,
                   toInteger(null) AS n
        """)
        assert results == [{'i': 42, 'f': 3.5, 's': '123', 'n': None}]
        db.close()

    def test_return_string_functions(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN toUpper('hi') AS up,
                   toLower('HI') AS low,
                   trim('  hi  ') AS trimmed,
                   split('a,b,c', ',') AS parts,
                   substring('abcdef', 1, 3) AS sub,
                   substring('abcdef', 2) AS tail,
                   regex('abc', 'a.c') AS re1,
                   matches('abc', '^d') AS re2
        """)
        assert results == [{
            'up': 'HI',
            'low': 'hi',
            'trimmed': 'hi',
            'parts': ['a', 'b', 'c'],
            'sub': 'bcd',
            'tail': 'cdef',
            're1': True,
            're2': False,
        }]
        db.close()

    def test_return_string_functions_invalid_types(self):
        db = GrafitoDatabase(':memory:')
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN split(1, ',') AS parts")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN substring('abc', '1') AS sub")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN regex('abc', 1) AS ok")
        db.close()

    def test_return_string_functions_bounds(self):
        db = GrafitoDatabase(':memory:')
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN substring('abc', 0 - 1) AS sub")
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN substring('abc', 1, 0 - 2) AS sub")
        db.close()

    def test_return_string_functions_invalid_regex(self):
        db = GrafitoDatabase(':memory:')
        with pytest.raises(CypherExecutionError):
            db.execute("WITH 1 AS dummy RETURN regex('abc', '[') AS ok")
        db.close()

    def test_return_point_and_distance(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN point({x: 0, y: 3}) AS p,
                   distance(point({x: 0, y: 3}), point({x: 0, y: 0})) AS d
        """)
        assert results[0]['p'] == {'x': 0.0, 'y': 3.0}
        assert results[0]['d'] == 3.0
        db.close()

    def test_return_coalesce(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH 1 AS dummy
            RETURN coalesce(null, 'a', 'b') AS v,
                   coalesce(null, null) AS none
        """)
        assert results == [{'v': 'a', 'none': None}]
        db.close()

    def test_return_filter_extract_reduce(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            WITH [1,2,3] AS xs
            RETURN filter(x IN xs WHERE x > 1) AS filtered,
                   extract(x IN xs | x + 1) AS extracted,
                   reduce(acc = 0, x IN xs | acc + x) AS total
        """)
        assert results == [{'filtered': [2, 3], 'extracted': [2, 3, 4], 'total': 6}]
        db.close()

    def test_return_path_nodes_relationships(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        db.create_relationship(a.id, b.id, 'KNOWS')

        results = db.execute("""
            MATCH p=(a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})
            RETURN p, nodes(p) AS nodes, relationships(p) AS rels
        """)
        row = results[0]
        assert len(row['p']['nodes']) == 2
        assert len(row['p']['relationships']) == 1
        assert [node['properties']['name'] for node in row['nodes']] == ['A', 'B']
        assert row['rels'][0]['type'] == 'KNOWS'
        db.close()

    def test_return_case_multiple_when(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Bob', age: 16})")
        results = db.execute("""
            MATCH (n:Person)
            RETURN CASE
                WHEN n.age < 13 THEN 'child'
                WHEN n.age < 20 THEN 'teen'
                ELSE 'adult'
            END AS group
        """)
        assert len(results) == 1
        assert results[0]['group'] == 'teen'
        db.close()

    def test_return_case_no_else(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Cara', active: false})")
        results = db.execute("""
            MATCH (n:Person)
            RETURN CASE WHEN n.active THEN 'yes' END AS active
        """)
        assert len(results) == 1
        assert results[0]['active'] is None
        db.close()

    def test_return_case_simple(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Dana', status: 'active'})")
        results = db.execute("""
            MATCH (n:Person)
            RETURN CASE n.status WHEN 'active' THEN 1 ELSE 0 END AS score
        """)
        assert len(results) == 1
        assert results[0]['score'] == 1
        db.close()

    def test_return_case_aggregation(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 20})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 15})")
        results = db.execute("""
            MATCH (n:Person)
            RETURN SUM(CASE WHEN n.age >= 18 THEN 1 ELSE 0 END) AS adults
        """)
        assert len(results) == 1
        assert results[0]['adults'] == 1
        db.close()

    def test_return_case_order_by(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 20})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 15})")
        results = db.execute("""
            MATCH (n:Person)
            RETURN n.name, CASE WHEN n.age >= 18 THEN 1 ELSE 0 END AS adult
            ORDER BY adult DESC, n.name ASC
        """)
        assert [row['n.name'] for row in results] == ['Alice', 'Bob']
        db.close()


class TestCypherUnwind:
    """Test UNWIND clause."""

    def test_unwind_list(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            UNWIND [1,2,3] AS x
            RETURN x
        """)
        assert results == [{'x': 1}, {'x': 2}, {'x': 3}]
        db.close()

    def test_unwind_with_order_by(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            UNWIND [3,1,2] AS x
            WITH x ORDER BY x ASC
            RETURN x
        """)
        assert results == [{'x': 1}, {'x': 2}, {'x': 3}]
        db.close()

    def test_unwind_maps_with_order_by(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            UNWIND [{name:'Bob', age:25}, {name:'Alice', age:30}] AS p
            WITH p ORDER BY p.age DESC
            RETURN p.name, p.age
        """)
        assert results == [
            {'p.name': 'Alice', 'p.age': 30},
            {'p.name': 'Bob', 'p.age': 25},
        ]
        db.close()

    def test_unwind_with_aggregation(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            UNWIND [1,2,3] AS x
            WITH SUM(x) AS total
            RETURN total
        """)
        assert results == [{'total': 6}]
        db.close()

    def test_unwind_with_ordered_collect(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("""
            UNWIND [3,1,2] AS x
            WITH x ORDER BY x DESC
            WITH COLLECT(x) AS xs
            RETURN xs
        """)
        assert results == [{'xs': [3, 2, 1]}]
        db.close()


class TestCypherRelationships:
    """Test end-to-end relationship queries."""

    def test_match_outgoing_relationship(self):
        db = GrafitoDatabase(':memory:')
        # Create nodes using API
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        db.create_relationship(alice.id, bob.id, 'KNOWS', {'since': 2020})

        results = db.execute("""
            MATCH (a:Person)-[r:KNOWS]->(b:Person)
            RETURN a.name, b.name
        """)
        assert len(results) == 1
        assert results[0]['a.name'] == 'Alice'
        assert results[0]['b.name'] == 'Bob'
        db.close()

    def test_match_relationship_with_where(self):
        db = GrafitoDatabase(':memory:')
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice', 'age': 30})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob', 'age': 25})
        charlie = db.create_node(labels=['Person'], properties={'name': 'Charlie', 'age': 20})
        db.create_relationship(alice.id, bob.id, 'KNOWS')
        db.create_relationship(alice.id, charlie.id, 'KNOWS')

        results = db.execute("""
            MATCH (a:Person)-[r:KNOWS]->(b:Person)
            WHERE a.age > 25 AND b.age > 21 AND b.age < 26
            RETURN a.name, b.name
        """)
        assert len(results) == 1
        assert results[0]['a.name'] == 'Alice'
        assert results[0]['b.name'] == 'Bob'
        db.close()

    def test_match_incoming_relationship(self):
        db = GrafitoDatabase(':memory:')
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        db.create_relationship(bob.id, alice.id, 'KNOWS')

        results = db.execute("""
            MATCH (a:Person)<-[r:KNOWS]-(b:Person)
            RETURN a.name, b.name
        """)
        assert len(results) == 1
        assert results[0]['a.name'] == 'Alice'
        assert results[0]['b.name'] == 'Bob'
        db.close()


class TestCypherShortestPath:
    """Test shortestPath/allShortestPaths patterns."""

    def test_shortest_path(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')

        results = db.execute("""
            MATCH p=shortestPath((a:Person {name: 'A'})-[:KNOWS*1..2]->(c:Person {name: 'C'}))
            RETURN p
        """)
        assert len(results) == 1
        assert len(results[0]['p']['nodes']) == 3
        assert len(results[0]['p']['relationships']) == 2
        db.close()

    def test_all_shortest_paths(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        d = db.create_node(labels=['Person'], properties={'name': 'D'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')
        db.create_relationship(a.id, d.id, 'KNOWS')
        db.create_relationship(d.id, c.id, 'KNOWS')

        results = db.execute("""
            MATCH p=allShortestPaths((a:Person {name: 'A'})-[:KNOWS*1..2]->(c:Person {name: 'C'}))
            RETURN p
        """)
        assert len(results) == 2
        lengths = sorted(len(row['p']['relationships']) for row in results)
        assert lengths == [2, 2]
        db.close()

    def test_all_shortest_paths_unbounded_uses_max_hops(self):
        db = GrafitoDatabase(':memory:', cypher_max_hops=3)
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        d = db.create_node(labels=['Person'], properties={'name': 'D'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')
        db.create_relationship(c.id, d.id, 'KNOWS')

        results = db.execute("""
            MATCH p=allShortestPaths((a:Person {name: 'A'})-[:KNOWS*..]->(d:Person {name: 'D'}))
            RETURN p
        """)
        assert len(results) == 1
        assert len(results[0]['p']['relationships']) == 3
        db.close()

    def test_match_variable_length_relationships(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        d = db.create_node(labels=['Person'], properties={'name': 'D'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')
        db.create_relationship(c.id, d.id, 'KNOWS')

        results = db.execute("""
            MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)
            RETURN a.name, b.name
        """)
        pairs = {(row['a.name'], row['b.name']) for row in results}
        assert pairs == {
            ('A', 'B'),
            ('A', 'C'),
            ('A', 'D'),
            ('B', 'C'),
            ('B', 'D'),
            ('C', 'D'),
        }
        db.close()

    def test_return_variable_length_relationships(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')

        results = db.execute("""
            MATCH (a:Person)-[r:KNOWS*2]->(b:Person)
            RETURN r
        """)
        assert len(results) == 1
        rels = results[0]['r']
        assert isinstance(rels, list)
        assert len(rels) == 2
        assert rels[0]['type'] == 'KNOWS'
        assert rels[1]['type'] == 'KNOWS'
        db.close()

    def test_match_variable_length_unbounded_uses_default_max(self):
        db = GrafitoDatabase(':memory:', cypher_max_hops=2)
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        d = db.create_node(labels=['Person'], properties={'name': 'D'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')
        db.create_relationship(c.id, d.id, 'KNOWS')

        results = db.execute("""
            MATCH (a:Person)-[:KNOWS*..]->(b:Person)
            RETURN a.name, b.name
        """)
        pairs = {(row['a.name'], row['b.name']) for row in results}
        assert pairs == {
            ('A', 'B'),
            ('A', 'C'),
            ('B', 'C'),
            ('B', 'D'),
            ('C', 'D'),
        }
        db.close()

    def test_match_multi_hop_with_variable_length(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        company = db.create_node(labels=['Company'], properties={'name': 'Acme'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')
        db.create_relationship(c.id, company.id, 'WORKS_AT')

        results = db.execute("""
            MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)-[:WORKS_AT]->(c:Company)
            RETURN a.name, b.name, c.name
        """)
        rows = {(row['a.name'], row['b.name'], row['c.name']) for row in results}
        assert rows == {
            ('A', 'C', 'Acme'),
            ('B', 'C', 'Acme'),
        }
        db.close()

    def test_match_multi_hop_mixed_directions(self):
        db = GrafitoDatabase(':memory:')
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        company = db.create_node(labels=['Company'], properties={'name': 'Acme'})
        db.create_relationship(bob.id, alice.id, 'KNOWS')
        db.create_relationship(bob.id, company.id, 'WORKS_AT')

        results = db.execute("""
            MATCH (a:Person)<-[:KNOWS]-(b:Person)-[:WORKS_AT]->(c:Company)
            RETURN a.name, b.name, c.name
        """)
        assert results == [{'a.name': 'Alice', 'b.name': 'Bob', 'c.name': 'Acme'}]
        db.close()

    def test_match_multi_hop_relationship_variables(self):
        db = GrafitoDatabase(':memory:')
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        company = db.create_node(labels=['Company'], properties={'name': 'Acme'})
        db.create_relationship(alice.id, bob.id, 'KNOWS', {'since': 2020})
        db.create_relationship(bob.id, company.id, 'WORKS_AT', {'role': 'Engineer'})

        results = db.execute("""
            MATCH (a:Person)-[r1:KNOWS]->(b:Person)-[r2:WORKS_AT]->(c:Company)
            RETURN r1, r2
        """)
        assert len(results) == 1
        assert results[0]['r1']['type'] == 'KNOWS'
        assert results[0]['r1']['properties']['since'] == 2020
        assert results[0]['r2']['type'] == 'WORKS_AT'
        assert results[0]['r2']['properties']['role'] == 'Engineer'
        db.close()

    def test_match_variable_length_middle_with_where(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        company = db.create_node(labels=['Company'], properties={'name': 'Acme', 'size': 200})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')
        db.create_relationship(c.id, company.id, 'WORKS_AT')

        results = db.execute("""
            MATCH (a:Person)-[:KNOWS*1..2]->(b:Person)-[:WORKS_AT]->(c:Company)
            WHERE c.size > 100
            RETURN a.name, b.name, c.name
        """)
        rows = {(row['a.name'], row['b.name'], row['c.name']) for row in results}
        assert rows == {
            ('A', 'C', 'Acme'),
            ('B', 'C', 'Acme'),
        }
        db.close()

    def test_match_variable_length_with_rel_property_where(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        company = db.create_node(labels=['Company'], properties={'name': 'Acme'})
        db.create_relationship(a.id, b.id, 'KNOWS', {'since': 2020})
        db.create_relationship(b.id, c.id, 'KNOWS', {'since': 2019})
        db.create_relationship(c.id, company.id, 'WORKS_AT')

        results = db.execute("""
            MATCH (a:Person)-[r:KNOWS*1..2]->(b:Person)-[:WORKS_AT]->(c:Company)
            WHERE r.since = 2020
            RETURN a.name, b.name
        """)
        assert results == [{'a.name': 'A', 'b.name': 'C'}]
        db.close()

    def test_match_multi_hop_relationship_variable_and_list(self):
        db = GrafitoDatabase(':memory:')
        a = db.create_node(labels=['Person'], properties={'name': 'A'})
        b = db.create_node(labels=['Person'], properties={'name': 'B'})
        c = db.create_node(labels=['Person'], properties={'name': 'C'})
        company = db.create_node(labels=['Company'], properties={'name': 'Acme'})
        db.create_relationship(a.id, b.id, 'KNOWS')
        db.create_relationship(b.id, c.id, 'KNOWS')
        db.create_relationship(c.id, company.id, 'WORKS_AT', {'role': 'Engineer'})

        results = db.execute("""
            MATCH (a:Person)-[r1:KNOWS*1..2]->(b:Person)-[r2:WORKS_AT]->(c:Company)
            RETURN r1, r2
        """)
        assert len(results) == 2
        for row in results:
            assert isinstance(row['r1'], list)
            assert row['r2']['type'] == 'WORKS_AT'
        db.close()

    def test_match_bidirectional_relationship(self):
        db = GrafitoDatabase(':memory:')
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        db.create_relationship(alice.id, bob.id, 'KNOWS')

        results = db.execute("""
            MATCH (a:Person)-[r:KNOWS]-(b:Person)
            RETURN a.name, b.name
        """)
        # Should match both directions
        assert len(results) == 2
        db.close()


class TestCypherUnion:
    """Test UNION and UNION ALL queries."""

    def test_union_distinct(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (:Person {name: 'Alice'})")
        db.execute("CREATE (:Person {name: 'Bob'})")

        results = db.execute("""
            MATCH (n:Person) RETURN n.name
            UNION
            MATCH (n:Person {name: 'Alice'}) RETURN n.name
        """)
        names = [row['n.name'] for row in results]
        assert sorted(names) == ['Alice', 'Bob']
        db.close()

    def test_union_all(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (:Person {name: 'Alice'})")

        results = db.execute("""
            MATCH (n:Person {name: 'Alice'}) RETURN n.name
            UNION ALL
            MATCH (n:Person {name: 'Alice'}) RETURN n.name
        """)
        names = [row['n.name'] for row in results]
        assert names == ['Alice', 'Alice']
        db.close()


class TestCypherSubquery:
    """Test CALL { ... } subqueries."""

    def test_call_subquery_simple(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (:Person {name: 'Alice'})")
        db.execute("CREATE (:Person {name: 'Bob'})")

        results = db.execute("""
            CALL {
                MATCH (n:Person)
                RETURN n.name AS name
            }
        """)
        names = sorted(row['name'] for row in results)
        assert names == ['Alice', 'Bob']
        db.close()

    def test_call_subquery_with_with_passthrough(self):
        db = GrafitoDatabase(':memory:')
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        db.create_relationship(alice.id, bob.id, 'KNOWS')

        results = db.execute("""
            MATCH (n:Person {name: 'Alice'})
            WITH n
            CALL {
                WITH n
                MATCH (n)-[:KNOWS]->(m)
                RETURN m.name AS friend
            }
            WITH n, friend
            RETURN n.name, friend
        """)
        assert results == [{'n.name': 'Alice', 'friend': 'Bob'}]
        db.close()


class TestCypherComplexQueries:
    """Test complex end-to-end scenarios."""

    def test_social_network_scenario(self):
        db = GrafitoDatabase(':memory:')

        # Create social network
        db.execute("CREATE (n:Person {name: 'Alice', age: 30, city: 'NYC'})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25, city: 'LA'})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35, city: 'NYC'})")

        # Get all persons in NYC older than 28
        results = db.execute("""
            MATCH (n:Person)
            WHERE n.age > 28 AND n.city = 'NYC'
            RETURN n.name, n.age
        """)
        assert len(results) == 2
        names = {r['n.name'] for r in results}
        assert 'Alice' in names
        assert 'Charlie' in names
        db.close()

    def test_combined_create_and_query(self):
        db = GrafitoDatabase(':memory:')

        # Create multiple nodes
        db.execute("CREATE (n:Person {name: 'Alice', score: 95})")
        db.execute("CREATE (n:Person {name: 'Bob', score: 87})")
        db.execute("CREATE (n:Person {name: 'Charlie', score: 92})")

        # Query high scorers
        results = db.execute("""
            MATCH (n:Person)
            WHERE n.score >= 90
            RETURN n.name, n.score
        """)
        assert len(results) == 2
        names = {r['n.name'] for r in results}
        assert 'Alice' in names
        assert 'Charlie' in names
        db.close()


class TestCypherDelete:
    """Test end-to-end DELETE queries."""

    def test_delete_single_node(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")

        results = db.execute("MATCH (n:Person {name: 'Alice'}) DELETE n")
        assert results[0]['deleted_nodes'] == 1

        # Verify node is deleted
        remaining = db.execute("MATCH (n:Person) RETURN n")
        assert len(remaining) == 1
        assert remaining[0]['n']['properties']['name'] == 'Bob'
        db.close()

    def test_delete_multiple_nodes(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 20})")

        results = db.execute("MATCH (n:Person) WHERE n.age < 28 DELETE n")
        assert results[0]['deleted_nodes'] == 2

        # Verify only Alice remains
        remaining = db.execute("MATCH (n:Person) RETURN n")
        assert len(remaining) == 1
        assert remaining[0]['n']['properties']['name'] == 'Alice'
        db.close()

    def test_delete_relationship(self):
        db = GrafitoDatabase(':memory:')
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        rel_id = db.create_relationship(alice.id, bob.id, 'KNOWS')

        results = db.execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) DELETE r")
        assert results[0]['deleted_relationships'] == 1

        # Verify relationship is deleted but nodes remain
        nodes = db.execute("MATCH (n:Person) RETURN n")
        assert len(nodes) == 2

        rels = db.execute("MATCH (a)-[r:KNOWS]->(b) RETURN r")
        assert len(rels) == 0
        db.close()


class TestCypherSet:
    """Test end-to-end SET queries."""

    def test_set_single_property(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")

        db.execute("MATCH (n:Person {name: 'Alice'}) SET n.age = 31")

        results = db.execute("MATCH (n:Person) RETURN n.name, n.age")
        assert len(results) == 1
        assert results[0]['n.age'] == 31
        db.close()

    def test_set_multiple_properties(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")

        db.execute("MATCH (n:Person) SET n.age = 35, n.city = 'NYC'")

        results = db.execute("MATCH (n:Person) RETURN n")
        assert results[0]['n']['properties']['age'] == 35
        assert results[0]['n']['properties']['city'] == 'NYC'
        db.close()

    def test_set_with_where(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")

        db.execute("MATCH (n:Person) WHERE n.age < 28 SET n.status = 'young'")

        results = db.execute("MATCH (n:Person) WHERE n.status = 'young' RETURN n.name")
        assert len(results) == 1
        assert results[0]['n.name'] == 'Bob'
        db.close()


class TestCypherOrderBy:
    """Test end-to-end ORDER BY queries."""

    def test_order_by_ascending(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")

        results = db.execute("""
            MATCH (n:Person)
            RETURN n.name, n.age
            ORDER BY n.age ASC
        """)
        assert len(results) == 3
        assert results[0]['n.name'] == 'Bob'
        assert results[1]['n.name'] == 'Alice'
        assert results[2]['n.name'] == 'Charlie'
        db.close()

    def test_order_by_descending(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")

        results = db.execute("""
            MATCH (n:Person)
            RETURN n.name, n.age
            ORDER BY n.age DESC
        """)
        assert len(results) == 3
        assert results[0]['n.name'] == 'Charlie'
        assert results[1]['n.name'] == 'Alice'
        assert results[2]['n.name'] == 'Bob'
        db.close()

    def test_order_by_default_is_asc(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")

        results = db.execute("""
            MATCH (n:Person)
            RETURN n.name
            ORDER BY n.age
        """)
        assert results[0]['n.name'] == 'Bob'
        assert results[1]['n.name'] == 'Alice'
        db.close()

    def test_order_by_temporal(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'A'})")
        db.execute("CREATE (n:Person {name: 'B'})")

        results = db.execute("""
            MATCH (n:Person)
            RETURN n.name,
                   CASE WHEN n.name = 'A' THEN date('2024-01-02') ELSE date('2024-01-01') END AS d
            ORDER BY CASE WHEN n.name = 'A' THEN date('2024-01-02') ELSE date('2024-01-01') END ASC
        """)
        assert [row['n.name'] for row in results] == ['B', 'A']
        db.close()


class TestCypherLimit:
    """Test end-to-end LIMIT queries."""

    def test_limit_basic(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")
        db.execute("CREATE (n:Person {name: 'Charlie'})")

        results = db.execute("MATCH (n:Person) RETURN n LIMIT 2")
        assert len(results) == 2
        db.close()

    def test_limit_with_order_by(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")

        results = db.execute("""
            MATCH (n:Person)
            RETURN n.name, n.age
            ORDER BY n.age DESC
            LIMIT 2
        """)
        assert len(results) == 2
        assert results[0]['n.name'] == 'Charlie'
        assert results[1]['n.name'] == 'Alice'
        db.close()

    def test_limit_with_where(self):
        db = GrafitoDatabase(':memory:')
        for i in range(10):
            db.execute(f"CREATE (n:Person {{name: 'Person{i}', age: {20 + i}}})")

        results = db.execute("""
            MATCH (n:Person)
            WHERE n.age > 22
            RETURN n.name
            LIMIT 3
        """)
        assert len(results) == 3
        db.close()


class TestCypherSkip:
    """Test end-to-end SKIP queries."""

    def test_skip_basic(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")
        db.execute("CREATE (n:Person {name: 'Charlie'})")

        results = db.execute("MATCH (n:Person) RETURN n SKIP 1")
        assert len(results) == 2
        db.close()

    def test_skip_with_order_by(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")

        results = db.execute("""
            MATCH (n:Person)
            RETURN n.name, n.age
            ORDER BY n.age ASC
            SKIP 1
        """)
        assert len(results) == 2
        assert results[0]['n.name'] == 'Alice'
        assert results[1]['n.name'] == 'Charlie'
        db.close()

    def test_skip_with_order_by_and_limit(self):
        db = GrafitoDatabase(':memory:')
        for i in range(10):
            db.execute(f"CREATE (n:Person {{name: 'Person{i}', age: {20 + i}}})")

        # Pagination: skip first 3, take next 2
        results = db.execute("""
            MATCH (n:Person)
            RETURN n.name, n.age
            ORDER BY n.age ASC
            SKIP 3
            LIMIT 2
        """)
        assert len(results) == 2
        assert results[0]['n.age'] == 23
        assert results[1]['n.age'] == 24
        db.close()


class TestCypherOptionalMatch:
    """Test end-to-end OPTIONAL MATCH queries."""

    def test_optional_match_no_matches(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")

        # Try to match a non-existent label
        results = db.execute("OPTIONAL MATCH (n:Company) RETURN n")
        assert len(results) == 1
        assert results[0]['n'] is None
        db.close()

    def test_optional_match_with_matches(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")

        results = db.execute("OPTIONAL MATCH (n:Person) RETURN n")
        assert len(results) == 2
        assert results[0]['n'] is not None
        assert results[1]['n'] is not None
        db.close()

    def test_optional_match_with_where_no_matches(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")

        # WHERE filters out all matches
        results = db.execute("""
            OPTIONAL MATCH (n:Person)
            WHERE n.age > 50
            RETURN n.name
        """)
        assert len(results) == 1
        assert results[0]['n.name'] is None
        db.close()

    def test_optional_match_property_access(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")

        # Match non-existent label, access property
        results = db.execute("OPTIONAL MATCH (n:Company) RETURN n.name")
        assert len(results) == 1
        assert results[0]['n.name'] is None
        db.close()

    def test_optional_match_with_context_no_match(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice'})")
        results = db.execute("""
            MATCH (a:Person {name: 'Alice'})
            OPTIONAL MATCH (a)-[:KNOWS]->(b:Person)
            RETURN a.name, b.name
        """)
        assert results == [{'a.name': 'Alice', 'b.name': None}]
        db.close()

    def test_optional_match_with_where_keeps_row(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (a:Person {name: 'Alice'})")
        db.execute("CREATE (b:Person {name: 'Bob'})")
        db.execute("CREATE (a:Person {name: 'Cara'})")
        db.execute("CREATE (b:Person {name: 'Dan'})")
        db.execute("MATCH (a:Person {name: 'Alice'}) MATCH (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")

        results = db.execute("""
            MATCH (a:Person)
            OPTIONAL MATCH (a)-[:KNOWS]->(b:Person)
            WHERE b.name = 'Bob'
            RETURN a.name, b.name
            ORDER BY a.name
        """)
        assert results == [
            {'a.name': 'Alice', 'b.name': 'Bob'},
            {'a.name': 'Bob', 'b.name': None},
            {'a.name': 'Cara', 'b.name': None},
            {'a.name': 'Dan', 'b.name': None},
        ]
        db.close()


class TestCypherAggregations:
    """Test end-to-end aggregation queries."""

    def test_count_star(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")
        db.execute("CREATE (n:Person {name: 'Charlie'})")

        results = db.execute("MATCH (n:Person) RETURN COUNT(*)")
        assert len(results) == 1
        assert results[0]['COUNT(*)'] == 3
        db.close()

    def test_count_variable(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("CREATE (n:Person {name: 'Bob'})")

        results = db.execute("MATCH (n:Person) RETURN COUNT(n)")
        assert len(results) == 1
        assert results[0]['COUNT(n)'] == 2
        db.close()

    def test_sum(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")

        results = db.execute("MATCH (n:Person) RETURN SUM(n.age)")
        assert len(results) == 1
        assert results[0]['SUM(n.age)'] == 90
        db.close()

    def test_avg(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 20})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 40})")

        results = db.execute("MATCH (n:Person) RETURN AVG(n.age)")
        assert len(results) == 1
        assert results[0]['AVG(n.age)'] == 30.0
        db.close()

    def test_min(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")

        results = db.execute("MATCH (n:Person) RETURN MIN(n.age)")
        assert len(results) == 1
        assert results[0]['MIN(n.age)'] == 25
        db.close()

    def test_max(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")

        results = db.execute("MATCH (n:Person) RETURN MAX(n.age)")
        assert len(results) == 1
        assert results[0]['MAX(n.age)'] == 35
        db.close()

    def test_count_with_where(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")

        results = db.execute("""
            MATCH (n:Person)
            WHERE n.age > 26
            RETURN COUNT(*)
        """)
        assert len(results) == 1
        assert results[0]['COUNT(*)'] == 2
        db.close()

    def test_aggregation_empty_result(self):
        db = GrafitoDatabase(':memory:')
        results = db.execute("MATCH (n:Person) RETURN COUNT(*)")
        assert len(results) == 1
        assert results[0]['COUNT(*)'] == 0
        db.close()

    def test_sum_with_null_values(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob'})")  # No age

        results = db.execute("MATCH (n:Person) RETURN SUM(n.age)")
        assert len(results) == 1
        # Should only sum non-null values
        assert results[0]['SUM(n.age)'] == 30
        db.close()

    def test_stddev(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {score: 1})")
        db.execute("CREATE (n:Person {score: 2})")
        db.execute("CREATE (n:Person {score: 3})")
        results = db.execute("MATCH (n:Person) RETURN stdDev(n.score) AS s")
        assert results[0]['s'] == pytest.approx(0.816496, rel=1e-6)
        db.close()

    def test_percentile_cont(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {score: 1})")
        db.execute("CREATE (n:Person {score: 2})")
        db.execute("CREATE (n:Person {score: 3})")
        db.execute("CREATE (n:Person {score: 4})")
        results = db.execute("MATCH (n:Person) RETURN percentileCont(n.score, 0.5) AS p")
        assert results == [{'p': 2.5}]
        db.close()

    def test_percentile_cont_missing_percentile(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {score: 1})")
        with pytest.raises(CypherExecutionError):
            db.execute("MATCH (n:Person) RETURN percentileCont(n.score) AS p")
        db.close()

    def test_count_missing_argument(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        with pytest.raises(CypherExecutionError):
            db.execute("MATCH (n:Person) RETURN COUNT() AS c")
        db.close()

    def test_sum_star_not_supported(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {age: 1})")
        with pytest.raises(CypherExecutionError):
            db.execute("MATCH (n:Person) RETURN SUM(*) AS total")
        db.close()


class TestCypherMerge:
    """Test end-to-end MERGE queries."""

    def test_merge_creates_when_not_exists(self):
        db = GrafitoDatabase(':memory:')

        # MERGE when pattern doesn't exist - should CREATE
        results = db.execute("MERGE (n:Person {email: 'alice@example.com'})")
        assert len(results) == 1
        assert 'n' in results[0]

        # Verify it was created
        results = db.execute("MATCH (n:Person) RETURN COUNT(*)")
        assert results[0]['COUNT(*)'] == 1
        db.close()

    def test_merge_finds_when_exists(self):
        db = GrafitoDatabase(':memory:')

        # Create a node
        db.execute("CREATE (n:Person {email: 'alice@example.com', name: 'Alice'})")

        # MERGE when pattern exists - should find it (not create duplicate)
        results = db.execute("MERGE (n:Person {email: 'alice@example.com'})")
        assert len(results) == 1

        # Verify only one node exists
        results = db.execute("MATCH (n:Person) RETURN COUNT(*)")
        assert results[0]['COUNT(*)'] == 1
        db.close()

    def test_merge_on_create_set(self):
        db = GrafitoDatabase(':memory:')

        # MERGE with ON CREATE SET
        results = db.execute("""
            MERGE (n:Person {email: 'bob@example.com'})
            ON CREATE SET n.created = 2024
        """)
        assert len(results) == 1

        # Verify ON CREATE SET was executed
        results = db.execute("MATCH (n:Person {email: 'bob@example.com'}) RETURN n.created")
        assert results[0]['n.created'] == 2024
        db.close()

    def test_merge_on_match_set(self):
        db = GrafitoDatabase(':memory:')

        # Create a node
        db.execute("CREATE (n:Person {email: 'charlie@example.com', visits: 1})")

        # MERGE with ON MATCH SET - should update
        results = db.execute("""
            MERGE (n:Person {email: 'charlie@example.com'})
            ON MATCH SET n.visits = 2
        """)
        assert len(results) == 1

        # Verify ON MATCH SET was executed
        results = db.execute("MATCH (n:Person {email: 'charlie@example.com'}) RETURN n.visits")
        assert results[0]['n.visits'] == 2
        db.close()

    def test_merge_on_create_and_on_match(self):
        db = GrafitoDatabase(':memory:')

        # First MERGE - should trigger ON CREATE
        results = db.execute("""
            MERGE (n:Person {email: 'dave@example.com'})
            ON CREATE SET n.created = 2024
            ON MATCH SET n.lastSeen = 2025
        """)

        results = db.execute("MATCH (n:Person {email: 'dave@example.com'}) RETURN n.created, n.lastSeen")
        assert results[0]['n.created'] == 2024
        assert results[0]['n.lastSeen'] is None  # ON MATCH not triggered

        # Second MERGE - should trigger ON MATCH
        db.execute("""
            MERGE (n:Person {email: 'dave@example.com'})
            ON CREATE SET n.created = 9999
            ON MATCH SET n.lastSeen = 2025
        """)

        results = db.execute("MATCH (n:Person {email: 'dave@example.com'}) RETURN n.created, n.lastSeen")
        assert results[0]['n.created'] == 2024  # Not changed (ON CREATE not triggered)
        assert results[0]['n.lastSeen'] == 2025  # Updated by ON MATCH

        db.close()

    def test_merge_with_list_and_map_properties(self):
        db = GrafitoDatabase(':memory:')
        db.execute("""
            MERGE (n:Person {tags: ['a', 'b'], meta: {score: 1}})
        """)
        results = db.execute("""
            MATCH (n:Person {tags: ['a', 'b'], meta: {score: 1}})
            RETURN n
        """)
        node = results[0]['n']
        assert node['properties']['tags'] == ['a', 'b']
        assert node['properties']['meta'] == {'score': 1}
        db.close()

    def test_merge_prevents_duplicates(self):
        db = GrafitoDatabase(':memory:')

        # Multiple MERGE calls with same pattern
        db.execute("MERGE (n:User {username: 'alice'})")
        db.execute("MERGE (n:User {username: 'alice'})")
        db.execute("MERGE (n:User {username: 'alice'})")

        # Should only have one node
        results = db.execute("MATCH (n:User) RETURN COUNT(*)")
        assert results[0]['COUNT(*)'] == 1
        db.close()


class TestCypherWith:
    """Test WITH clause (query chaining/pipelines)."""

    def test_with_simple_projection(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")

        # Use WITH to project specific properties
        results = db.execute("""
            MATCH (n:Person)
            WITH n.name AS name, n.age AS age
            RETURN name, age
        """)

        assert len(results) == 2
        assert 'name' in results[0]
        assert 'age' in results[0]
        db.close()

    def test_with_filter(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")

        # Use WITH to filter intermediate results
        results = db.execute("""
            MATCH (n:Person)
            WITH n
            WHERE n.age > 27
            RETURN n.name
        """)

        assert len(results) == 2
        names = [r['n.name'] for r in results]
        assert 'Alice' in names
        assert 'Charlie' in names
        assert 'Bob' not in names
        db.close()

    def test_with_order_by_limit(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")

        # Use WITH to sort and limit
        results = db.execute("""
            MATCH (n:Person)
            WITH n
            ORDER BY n.age DESC
            LIMIT 2
            RETURN n.name, n.age
        """)

        assert len(results) == 2
        assert results[0]['n.name'] == 'Charlie'
        assert results[0]['n.age'] == 35
        assert results[1]['n.name'] == 'Alice'
        assert results[1]['n.age'] == 30
        db.close()

    def test_with_aggregation(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")

        # Use WITH with aggregation
        results = db.execute("""
            MATCH (n:Person)
            WITH COUNT(n) AS total, AVG(n.age) AS avgAge
            RETURN total, avgAge
        """)

        assert len(results) == 1
        assert results[0]['total'] == 3
        assert results[0]['avgAge'] == 30.0
        db.close()

    def test_with_and_final_return(self):
        db = GrafitoDatabase(':memory:')

        # Create some data
        db.execute("""
            CREATE (a:Person {name: 'Alice', city: 'NYC', age: 30}),
                   (b:Person {name: 'Bob', city: 'LA', age: 25}),
                   (c:Person {name: 'Charlie', city: 'NYC', age: 35})
        """)

        # Use WITH to filter, then return specific fields
        results = db.execute("""
            MATCH (p:Person)
            WITH p
            WHERE p.city = 'NYC'
            RETURN p.name, p.age
        """)

        assert len(results) == 2
        names = [r['p.name'] for r in results]
        assert 'Alice' in names
        assert 'Charlie' in names
        assert 'Bob' not in names
        db.close()

    def test_with_skip(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25})")
        db.execute("CREATE (n:Person {name: 'Charlie', age: 35})")

        # Use WITH with SKIP
        results = db.execute("""
            MATCH (n:Person)
            WITH n
            ORDER BY n.age
            SKIP 1
            RETURN n.name
        """)

        assert len(results) == 2
        names = [r['n.name'] for r in results]
        assert 'Alice' in names
        assert 'Charlie' in names
        assert 'Bob' not in names  # Skipped (youngest)
        db.close()


class TestCypherRemove:
    """Test REMOVE clause (remove properties and labels)."""

    def test_remove_property(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30, city: 'NYC'})")

        # Remove a property
        db.execute("MATCH (n:Person) REMOVE n.age")

        # Verify property was removed
        results = db.execute("MATCH (n:Person) RETURN n.name, n.age, n.city")
        assert len(results) == 1
        assert results[0]['n.name'] == 'Alice'
        assert results[0]['n.age'] is None  # Property removed
        assert results[0]['n.city'] == 'NYC'
        db.close()

    def test_remove_multiple_properties(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Bob', age: 25, city: 'LA', email: 'bob@example.com'})")

        # Remove multiple properties
        db.execute("MATCH (n:Person) REMOVE n.age, n.email")

        # Verify properties were removed
        results = db.execute("MATCH (n:Person) RETURN n.name, n.age, n.city, n.email")
        assert results[0]['n.name'] == 'Bob'
        assert results[0]['n.age'] is None
        assert results[0]['n.city'] == 'LA'
        assert results[0]['n.email'] is None
        db.close()

    def test_remove_nonexistent_property(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Charlie'})")

        # Remove property that doesn't exist (should not error)
        db.execute("MATCH (n:Person) REMOVE n.age")

        # Verify node still exists
        results = db.execute("MATCH (n:Person) RETURN n.name")
        assert len(results) == 1
        assert results[0]['n.name'] == 'Charlie'
        db.close()

    def test_remove_label(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person:Employee {name: 'Alice'})")

        # Remove one label
        db.execute("MATCH (n:Person:Employee) REMOVE n:Employee")

        # Verify label was removed
        results = db.execute("MATCH (n:Person) RETURN n")
        assert len(results) == 1
        assert 'Person' in results[0]['n']['labels']
        assert 'Employee' not in results[0]['n']['labels']
        db.close()

    def test_remove_label_keeps_node(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person:Manager {name: 'Bob'})")

        # Remove label
        db.execute("MATCH (n) REMOVE n:Manager")

        # Node should still exist with Person label
        results = db.execute("MATCH (n:Person) RETURN n.name")
        assert len(results) == 1
        assert results[0]['n.name'] == 'Bob'
        db.close()

    def test_remove_nonexistent_label(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Charlie'})")

        # Remove label that doesn't exist (should not error)
        db.execute("MATCH (n:Person) REMOVE n:Employee")

        # Verify node still exists with original label
        results = db.execute("MATCH (n:Person) RETURN n")
        assert len(results) == 1
        assert results[0]['n']['labels'] == ['Person']
        db.close()

    def test_remove_with_where(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30, temp: true})")
        db.execute("CREATE (n:Person {name: 'Bob', age: 25, temp: false})")

        # Remove property only from nodes matching WHERE condition
        db.execute("MATCH (n:Person) WHERE n.age > 27 REMOVE n.temp")

        # Verify only Alice's temp was removed
        results = db.execute("MATCH (n:Person) RETURN n.name, n.temp ORDER BY n.name")
        assert results[0]['n.name'] == 'Alice'
        assert results[0]['n.temp'] is None  # Removed
        assert results[1]['n.name'] == 'Bob'
        assert results[1]['n.temp'] is False  # Not removed
        db.close()

    def test_remove_and_return(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice', age: 30})")

        # Remove property and return in same query
        results = db.execute("MATCH (n:Person) REMOVE n.age RETURN n.name, n.age")

        assert len(results) == 1
        assert results[0]['n.name'] == 'Alice'
        assert results[0]['n.age'] is None
        db.close()


class TestCypherIndexes:
    """Test Cypher index management."""

    def test_create_show_drop_indexes(self):
        db = GrafitoDatabase(':memory:')
        result = db.execute("CREATE INDEX FOR NODE :Person(name)")
        assert result and result[0]['name'].startswith('idx_node_person_name')
        index_name = result[0]['name']

        rel_result = db.execute("CREATE INDEX FOR RELATIONSHIP :KNOWS(since)")
        rel_name = rel_result[0]['name']

        indexes = db.execute("SHOW INDEXES")
        assert any(
            idx['name'] == index_name
            and idx['entity'] == 'node'
            and idx['label_or_type'] == 'Person'
            and idx['property'] == 'name'
            for idx in indexes
        )
        assert any(
            idx['name'] == rel_name
            and idx['entity'] == 'relationship'
            and idx['label_or_type'] == 'KNOWS'
            and idx['property'] == 'since'
            for idx in indexes
        )

        db.execute(f"DROP INDEX {index_name}")
        indexes = db.execute("SHOW INDEXES")
        assert all(idx['name'] != index_name for idx in indexes)
        db.close()

    def test_create_unique_index(self):
        db = GrafitoDatabase(':memory:')
        result = db.execute("CREATE UNIQUE INDEX FOR NODE :Person(email)")
        assert result and result[0]['name'].startswith('idx_node_person_email')
        indexes = db.execute("SHOW INDEXES WHERE unique = true")
        assert any(idx['property'] == 'email' and idx['unique'] is True for idx in indexes)
        db.close()

    def test_show_indexes_where(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE INDEX FOR NODE :Person(name)")
        db.execute("CREATE INDEX FOR RELATIONSHIP :KNOWS(since)")
        results = db.execute("SHOW INDEXES WHERE entity = 'node'")
        assert all(idx['entity'] == 'node' for idx in results)
        db.close()

    def test_create_index_if_not_exists(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE INDEX IF NOT EXISTS FOR NODE :Person(name)")
        db.execute("CREATE INDEX IF NOT EXISTS FOR NODE :Person(name)")
        indexes = db.execute("SHOW INDEXES")
        matches = [idx for idx in indexes if idx['entity'] == 'node' and idx['label_or_type'] == 'Person']
        assert len(matches) == 1
        db.close()

    def test_create_unique_index_if_not_exists(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS FOR NODE :Person(email)")
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS FOR NODE :Person(email)")
        indexes = db.execute("SHOW INDEXES WHERE unique = true")
        matches = [idx for idx in indexes if idx['property'] == 'email' and idx['unique'] is True]
        assert len(matches) == 1
        db.close()

    def test_create_index_neo4j_node_syntax(self):
        db = GrafitoDatabase(':memory:')
        result = db.execute("CREATE INDEX FOR (n:Person) ON (n.name)")
        assert result and result[0]['name'].startswith('idx_node_person_name')
        db.close()

    def test_create_index_neo4j_node_without_var(self):
        db = GrafitoDatabase(':memory:')
        result = db.execute("CREATE INDEX FOR (:Person) ON (n.name)")
        assert result and result[0]['name'].startswith('idx_node_person_name')
        db.close()

    def test_create_index_neo4j_relationship_syntax(self):
        db = GrafitoDatabase(':memory:')
        result = db.execute("CREATE INDEX FOR ()-[r:KNOWS]-() ON (r.since)")
        assert result and result[0]['name'].startswith('idx_relationship_knows_since')
        db.close()

    def test_create_index_neo4j_relationship_without_var(self):
        db = GrafitoDatabase(':memory:')
        result = db.execute("CREATE INDEX FOR ()-[:KNOWS]-() ON (r.since)")
        assert result and result[0]['name'].startswith('idx_relationship_knows_since')
        db.close()

    def test_drop_index_if_exists(self):
        db = GrafitoDatabase(':memory:')
        db.execute("DROP INDEX IF EXISTS idx_missing")
        db.close()

    def test_create_index_invalid_identifier(self):
        db = GrafitoDatabase(':memory:')
        with pytest.raises(CypherSyntaxError):
            db.execute("CREATE INDEX FOR NODE :Person(bad-name)")
        db.close()


class TestCypherConstraints:
    """Test Cypher constraints."""

    def test_create_unique_constraint(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR NODE :Person REQUIRE email IS UNIQUE")
        db.execute("CREATE (n:Person {email: 'a@example.com'})")
        with pytest.raises(ConstraintError):
            db.execute("CREATE (n:Person {email: 'a@example.com'})")
        db.close()

    def test_create_existence_constraint(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR NODE :Person REQUIRE name IS NOT NULL")
        with pytest.raises(ConstraintError):
            db.execute("CREATE (n:Person {age: 10})")
        db.close()

    def test_create_type_constraint(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR NODE :Person REQUIRE age IS INTEGER")
        with pytest.raises(ConstraintError):
            db.execute("CREATE (n:Person {age: 'old'})")
        db.close()

    def test_show_constraints_where(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR NODE :Person REQUIRE email IS UNIQUE")
        results = db.execute("SHOW CONSTRAINTS WHERE entity = 'node'")
        assert all(item['entity'] == 'node' for item in results)
        db.close()

    def test_drop_constraint_if_exists(self):
        db = GrafitoDatabase(':memory:')
        db.execute("DROP CONSTRAINT IF EXISTS missing_constraint")
        db.close()

    def test_named_constraint_create_drop(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT person_email_unique FOR NODE :Person REQUIRE email IS UNIQUE")
        results = db.execute("SHOW CONSTRAINTS WHERE name = 'person_email_unique'")
        assert len(results) == 1
        db.execute("DROP CONSTRAINT person_email_unique")
        results = db.execute("SHOW CONSTRAINTS WHERE name = 'person_email_unique'")
        assert results == []
        db.close()

    def test_create_constraint_on_style(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR (n:Person) ON (n.email) IS UNIQUE")
        db.execute("CREATE (n:Person {email: 'a@example.com'})")
        with pytest.raises(ConstraintError):
            db.execute("CREATE (n:Person {email: 'a@example.com'})")
        db.close()

    def test_create_constraint_on_style_relationship(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() ON (r.since) IS INTEGER")
        with pytest.raises(ConstraintError):
            db.execute("CREATE (a:Person)-[:KNOWS {since: 'old'}]->(b:Person)")
        db.close()


class TestCypherForeach:
    """Test FOREACH updates."""

    def test_foreach_set(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE (n:Person {name: 'Alice'})")
        db.execute("""
            MATCH (n:Person {name: 'Alice'})
            WITH n
            FOREACH (x IN [1,2,3] | SET n.age = x)
            RETURN n.age
        """)
        results = db.execute("MATCH (n:Person {name: 'Alice'}) RETURN n.age")
        assert results == [{'n.age': 3}]
        db.close()

    def test_foreach_create(self):
        db = GrafitoDatabase(':memory:')
        db.execute("FOREACH (x IN [1,2,3] | CREATE (:Tag))")
        results = db.execute("MATCH (n:Tag) RETURN n")
        assert len(results) == 3
        db.close()

    def test_create_relationship_unique_constraint(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR RELATIONSHIP :KNOWS REQUIRE since IS UNIQUE")
        db.execute("CREATE (a:Person {name: 'A'})-[:KNOWS {since: 2020}]->(b:Person {name: 'B'})")
        with pytest.raises(ConstraintError):
            db.execute("CREATE (c:Person {name: 'C'})-[:KNOWS {since: 2020}]->(d:Person {name: 'D'})")
        db.close()

    def test_create_relationship_existence_constraint(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR RELATIONSHIP :KNOWS REQUIRE since IS NOT NULL")
        with pytest.raises(ConstraintError):
            db.execute("CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})")
        db.close()

    def test_create_relationship_type_constraint(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR RELATIONSHIP :KNOWS REQUIRE since IS INTEGER")
        with pytest.raises(ConstraintError):
            db.execute("CREATE (a:Person {name: 'A'})-[:KNOWS {since: 'old'}]->(b:Person {name: 'B'})")
        db.close()

    def test_show_constraints_where_relationship(self):
        db = GrafitoDatabase(':memory:')
        db.execute("CREATE CONSTRAINT FOR RELATIONSHIP :KNOWS REQUIRE since IS INTEGER")
        results = db.execute("SHOW CONSTRAINTS WHERE entity = 'relationship'")
        assert all(item['entity'] == 'relationship' for item in results)
        db.close()
