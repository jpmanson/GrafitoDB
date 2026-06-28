"""Example demonstrating Cypher query language support in Grafito.

This example shows how to use both the programmatic API and Cypher queries
interchangeably with Grafito.
"""

from grafito import GrafitoDatabase


def main():
    # Initialize database
    db = GrafitoDatabase(':memory:')

    print("=" * 60)
    print("Grafito Cypher Query Language Demo")
    print("=" * 60)
    print()

    # ========================================================================
    # Example 1: CREATE nodes with Cypher
    # ========================================================================
    print("1. Creating nodes with Cypher:")
    print("-" * 60)

    db.execute("CREATE (n:Person {name: 'Alice', age: 30, city: 'NYC'})")
    db.execute("CREATE (n:Person {name: 'Bob', age: 25, city: 'LA'})")
    db.execute("CREATE (n:Person {name: 'Charlie', age: 35, city: 'NYC'})")

    print("[OK] Created 3 Person nodes using Cypher")
    print()

    # ========================================================================
    # Example 2: MATCH with labels
    # ========================================================================
    print("2. Querying all persons:")
    print("-" * 60)

    results = db.execute("MATCH (n:Person) RETURN n.name, n.age, n.city")
    for result in results:
        print(f"  {result['n.name']}, age {result['n.age']}, lives in {result['n.city']}")
    print()

    # ========================================================================
    # Example 3: MATCH with WHERE clause
    # ========================================================================
    print("3. Finding persons older than 28 in NYC:")
    print("-" * 60)

    results = db.execute("""
        MATCH (n:Person)
        WHERE n.age > 28 AND n.city = 'NYC'
        RETURN n.name, n.age
    """)
    for result in results:
        print(f"  {result['n.name']}, age {result['n.age']}")
    print()

    # ========================================================================
    # Example 4: Complex WHERE with OR and NOT
    # ========================================================================
    print("4. Complex WHERE query (age > 32 OR city = 'LA'):")
    print("-" * 60)

    results = db.execute("""
        MATCH (n:Person)
        WHERE n.age > 32 OR n.city = 'LA'
        RETURN n.name
    """)
    for result in results:
        print(f"  {result['n.name']}")
    print()

    # ========================================================================
    # Example 5: Hybrid approach - API + Cypher
    # ========================================================================
    print("5. Hybrid approach - creating relationships with API:")
    print("-" * 60)

    # Get nodes using Cypher
    alice_result = db.execute("MATCH (n:Person {name: 'Alice'}) RETURN n")
    bob_result = db.execute("MATCH (n:Person {name: 'Bob'}) RETURN n")

    # Extract IDs (note: in real usage you'd use the programmatic API for this)
    alice_id = alice_result[0]['n']['id']
    bob_id = bob_result[0]['n']['id']

    # Create relationships using programmatic API
    db.create_relationship(alice_id, bob_id, 'KNOWS', {'since': 2020})

    print(f"[OK] Created KNOWS relationship: Alice -> Bob")
    print()

    # ========================================================================
    # Example 6: Querying relationships with Cypher
    # ========================================================================
    print("6. Querying relationships:")
    print("-" * 60)

    results = db.execute("""
        MATCH (a:Person)-[r:KNOWS]->(b:Person)
        RETURN a.name, b.name
    """)
    for result in results:
        print(f"  {result['a.name']} knows {result['b.name']}")
    print()

    # ========================================================================
    # Example 7: Relationship pattern with WHERE
    # ========================================================================
    print("7. Complex relationship query with WHERE:")
    print("-" * 60)

    # Add more relationships
    charlie_result = db.execute("MATCH (n:Person {name: 'Charlie'}) RETURN n")
    charlie_id = charlie_result[0]['n']['id']
    db.create_relationship(alice_id, charlie_id, 'KNOWS', {'since': 2018})

    results = db.execute("""
        MATCH (a:Person)-[r:KNOWS]->(b:Person)
        WHERE a.city = 'NYC' AND b.age < 30
        RETURN a.name, b.name, b.age
    """)
    for result in results:
        print(f"  {result['a.name']} (NYC) knows {result['b.name']} (age {result['b.age']})")
    print()

    # ========================================================================
    # Example 8: Bidirectional relationship matching
    # ========================================================================
    print("8. Bidirectional relationship matching:")
    print("-" * 60)

    results = db.execute("""
        MATCH (a:Person)-[r:KNOWS]-(b:Person)
        WHERE a.name = 'Alice'
        RETURN b.name
    """)
    print(f"  Alice is connected to:")
    for result in results:
        print(f"    - {result['b.name']}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"[OK] Total nodes: {db.get_node_count()}")
    print(f"[OK] Total relationships: {db.get_relationship_count()}")
    print(f"[OK] Cypher queries: CREATE, MATCH, WHERE, RETURN all working!")
    print(f"[OK] Hybrid API: Programmatic + Cypher works seamlessly")
    print()

    db.close()


if __name__ == '__main__':
    main()
