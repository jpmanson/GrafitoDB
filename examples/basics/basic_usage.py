"""Basic usage example for Grafito graph database.

This example demonstrates the core CRUD operations:
- Creating nodes with labels and properties
- Creating relationships between nodes
- Querying and updating data
- Using metadata queries
"""

from grafito import GrafitoDatabase


def main():
    print("=== Grafito Basic Usage Example ===\n")

    # Initialize in-memory database
    db = GrafitoDatabase(':memory:')

    # =========================================================================
    # Create Nodes
    # =========================================================================
    print("1. Creating nodes...")

    alice = db.create_node(
        labels=['Person', 'Employee'],
        properties={'name': 'Alice', 'age': 30, 'city': 'NYC'}
    )
    print(f"   Created: {alice}")

    bob = db.create_node(
        labels=['Person'],
        properties={'name': 'Bob', 'age': 25}
    )
    print(f"   Created: {bob}")

    company = db.create_node(
        labels=['Company'],
        properties={'name': 'TechCorp', 'founded': 2010}
    )
    print(f"   Created: {company}\n")

    # =========================================================================
    # Create Relationships
    # =========================================================================
    print("2. Creating relationships...")

    works_at = db.create_relationship(
        alice.id, company.id, 'WORKS_AT',
        properties={'since': 2020, 'position': 'Engineer'}
    )
    print(f"   Created: {works_at}")

    knows = db.create_relationship(
        alice.id, bob.id, 'KNOWS',
        properties={'since': 2015}
    )
    print(f"   Created: {knows}\n")

    # =========================================================================
    # Query Nodes
    # =========================================================================
    print("3. Querying nodes...")

    # Find all persons
    persons = db.match_nodes(labels=['Person'])
    print(f"   Found {len(persons)} persons:")
    for person in persons:
        print(f"     - {person.properties['name']} (age: {person.properties.get('age', 'N/A')})")

    # Find employees in NYC
    nyc_employees = db.match_nodes(
        labels=['Employee'],
        properties={'city': 'NYC'}
    )
    print(f"\n   Found {len(nyc_employees)} employees in NYC:")
    for emp in nyc_employees:
        print(f"     - {emp.properties['name']}")
    print()

    # =========================================================================
    # Update Node
    # =========================================================================
    print("4. Updating node properties...")

    db.update_node_properties(bob.id, {'city': 'LA', 'title': 'Designer'})
    updated_bob = db.get_node(bob.id)
    print(f"   Updated Bob: {updated_bob}\n")

    # =========================================================================
    # Add Labels
    # =========================================================================
    print("5. Adding labels...")

    db.add_labels(bob.id, ['Employee', 'Contractor'])
    updated_bob = db.get_node(bob.id)
    print(f"   Bob's labels: {updated_bob.labels}\n")

    # =========================================================================
    # Query Relationships
    # =========================================================================
    print("6. Querying relationships...")

    # Find all WORKS_AT relationships
    work_rels = db.match_relationships(rel_type='WORKS_AT')
    print(f"   Found {len(work_rels)} WORKS_AT relationships")
    for rel in work_rels:
        source = db.get_node(rel.source_id)
        target = db.get_node(rel.target_id)
        print(f"     - {source.properties['name']} works at {target.properties['name']}")
    print()

    # =========================================================================
    # Get Neighbors
    # =========================================================================
    print("7. Finding neighbors...")

    alice_neighbors = db.get_neighbors(alice.id, direction='outgoing')
    print(f"   Alice's outgoing connections ({len(alice_neighbors)}):")
    for neighbor in alice_neighbors:
        print(f"     - {neighbor.properties['name']}")
    print()

    # =========================================================================
    # Graph Traversal - Find Shortest Path
    # =========================================================================
    print("8. Finding shortest path...")

    # Create another connection to demonstrate path finding
    charlie = db.create_node(
        labels=['Person'],
        properties={'name': 'Charlie', 'age': 35}
    )
    db.create_relationship(bob.id, charlie.id, 'KNOWS')

    path = db.find_shortest_path(alice.id, charlie.id)
    if path:
        print(f"   Shortest path from Alice to Charlie ({len(path)} nodes):")
        for i, node in enumerate(path):
            print(f"     {i+1}. {node.properties['name']}")
    print()

    # =========================================================================
    # Metadata Queries
    # =========================================================================
    print("9. Metadata queries...")

    all_labels = db.get_all_labels()
    print(f"   All labels in database: {all_labels}")

    all_rel_types = db.get_all_relationship_types()
    print(f"   All relationship types: {all_rel_types}")

    total_nodes = db.get_node_count()
    person_count = db.get_node_count(label='Person')
    print(f"   Total nodes: {total_nodes}, Persons: {person_count}")

    total_rels = db.get_relationship_count()
    knows_count = db.get_relationship_count(rel_type='KNOWS')
    print(f"   Total relationships: {total_rels}, KNOWS: {knows_count}\n")

    # =========================================================================
    # Transaction Example
    # =========================================================================
    print("10. Transaction example...")

    try:
        with db:
            # These operations are in a transaction
            diana = db.create_node(
                labels=['Person'],
                properties={'name': 'Diana'}
            )
            db.create_relationship(diana.id, company.id, 'WORKS_AT')
            print("   Transaction committed successfully")
    except Exception as e:
        print(f"   Transaction failed: {e}")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n11. Cleanup...")

    # Delete a node (cascade deletes relationships)
    db.delete_node(charlie.id)
    print(f"   Deleted Charlie and associated relationships")
    print(f"   Remaining nodes: {db.get_node_count()}\n")

    # Close database
    db.close()
    print("=== Example completed successfully ===")


if __name__ == '__main__':
    main()
