# Quick Start

Get up and running with GrafitoDB in 5 minutes.

## Create Your First Graph

```python
from grafito import GrafitoDatabase

# Initialize database (in-memory for this example)
db = GrafitoDatabase(':memory:')

# Create nodes with labels and properties
alice = db.create_node(
    labels=['Person', 'Employee'],
    properties={'name': 'Alice', 'age': 30, 'city': 'NYC'}
)

bob = db.create_node(
    labels=['Person'],
    properties={'name': 'Bob', 'age': 25, 'city': 'LA'}
)

company = db.create_node(
    labels=['Company'],
    properties={'name': 'TechCorp', 'founded': 2010}
)

print(f"Created nodes: Alice ({alice.id}), Bob ({bob.id}), TechCorp ({company.id})")
```

## Add Relationships

```python
# Create a relationship between Alice and TechCorp
works_at = db.create_relationship(
    alice.id, company.id, 'WORKS_AT',
    properties={'since': 2020, 'position': 'Engineer'}
)

# Create a KNOWS relationship between Alice and Bob
knows = db.create_relationship(alice.id, bob.id, 'KNOWS')

print(f"Alice works at TechCorp since 2020")
```

## Query Your Data

### Programmatic API

```python
# Find all Person nodes
persons = db.match_nodes(labels=['Person'])
for person in persons:
    print(f"{person.properties['name']} - {person.properties.get('age')} years old")

# Find employees over 25
experienced = db.match_nodes(
    labels=['Employee'],
    properties={'age': 30}
)

# Get Alice's connections
connections = db.get_neighbors(alice.id, direction='outgoing')
for node in connections:
    print(f"Alice knows: {node.properties['name']}")
```

### Using Cypher

```python
# Execute Cypher queries
results = db.execute("MATCH (n:Person) RETURN n.name, n.age")
for row in results:
    print(f"{row['n.name']}: {row['n.age']}")

# Find who works where
results = db.execute("""
    MATCH (p:Person)-[r:WORKS_AT]->(c:Company)
    RETURN p.name, c.name, r.position
""")
for row in results:
    print(f"{row['p.name']} is a {row['r.position']} at {row['c.name']}")
```

## Graph Traversal

```python
# Find shortest path between two nodes
path = db.find_shortest_path(alice.id, bob.id)
if path:
    names = ' -> '.join(node.properties['name'] for node in path)
    print(f"Path: {names}")

# Find any path with depth limit
path = db.find_path(alice.id, bob.id, max_depth=3)
```

## Use Transactions

```python
# Option 1: Context manager (recommended)
with db:
    node1 = db.create_node(labels=['Test'])
    node2 = db.create_node(labels=['Test'])
    db.create_relationship(node1.id, node2.id, 'CONNECTS')
# Auto-commits on success, rolls back on exception

# Option 2: Explicit control
db.begin_transaction()
try:
    # ... operations ...
    db.commit()
except Exception:
    db.rollback()
    raise
```

## Metadata Queries

```python
# Inspect your graph
print(f"Total nodes: {db.get_node_count()}")
print(f"Total relationships: {db.get_relationship_count()}")
print(f"All labels: {db.get_all_labels()}")
print(f"All relationship types: {db.get_all_relationship_types()}")
```

## Complete Example

```python
from grafito import GrafitoDatabase

def main():
    # Create database
    db = GrafitoDatabase(':memory:')

    # Create a mini social network
    with db:
        # Create people
        alice = db.create_node(labels=['Person'], properties={'name': 'Alice', 'age': 30})
        bob = db.create_node(labels=['Person'], properties={'name': 'Bob', 'age': 25})
        carol = db.create_node(labels=['Person'], properties={'name': 'Carol', 'age': 35})

        # Create relationships
        db.create_relationship(alice.id, bob.id, 'KNOWS', properties={'since': 2015})
        db.create_relationship(bob.id, carol.id, 'KNOWS', properties={'since': 2018})
        db.create_relationship(alice.id, carol.id, 'KNOWS', properties={'since': 2020})

    # Query: Find friends of friends
    results = db.execute("""
        MATCH (me:Person {name: 'Alice'})-[:KNOWS]->(friend)-[:KNOWS]->(fof)
        WHERE fof <> me
        RETURN DISTINCT fof.name
    """)

    print("Friends of Alice's friends:")
    for row in results:
        print(f"  - {row['fof.name']}")

    # Cleanup
    db.close()

if __name__ == '__main__':
    main()
```

## Next Steps

- Learn about [Core Concepts](concepts.md)
- Explore the [Core API](../api/database.md)
- Master [Cypher Query Language](../cypher/overview.md)
