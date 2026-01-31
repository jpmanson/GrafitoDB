# Database API

The `GrafitoDatabase` class is the main entry point for all graph operations.

## Initialization

```python
from grafito import GrafitoDatabase

# In-memory database (data lost on close)
db = GrafitoDatabase(':memory:')

# Persistent database
db = GrafitoDatabase('path/to/database.db')

# With custom settings
db = GrafitoDatabase(
    'graph.db',
    cypher_max_hops=5,      # Default max hops for variable-length paths
    default_top_k=20        # Default k for vector search
)
```

## Context Manager

Use the database as a context manager for automatic transaction handling:

```python
with db:
    node = db.create_node(labels=['Person'], properties={'name': 'Alice'})
    # Automatically commits on success
# Automatically rolls back on exception
```

## Core Operations

### Node Operations

```python
# Create a node
node = db.create_node(
    labels=['Person', 'Employee'],
    properties={'name': 'Alice', 'age': 30}
)

# Get a node by ID
node = db.get_node(node_id)

# Update properties (merges with existing)
db.update_node_properties(node_id, {'city': 'NYC'})

# Add/remove labels
db.add_labels(node_id, ['Manager'])
db.remove_labels(node_id, ['Employee'])

# Delete a node (cascades to relationships)
db.delete_node(node_id)
```

### Relationship Operations

```python
# Create a relationship
rel = db.create_relationship(
    source_id=alice.id,
    target_id=bob.id,
    rel_type='KNOWS',
    properties={'since': 2020}
)

# Get relationship by ID
rel = db.get_relationship(rel_id)

# Delete relationship
db.delete_relationship(rel_id)
```

### Pattern Matching

```python
# Match nodes by labels
persons = db.match_nodes(labels=['Person'])

# Match with property filter
engineers = db.match_nodes(
    labels=['Employee'],
    properties={'department': 'Engineering'}
)

# Match relationships
rels = db.match_relationships(
    source_id=alice.id,
    rel_type='KNOWS'
)
```

### Neighbors and Traversal

```python
# Get neighbors
neighbors = db.get_neighbors(
    node_id=alice.id,
    direction='outgoing',  # 'incoming', 'outgoing', or 'both'
    rel_type='KNOWS'       # optional filter
)

# Find shortest path (BFS)
path = db.find_shortest_path(alice.id, bob.id)

# Find any path with depth limit (DFS)
path = db.find_path(alice.id, bob.id, max_depth=5)
```

## Cypher Queries

Execute Cypher queries directly:

```python
# Create nodes and relationships
db.execute("CREATE (n:Person {name: 'Alice', age: 30})")

# Query with results
results = db.execute("MATCH (n:Person) RETURN n.name, n.age")
for row in results:
    print(f"{row['n.name']}: {row['n.age']}")

# Pattern matching
results = db.execute("""
    MATCH (a:Person)-[:KNOWS]->(b:Person)
    WHERE a.name = 'Alice'
    RETURN b.name
""")
```

## Transactions

### Automatic (Context Manager)

```python
with db:
    node1 = db.create_node(labels=['A'])
    node2 = db.create_node(labels=['B'])
    db.create_relationship(node1.id, node2.id, 'LINKS')
# Commits if no exception, rolls back otherwise
```

### Manual Control

```python
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
# Counts
total_nodes = db.get_node_count()
person_count = db.get_node_count(label='Person')
total_rels = db.get_relationship_count()
knows_count = db.get_relationship_count(rel_type='KNOWS')

# Labels and types
labels = db.get_all_labels()
rel_types = db.get_all_relationship_types()

# Indexes
db.create_node_index('Person', 'name')
db.create_relationship_index('KNOWS', 'since')
db.list_indexes()
db.drop_index('idx_node_person_name')

# Constraints (via Cypher)
db.execute("CREATE CONSTRAINT FOR (n:Person) REQUIRE n.email IS UNIQUE")
db.execute("SHOW CONSTRAINTS")
```

## Vector Search

```python
# Create a vector index
db.create_vector_index(
    name='people_vec',
    dim=384,
    backend='faiss',
    method='flat',
    options={'metric': 'l2'}
)

# Insert embeddings
db.upsert_embedding(
    node_id=node.id,
    vector=[0.1, 0.2, ...],  # 384 dimensions
    index='people_vec'
)

# Semantic search
results = db.semantic_search(
    query_vector=[0.1, 0.2, ...],
    k=10,
    index='people_vec'
)
```

## Full-Text Search

```python
# Check FTS5 availability
if db.has_fts5():
    # Configure text index
    db.create_text_index('node', 'Person', ['name', 'bio'])
    db.rebuild_text_index()

    # Search
    results = db.text_search('engineer', k=10, labels=['Person'])
```

## Import/Export

```python
# NetworkX
graph = db.to_networkx()
db.from_networkx(graph)

# Neo4j dump
db.import_neo4j_dump('path/to/dump.db')
```

## Cleanup

```python
# Close the database (important for persistent databases)
db.close()
```
