# Cypher Query Language

GrafitoDB includes a complete Cypher parser and executor, allowing you to use Neo4j-style declarative queries.

## What is Cypher?

Cypher is a declarative graph query language that describes patterns in graphs using ASCII-art notation.

```cypher
// Find Alice's friends
MATCH (a:Person {name: 'Alice'})-[:KNOWS]->(friend)
RETURN friend.name
```

## Why Use Cypher?

| Approach | Best For |
|----------|----------|
| **Programmatic API** | Simple CRUD, dynamic queries |
| **Cypher** | Complex patterns, analytics, reporting |

## Basic Pattern Syntax

### Nodes

```cypher
// Any node
(n)

// Node with label
(p:Person)

// Node with multiple labels
(p:Person:Employee)

// Node with variable and label
(a:Person)

// Anonymous node
(:Person)
```

### Relationships

```cypher
// Directed relationship
(a)-[:KNOWS]->(b)

// Relationship with properties
(a)-[:KNOWS {since: 2020}]->(b)

// Anonymous relationship
(a)-->(b)

// Bidirectional (matches both directions)
(a)-[:KNOWS]-(b)
```

### Paths

```cypher
// Two-hop path
(a)-[:KNOWS]->()-[:WORKS_AT]->(c)

// Variable-length path
(a)-[:KNOWS*1..3]->(b)

// Named path
path = (a)-[:KNOWS*]->(b)
```

## Executing Cypher in Grafito

```python
from grafito import GrafitoDatabase

db = GrafitoDatabase(':memory:')

# Execute without results
db.execute("CREATE (n:Person {name: 'Alice'})")

# Execute with results
results = db.execute("MATCH (n:Person) RETURN n.name")
for row in results:
    print(row['n.name'])
```

## Query Structure

A typical Cypher query has this structure:

```cypher
[USE]           // Graph selection (not in Grafito)
[READING]       // MATCH, OPTIONAL MATCH
[PROJECTING]    // WITH
[READING]       // Additional MATCH
[WHERE]         // Filters
[PROJECTING]    // RETURN
[ORDER BY]      // Sorting
[SKIP]          // Pagination
[LIMIT]         // Pagination
```

## Clauses Overview

| Clause | Purpose | Example |
|--------|---------|---------|
| `MATCH` | Find patterns | `MATCH (n:Person)` |
| `CREATE` | Create nodes/relationships | `CREATE (n:Person)` |
| `MERGE` | Find or create | `MERGE (n:Person {email: 'a@b'})` |
| `SET` | Update properties | `SET n.name = 'Alice'` |
| `DELETE` | Remove nodes/relationships | `DELETE n` |
| `REMOVE` | Remove labels/properties | `REMOVE n:OldLabel` |
| `RETURN` | Define output | `RETURN n.name` |
| `WHERE` | Filter results | `WHERE n.age > 25` |
| `ORDER BY` | Sort results | `ORDER BY n.name ASC` |
| `SKIP` | Skip N results | `SKIP 10` |
| `LIMIT` | Limit results | `LIMIT 10` |
| `UNION` | Combine results | `... UNION ...` |
| `CALL` | Execute procedures | `CALL db.vector.search(...)` |

## Quick Examples

### Create Data

```python
db.execute("""
    CREATE (a:Person {name: 'Alice', age: 30}),
           (b:Person {name: 'Bob', age: 25}),
           (a)-[:KNOWS {since: 2020}]->(b)
""")
```

### Query Patterns

```python
# Find all persons
results = db.execute("MATCH (n:Person) RETURN n.name")

# Find specific person
results = db.execute("MATCH (n:Person {name: 'Alice'}) RETURN n")

# Find relationships
results = db.execute("""
    MATCH (a:Person)-[r:KNOWS]->(b:Person)
    RETURN a.name, b.name, r.since
""")
```

### Update Data

```python
# Update property
db.execute("""
    MATCH (n:Person {name: 'Alice'})
    SET n.age = 31
""")

# Add label
db.execute("""
    MATCH (n:Person {name: 'Alice'})
    SET n:Employee
""")

# Remove property
db.execute("""
    MATCH (n:Person)
    REMOVE n.temporary
""")
```

### Delete Data

```python
# Delete relationship
db.execute("""
    MATCH (a)-[r:KNOWS]->(b)
    WHERE a.name = 'Alice'
    DELETE r
""")

# Delete node (and its relationships)
db.execute("""
    MATCH (n:Person {name: 'Alice'})
    DELETE n
""")

# Delete everything (careful!)
db.execute("MATCH (n) DETACH DELETE n")
```

## Variable-Length Paths

Configure default max hops when creating the database:

```python
db = GrafitoDatabase(':memory:', cypher_max_hops=5)

# Unbounded uses default max
db.execute("MATCH (a)-[:KNOWS*..]->(b) RETURN b")

# Explicit bounds
db.execute("MATCH (a)-[:KNOWS*1..3]->(b) RETURN b")
```

## Result Format

Query results are returned as a list of dictionaries:

```python
results = db.execute("""
    MATCH (n:Person)
    RETURN n.name, n.age
""")

for row in results:
    print(f"Name: {row['n.name']}")
    print(f"Age: {row['n.age']}")
```

With aliases:

```python
results = db.execute("""
    MATCH (n:Person)
    RETURN n.name AS name, n.age AS age
""")

for row in results:
    print(f"{row['name']}: {row['age']}")
```

## Hybrid Usage

Mix Cypher with the programmatic API:

```python
# Create with API
alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})

# Query with Cypher
results = db.execute("MATCH (n:Person) RETURN n")

# Update with Cypher
db.execute(f"MATCH (n) WHERE id(n) = {alice.id} SET n.active = true")

# Verify with API
updated = db.get_node(alice.id)
print(updated.properties['active'])  # True
```

## Next Steps

- [CREATE and MATCH](create-match.md)
- [Filtering with WHERE](filtering.md)
- [Return and Aggregation](return-aggregation.md)
- [Data Modification](modification.md)
- [Complex Patterns](patterns.md)
