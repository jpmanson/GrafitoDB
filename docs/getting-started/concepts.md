# Core Concepts

Understanding the Property Graph Model used by Grafito.

## What is a Property Graph?

A **Property Graph** is a graph data model where:

- **Nodes** (also called vertices) represent entities
- **Relationships** (also called edges) connect nodes and represent how entities relate
- Both nodes and relationships can have **properties** (key-value pairs)
- Nodes can have **labels** that categorize them

## Nodes

Nodes are the primary entities in your graph.

### Labels

Nodes can have multiple labels, similar to how an object can implement multiple interfaces:

```python
# Alice is both a Person and an Employee
alice = db.create_node(
    labels=['Person', 'Employee'],
    properties={'name': 'Alice'}
)
```

Common use cases for multiple labels:
- `Person` + `Employee` + `Manager` (role hierarchy)
- `Product` + `Featured` (categorization)
- `User` + `Admin` (permissions)

### Properties

Properties are JSON-serializable key-value pairs:

```python
node = db.create_node(
    labels=['Person'],
    properties={
        'name': 'Alice',
        'age': 30,
        'active': True,
        'tags': ['developer', 'python'],
        'metadata': {'joined': '2024-01-15'}
    }
)
```

**Supported property types:**
- `int` - Integer numbers
- `float` - Floating point numbers
- `str` - Strings
- `bool` - Boolean values
- `list` - Arrays (must contain supported types)
- `dict` - Maps/objects (keys must be strings)
- `None` - Null values

**Temporal types** (stored as ISO 8601 strings):
- `date` - ISO date: `'2024-01-15'`
- `datetime` - ISO datetime: `'2024-01-15T10:30:00'`
- `time` - ISO time: `'10:30:00'`
- `duration` - ISO duration: `'P1D'` (1 day)

## Relationships

Relationships are directed connections between nodes.

### Direction

Relationships have a direction (from source to target):

```python
# Alice KNOWS Bob (direction matters)
db.create_relationship(alice.id, bob.id, 'KNOWS')

# Bob is not necessarily knowing Alice in this model
```

### Type

Relationship types describe the nature of the connection:

```python
db.create_relationship(alice.id, company.id, 'WORKS_AT')
db.create_relationship(alice.id, bob.id, 'KNOWS')
db.create_relationship(alice.id, project.id, 'MANAGES')
```

### Properties

Like nodes, relationships can have properties:

```python
db.create_relationship(
    alice.id, company.id, 'WORKS_AT',
    properties={
        'since': 2020,
        'position': 'Senior Engineer',
        'department': 'Engineering'
    }
)
```

## Graph Patterns

Graph patterns describe the structure you're looking for:

### Basic Pattern

```cypher
(a:Person)-[:KNOWS]->(b:Person)
```

This pattern matches:
- A node labeled `Person` (bound to variable `a`)
- A `KNOWS` relationship (direction: `a` → `b`)
- Another node labeled `Person` (bound to variable `b`)

### Variable-Length Patterns

Match paths of variable length:

```cypher
# 1 to 3 hops
(a:Person)-[:KNOWS*1..3]->(b:Person)

# Any number of hops (uses default max limit)
(a:Person)-[:KNOWS*..]->(b:Person)
```

### Multiple Patterns

Combine multiple patterns in one query:

```cypher
MATCH
  (a:Person)-[:KNOWS]->(b:Person),
  (b)-[:WORKS_AT]->(c:Company)
RETURN a.name, b.name, c.name
```

## Traversal

Graph traversal means navigating through the graph by following relationships.

### Direction

When traversing, you can specify direction:

```python
# Outgoing: Alice → ?
outgoing = db.get_neighbors(alice.id, direction='outgoing')

# Incoming: ? → Alice
incoming = db.get_neighbors(alice.id, direction='incoming')

# Both directions
all_neighbors = db.get_neighbors(alice.id, direction='both')
```

### Path Finding

Find paths between nodes:

```python
# Shortest path (BFS algorithm)
path = db.find_shortest_path(alice.id, bob.id)

# Any path with depth limit (DFS algorithm)
path = db.find_path(alice.id, bob.id, max_depth=5)
```

## Indexes and Constraints

### Property Indexes

Speed up queries on frequently accessed properties:

```python
# Create index on Person.name
db.create_node_index('Person', 'name')

# Or via Cypher
db.execute("CREATE INDEX FOR (n:Person) ON (n.name)")
```

### Constraints

Enforce data integrity:

```python
# Unique constraint
db.execute("CREATE CONSTRAINT FOR (n:User) REQUIRE n.email IS UNIQUE")

# Type constraint
db.execute("CREATE CONSTRAINT FOR (n:Person) REQUIRE n.age IS INTEGER")

# Existence constraint
db.execute("CREATE CONSTRAINT FOR (n:Person) REQUIRE n.name IS NOT NULL")
```

## Storage Model

Grafito uses a normalized SQLite schema:

```sql
-- Nodes table
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY,
    properties TEXT,  -- JSON
    created_at REAL
);

-- Labels (normalized)
CREATE TABLE labels (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);

-- Node-Label junction
CREATE TABLE node_labels (
    node_id INTEGER,
    label_id INTEGER
);

-- Relationships
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    source_node_id INTEGER,
    target_node_id INTEGER,
    type TEXT,
    properties TEXT  -- JSON
);
```

Benefits of this design:
- **Efficient queries** via indexes on labels and types
- **Flexible properties** via JSON storage
- **ACID compliance** via SQLite transactions
- **Small footprint** suitable for embedding

## Comparison with Other Models

### vs. Relational (SQL)

| Aspect | Relational | Property Graph |
|--------|-----------|----------------|
| Schema | Rigid tables | Flexible labels/properties |
| Relationships | Foreign keys | First-class entities |
| Traversal | JOINs | Direct navigation |
| Best for | Structured data | Connected data |

### vs. Document (MongoDB)

| Aspect | Document | Property Graph |
|--------|----------|----------------|
| Structure | Nested documents | Nodes + edges |
| Relationships | Embedded references | Native connections |
| Queries | Document-based | Pattern-based |
| Best for | Content management | Relationship analysis |

### vs. RDF (Triple Stores)

| Aspect | RDF | Property Graph |
|--------|-----|----------------|
| Model | Subject-Predicate-Object | Nodes + labeled relationships |
| Schema | Ontologies | Labels + properties |
| Standards | W3C standards | De facto (Neo4j) |
| Best for | Semantic web | Application graphs |

## Next Steps

- Learn the [Core API](../api/database.md)
- Explore [Cypher queries](../cypher/overview.md)
- See [examples](../examples/social-network.md) in action