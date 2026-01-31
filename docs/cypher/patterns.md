# Complex Patterns

Advanced pattern matching techniques.

## Multiple Patterns

### Matching Separate Patterns

```cypher
// Match multiple independent patterns
MATCH
  (a:Person {name: 'Alice'}),
  (b:Person {name: 'Bob'})
RETURN a, b
```

### Connecting Patterns

```cypher
// Alice knows someone who knows Bob
MATCH
  (a:Person {name: 'Alice'})-[:KNOWS]->(friend),
  (friend)-[:KNOWS]->(b:Person {name: 'Bob'})
RETURN friend.name
```

### Multiple Path Patterns

```cypher
// Find common connections through different paths
MATCH
  (a:Person)-[:WORKS_AT]->(c:Company),
  (a)-[:LIVES_IN]->(city:City)
RETURN a.name, c.name, city.name
```

## OPTIONAL MATCH

Returns NULL for missing parts instead of filtering out.

### Basic Optional Match

```cypher
// Get all persons, with their company if they have one
MATCH (p:Person)
OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company)
RETURN p.name, c.name  // c.name is NULL if no company
```

### Multiple Optionals

```cypher
// Person with optional company and optional location
MATCH (p:Person)
OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company)
OPTIONAL MATCH (p)-[:LIVES_IN]->(city:City)
RETURN p.name, c.name, city.name
```

### Optional with WHERE

```cypher
MATCH (p:Person)
OPTIONAL MATCH (p)-[:KNOWS]->(friend)
WHERE friend.active = true
RETURN p.name, friend.name
```

## Cyclic Patterns

### Self-Referencing

```cypher
// Find mutual relationships
MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(a)
RETURN a.name, b.name
```

### Triangles

```cypher
// Find friend triangles (A knows B, B knows C, C knows A)
MATCH
  (a)-[:KNOWS]->(b),
  (b)-[:KNOWS]->(c),
  (c)-[:KNOWS]->(a)
RETURN a.name, b.name, c.name
```

## Named Paths

### Capturing Paths

```cypher
// Capture entire path
MATCH p = (a:Person)-[:KNOWS*1..3]->(b:Person)
WHERE a.name = 'Alice' AND b.name = 'Bob'
RETURN p, length(p) as hops
```

### Path Functions

```cypher
// Analyze paths
MATCH p = (a)-[:KNOWS*]->(b)
RETURN
  nodes(p) as pathNodes,
  relationships(p) as pathRels,
  length(p) as numHops
```

## Pattern Comprehensions

### Basic Comprehension

```cypher
// Collect friends into a list
MATCH (a:Person {name: 'Alice'})
RETURN [(a)-[:KNOWS]->(b) | b.name] as friends
```

### With Filter

```cypher
// Only active friends
MATCH (a:Person {name: 'Alice'})
RETURN [(a)-[:KNOWS]->(b) WHERE b.active | b.name] as activeFriends
```

### Multi-Element Patterns

```cypher
// Get friends and their companies
MATCH (a:Person {name: 'Alice'})
RETURN [(a)-[:KNOWS]->(b)-[:WORKS_AT]->(c) | {friend: b.name, company: c.name}] as connections
```
