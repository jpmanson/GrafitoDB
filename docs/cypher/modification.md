# Data Modification

Modifying existing data with SET, DELETE, REMOVE, and MERGE.

## SET

Updates properties on nodes and relationships.

### Set Single Property

```cypher
// Set one property
MATCH (p:Person {name: 'Alice'})
SET p.age = 31
```

### Set Multiple Properties

```cypher
// Set multiple properties
MATCH (p:Person {name: 'Alice'})
SET p.city = 'NYC', p.country = 'USA'
```

### Set from Map

```cypher
// Replace all properties using =
MATCH (p:Person {name: 'Alice'})
SET p = {name: 'Alice', age: 31, city: 'NYC'}

// This removes all other properties!
```

### Add/Update Properties with +=

```cypher
// Merge properties (add new, update existing, keep others)
MATCH (p:Person {name: 'Alice'})
SET p += {city: 'NYC', department: 'Engineering'}
```

### Set with Expression

```cypher
// Increment value
MATCH (p:Person {name: 'Alice'})
SET p.loginCount = p.loginCount + 1

// Set from calculation
MATCH (p:Person)
SET p.ageNextYear = p.age + 1
```

### Set with CREATE/MATCH

```cypher
// Create and set in one
CREATE (p:Person)
SET p.name = 'Bob', p.age = 25
```

## DELETE

Removes nodes and relationships.

### Delete Relationship

```cypher
// Delete specific relationship
MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})
DELETE r
```

### Delete Node

```cypher
// Delete node (must detach relationships first)
MATCH (p:Person {name: 'Alice'})
DELETE p
```

### DETACH DELETE

```cypher
// Delete node and all its relationships
MATCH (p:Person {name: 'Alice'})
DETACH DELETE p
```

### Delete Multiple

```cypher
// Delete multiple items
MATCH (p:Person {name: 'Alice'})-[r]-(), (p)
DELETE r, p
```

## REMOVE

Removes labels and properties.

### Remove Property

```cypher
// Remove single property
MATCH (p:Person {name: 'Alice'})
REMOVE p.temporary

// Remove multiple properties
MATCH (p:Person {name: 'Alice'})
REMOVE p.temp1, p.temp2
```

### Remove Label

```cypher
// Remove single label
MATCH (p:Person:Employee {name: 'Alice'})
REMOVE p:Employee

// Remove multiple labels
MATCH (p:Person:Employee:Manager {name: 'Alice'})
REMOVE p:Employee, p:Manager
// Now p only has :Person label
```

### Remove All Labels

```cypher
MATCH (p:Person {name: 'Alice'})
REMOVE p:Person
// Node now has no labels
```

## MERGE

Finds existing patterns or creates them if not found.

### Basic MERGE

```cypher
// Find or create node
MERGE (p:Person {email: 'alice@example.com'})
RETURN p
```

### MERGE with ON CREATE

```cypher
// Set properties only when creating
MERGE (p:Person {email: 'alice@example.com'})
ON CREATE SET
  p.name = 'Alice',
  p.createdAt = datetime(),
  p.active = true
RETURN p
```

### MERGE with ON MATCH

```cypher
// Update properties only when existing
MERGE (p:Person {email: 'alice@example.com'})
ON MATCH SET
  p.lastSeen = datetime(),
  p.visitCount = p.visitCount + 1
RETURN p
```

### Combined ON CREATE and ON MATCH

```cypher
MERGE (p:Person {email: 'alice@example.com'})
ON CREATE SET
  p.name = 'Alice',
  p.createdAt = datetime()
ON MATCH SET
  p.lastSeen = datetime()
RETURN p
```

### MERGE Relationship

```cypher
// Find or create relationship
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
MERGE (a)-[r:KNOWS]->(b)
RETURN r

// With properties
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
MERGE (a)-[r:KNOWS]->(b)
ON CREATE SET r.since = 2020
RETURN r
```

### MERGE Path

```cypher
// Create entire path if not exists
MERGE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})
RETURN a, b
```

### MERGE with Multiple Patterns

```cypher
// Match/create both nodes and relationship
MERGE (a:Person {email: 'alice@example.com'})
MERGE (b:Person {email: 'bob@example.com'})
MERGE (a)-[:KNOWS]->(b)
```

## FOREACH

Iterate over a list and perform operations.

```cypher
// Set property on each element
MATCH (p:Person {name: 'Alice'})
FOREACH (tag IN ['developer', 'python', 'database'] |
  CREATE (t:Tag {name: tag})
  MERGE (p)-[:INTERESTED_IN]->(t)
)
```

## Common Patterns

### Upsert Pattern

```cypher
// Update or insert (upsert)
MERGE (p:Person {email: $email})
ON CREATE SET
  p.name = $name,
  p.createdAt = datetime(),
  p.version = 1
ON MATCH SET
  p.name = $name,
  p.updatedAt = datetime(),
  p.version = p.version + 1
RETURN p
```

### Soft Delete

```cypher
// Instead of DELETE, mark as inactive
MATCH (p:Person {name: 'Alice'})
SET p.active = false, p.deletedAt = datetime()
REMOVE p:Active
```

### Bulk Update

```cypher
// Update many nodes at once
MATCH (p:Person)
WHERE p.lastLogin < datetime() - duration('P1Y')
SET p.status = 'inactive'
```

### Property Migration

```cypher
// Rename property
MATCH (p:Person)
WHERE EXISTS(p.oldName)
SET p.newName = p.oldName
REMOVE p.oldName
```
