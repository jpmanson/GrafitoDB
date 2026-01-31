# Filtering with WHERE

The WHERE clause filters results from MATCH operations.

## Basic Comparisons

### Comparison Operators

```cypher
// Equal
MATCH (p:Person)
WHERE p.name = 'Alice'
RETURN p

// Not equal
MATCH (p:Person)
WHERE p.name <> 'Alice'
RETURN p

// Greater than
MATCH (p:Person)
WHERE p.age > 25
RETURN p.name

// Greater or equal
MATCH (p:Person)
WHERE p.age >= 18
RETURN p.name

// Less than
MATCH (p:Person)
WHERE p.age < 65
RETURN p.name

// Less or equal
MATCH (p:Person)
WHERE p.age <= 30
RETURN p.name
```

### Range Comparisons

```cypher
// Chained comparisons
MATCH (p:Person)
WHERE 18 <= p.age <= 65
RETURN p.name

// Equivalent to:
MATCH (p:Person)
WHERE p.age >= 18 AND p.age <= 65
RETURN p.name
```

## Logical Operators

### AND

```cypher
MATCH (p:Person)
WHERE p.age > 25 AND p.city = 'NYC'
RETURN p.name
```

### OR

```cypher
MATCH (p:Person)
WHERE p.city = 'NYC' OR p.city = 'LA'
RETURN p.name
```

### NOT

```cypher
MATCH (p:Person)
WHERE NOT p.active
RETURN p.name

// Can also use for negation
MATCH (p:Person)
WHERE NOT (p.age > 25)
RETURN p.name
```

### Combining Operators

```cypher
// Use parentheses for clarity
MATCH (p:Person)
WHERE (p.age > 25 AND p.city = 'NYC') OR p.city = 'LA'
RETURN p.name

// Complex condition
MATCH (p:Person)
WHERE p.active = true
  AND (p.age < 25 OR p.age > 65)
  AND NOT p.name = 'Admin'
RETURN p.name
```

## NULL Handling

### IS NULL / IS NOT NULL

```cypher
// Find persons without age
MATCH (p:Person)
WHERE p.age IS NULL
RETURN p.name

// Find persons with age set
MATCH (p:Person)
WHERE p.age IS NOT NULL
RETURN p.name, p.age
```

### NULL Comparisons

```cypher
// NULL comparisons return NULL (falsy in WHERE)
MATCH (p:Person)
WHERE p.age = NULL  // This never matches!
RETURN p

// Correct way
MATCH (p:Person)
WHERE p.age IS NULL
RETURN p
```

## String Matching

### STARTS WITH

```cypher
// Names starting with 'Al'
MATCH (p:Person)
WHERE p.name STARTS WITH 'Al'
RETURN p.name
```

### ENDS WITH

```cypher
// Emails ending with '@company.com'
MATCH (p:Person)
WHERE p.email ENDS WITH '@company.com'
RETURN p.name
```

### CONTAINS

```cypher
// Names containing 'li'
MATCH (p:Person)
WHERE p.name CONTAINS 'li'
RETURN p.name
```

### Regular Expressions

```cypher
// Regexp matching
MATCH (p:Person)
WHERE p.name =~ '^A.*e$'
RETURN p.name

// Case insensitive (with (?i))
MATCH (p:Person)
WHERE p.name =~ '(?i)^alice$'
RETURN p.name
```

## List Operations

### IN Operator

```cypher
// Match any of these names
MATCH (p:Person)
WHERE p.name IN ['Alice', 'Bob', 'Charlie']
RETURN p.name

// Check if value in list property
MATCH (p:Person)
WHERE 'developer' IN p.tags
RETURN p.name
```

### List Predicates

```cypher
// All elements satisfy condition
WITH [1, 2, 3] as list
WHERE ALL(x IN list WHERE x > 0)
RETURN list

// Any element satisfies condition
WITH [1, -1, 2] as list
WHERE ANY(x IN list WHERE x < 0)
RETURN list

// No element satisfies condition
WITH [1, 2, 3] as list
WHERE NONE(x IN list WHERE x < 0)
RETURN list

// Exactly one element satisfies condition
WITH [1, 2, 3] as list
WHERE SINGLE(x IN list WHERE x = 2)
RETURN list
```

## Pattern Filtering

### EXISTS

```cypher
// Persons who know someone
MATCH (p:Person)
WHERE EXISTS((p)-[:KNOWS]->())
RETURN p.name

// Persons who work at a company
MATCH (p:Person)
WHERE EXISTS((p)-[:WORKS_AT]->(:Company))
RETURN p.name
```

### Filtering on Patterns

```cypher
// Persons who know someone over 30
MATCH (p:Person)
WHERE EXISTS((p)-[:KNOWS]->({age: 30}))
RETURN p.name
```

## Property Existence

```cypher
// Using EXISTS for property (alternate to IS NOT NULL)
MATCH (p:Person)
WHERE EXISTS(p.email)
RETURN p.name

// Combined with value check
MATCH (p:Person)
WHERE EXISTS(p.age) AND p.age > 25
RETURN p.name
```

## Examples by Use Case

### User Management

```cypher
// Active users with verified email
MATCH (u:User)
WHERE u.active = true
  AND u.emailVerified = true
  AND u.createdAt > date('2024-01-01')
RETURN u.email
```

### Product Search

```cypher
// Available products in category
MATCH (p:Product)
WHERE p.category IN ['electronics', 'computers']
  AND p.stock > 0
  AND (p.price >= 100 AND p.price <= 500)
RETURN p.name, p.price
```

### Social Network

```cypher
// Active friends in same city
MATCH (me:Person {name: 'Alice'})-[:KNOWS]->(friend)
WHERE friend.active = true
  AND friend.city = me.city
  AND NOT friend.name = me.name
RETURN friend.name
```
