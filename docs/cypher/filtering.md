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

### Three-Valued Logic (True/False/Null)

Cypher uses three-valued logic for boolean operations involving `null`:

#### AND Operator

| Left | Right | Result |
|------|-------|--------|
| `true` | `true` | `true` |
| `true` | `false` | `false` |
| `true` | `null` | `null` |
| `false` | *any* | `false` |
| `null` | `false` | `false` |
| `null` | `true` | `null` |
| `null` | `null` | `null` |

```cypher
// Examples
RETURN true AND true      // true
RETURN true AND null      // null
RETURN false AND null     // false
RETURN null AND null      // null
```

#### OR Operator

| Left | Right | Result |
|------|-------|--------|
| `true` | *any* | `true` |
| `false` | `false` | `false` |
| `false` | `true` | `true` |
| `false` | `null` | `null` |
| `null` | `true` | `true` |
| `null` | `false` | `null` |
| `null` | `null` | `null` |

```cypher
// Examples
RETURN true OR null       // true
RETURN false OR null      // null
RETURN null OR false      // null
RETURN null OR true       // true
```

#### IN Operator with NULL

```cypher
// Value exists in list
RETURN 2 IN [1, 2, 3]           // true

// Value doesn't exist
RETURN 5 IN [1, 2, 3]           // false

// List contains NULL
RETURN 5 IN [1, 2, null]        // null (might exist, unknown)
RETURN 2 IN [1, 2, null]        // true (definitely exists)

// NULL value
RETURN null IN [1, 2, 3]        // null
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

### List-Scalar Comparisons

When comparing a list with a scalar value using `=` or `!=`, Cypher checks if the value exists in the list:

```cypher
// Check if scalar equals any element in list
WITH [1, 2, 3] as nums
WHERE nums = 2
RETURN 'found'      // 'found' - 2 is in the list

// Check if scalar is not in list
WITH [1, 2, 3] as nums
WHERE nums != 5
RETURN 'not found'  // 'not found' - 5 is not in the list

// Works with nodes too
MATCH (p:Person)
WITH collect(p) as persons
WHERE persons = 'Alice'    // true if any person is named Alice
RETURN persons
```

!!! note "Only = and !="
    This shorthand only works with equality (`=`) and inequality (`!=`) operators, not with `<`, `>`, etc.

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

## Label Predicates

Test a node's labels directly in `WHERE` with `n:Label`. Chain labels with `:`
to require all of them, and combine with `NOT`, `AND`, `OR` like any boolean:

```cypher
// Nodes carrying the Admin label
MATCH (n)
WHERE n:Admin
RETURN n.name

// Both labels required
MATCH (n)
WHERE n:Person:Admin
RETURN n.name

// Negation and combination
MATCH (n)
WHERE NOT n:Reference AND (n:ADR OR n:Term)
RETURN n.title
```

The same predicate can be projected as a boolean in `RETURN`:

```cypher
MATCH (n:Person {name: 'Alice'})
RETURN n:Admin AS is_admin
```

## Parameters

Use `$name` parameters instead of interpolating values into the `WHERE`
condition (see [Query Parameters](overview.md#query-parameters)):

```cypher
MATCH (n:Person)
WHERE n.age >= $min_age AND n.name IN $names
RETURN n.name
```

```python
db.execute(query, {"min_age": 18, "names": ["Alice", "Bob"]})
```

## Pattern Filtering

A pattern can be used directly as a truth value: it is true when it matches at
least once, holding fixed whatever variables are already bound.

```cypher
// Persons who know someone
MATCH (p:Person)
WHERE (p)-[:KNOWS]->()
RETURN p.name

// ...and the reason this exists: persons who know nobody
MATCH (p:Person)
WHERE NOT (p)-[:KNOWS]->()
RETURN p.name
```

Matching stops at the first hit, so a node with thousands of neighbours costs
the same as one with a single edge — unlike counting them.

### What the Pattern Can Contain

Anything `MATCH` accepts: types, labels, direction, inline properties, and
variable-length paths.

```cypher
WHERE (p)-[:WORKS_AT]->(:Company)          // label on the far node
WHERE (p)-[:KNOWS {since: 2020}]->()       // property on the relationship
WHERE (p)-->({age: 30})                    // property on the far node
WHERE ()-[:REPORTS_TO]->(p)                // incoming
WHERE (p)--()                              // either direction
WHERE (p)-[*1..3]->(:Company)              // within three hops
WHERE (p)-->()-->(:Company)                // multi-hop
```

Variables already bound by the surrounding query keep their values, so a pattern
over two of them asks about those specific nodes:

```cypher
MATCH (p:Person), (c:Company {name: 'Acme'})
WHERE NOT (p)-[:WORKS_AT]->(c)
RETURN p.name          // everyone who does not work at Acme
```

### EXISTS

`EXISTS()` over a pattern means the same thing, and reads better when combined
with other predicates:

```cypher
MATCH (p:Person)
WHERE EXISTS((p)-[:KNOWS]->()) AND p.age > 30
RETURN p.name
```

Note that `EXISTS()` is overloaded: given a property it tests for presence
(see [Property Existence](#property-existence)), given a pattern it tests for a
match.

### As a Value

A pattern predicate is an ordinary boolean expression, so it can be returned or
projected:

```cypher
MATCH (p:Person)
RETURN p.name, (p)-[:KNOWS]->() AS is_connected
```

To *count* matches rather than test for one, use a
[pattern comprehension](collections.md):

```cypher
MATCH (p:Person)
RETURN p.name, size([(p)-[:KNOWS]->(f) | f]) AS friends
```

!!! note "Filtering on a WITH alias"
    `WITH n, (n)-->() AS connected WHERE connected` does not work — and not
    because of patterns: referring to a `WITH` alias inside that same clause's
    `WHERE` is unsupported for any expression. Filter in a following clause:

    ```cypher
    MATCH (n:Person)
    WITH n
    WHERE (n)-[:KNOWS]->()
    RETURN n.name
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
