# Lists and Maps

Working with collections in Cypher.

## Lists

### Creating Lists

```cypher
// Literal list
RETURN [1, 2, 3] as numbers

// Mixed types
RETURN ['Alice', 30, true] as mixed

// Nested lists
RETURN [[1, 2], [3, 4]] as matrix
```

### List Operations

```cypher
// Concatenation
WITH [1, 2] as a, [3, 4] as b
RETURN a + b as combined  // [1, 2, 3, 4]

// Index access (0-based)
WITH ['Alice', 'Bob', 'Charlie'] as names
RETURN names[0] as first, names[1] as second

// Negative indexing (from end)
WITH [1, 2, 3, 4, 5] as nums
RETURN nums[-1] as last  // 5
```

### List Slicing

```cypher
WITH [1, 2, 3, 4, 5] as nums
RETURN nums[1..3]  // [2, 3] (exclusive end)
RETURN nums[2..]   // [3, 4, 5] (to end)
RETURN nums[..3]   // [1, 2, 3] (from start)
```

### List Functions

```cypher
// Size
RETURN size([1, 2, 3])  // 3

// Head, tail, last
WITH [1, 2, 3, 4] as list
RETURN head(list)  // 1
RETURN last(list)  // 4
RETURN tail(list)  // [2, 3, 4]

// Reverse
RETURN reverse([1, 2, 3])  // [3, 2, 1]
```

### List Comprehensions

```cypher
// Filter and transform
WITH [1, 2, 3, 4, 5] as nums
RETURN [x IN nums WHERE x > 2 | x * 10] as result
// [30, 40, 50]

// Extract only
RETURN [x IN [1, 2, 3] | x * 2]  // [2, 4, 6]

// Filter only
RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 2]  // [3, 4, 5]
```

### List Predicates

```cypher
// ALL
WITH [1, 2, 3] as nums
RETURN ALL(x IN nums WHERE x > 0)  // true

// ANY
WITH [1, -1, 2] as nums
RETURN ANY(x IN nums WHERE x < 0)  // true

// NONE
WITH [1, 2, 3] as nums
RETURN NONE(x IN nums WHERE x < 0)  // true

// SINGLE
WITH [1, 2, 1] as nums
RETURN SINGLE(x IN nums WHERE x = 2)  // true
```

### Reducing Lists

```cypher
// Sum all elements
WITH [1, 2, 3, 4, 5] as nums
RETURN reduce(sum = 0, x IN nums | sum + x)  // 15

// Build string
WITH ['Alice', 'Bob', 'Charlie'] as names
RETURN reduce(s = '', name IN names | s + ', ' + name)  // ', Alice, Bob, Charlie'
```

## Maps

### Creating Maps

```cypher
// Literal map
RETURN {name: 'Alice', age: 30} as person

// Nested maps
RETURN {
  name: 'Alice',
  address: {city: 'NYC', zip: '10001'}
} as data
```

### Map Access

```cypher
WITH {name: 'Alice', age: 30} as person
RETURN person.name, person.age
// Can also use: person['name']
```

### Map Functions

```cypher
// keys() and values()
WITH {a: 1, b: 2, c: 3} as m
RETURN keys(m)    // ['a', 'b', 'c']
RETURN values(m)  // [1, 2, 3]

// Dynamic key access
WITH {name: 'Alice'} as p, 'name' as key
RETURN p[key]  // 'Alice'
```

## Working with Properties

### Dynamic Property Access

```cypher
// Get all property values
MATCH (p:Person)
RETURN p.name, [key IN keys(p) | p[key]] as allValues
```

### Converting to Map

```cypher
// Node to map
MATCH (p:Person {name: 'Alice'})
RETURN apoc.map.fromPairs([
  key IN keys(p) | [key, p[key]]
]) as personMap
```

## UNWIND

Expands lists into rows.

```cypher
// Create nodes from list
UNWIND ['Alice', 'Bob', 'Charlie'] as name
CREATE (p:Person {name: name})
```

```cypher
// Process list property
MATCH (p:Person)
UNWIND p.tags as tag
RETURN p.name, tag
```

```cypher
// With multiple properties
MATCH (p:Person)
UNWIND p.interests as interest
WITH p, interest
WHERE interest.category = 'tech'
RETURN p.name, interest.name
```
