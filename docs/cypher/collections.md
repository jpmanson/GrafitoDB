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

#### size()

Returns the number of elements in a list.

```cypher
// Size of list
RETURN size([1, 2, 3])  // 3
RETURN size([])         // 0

// Size of string (number of characters)
RETURN size('Hello')    // 5
```

#### head(), tail(), last()

```cypher
WITH [1, 2, 3, 4] as list
RETURN head(list)  // 1  (first element)
RETURN last(list)  // 4  (last element)
RETURN tail(list)  // [2, 3, 4]  (all except first)

// With empty list
RETURN head([])    // null
RETURN last([])    // null
RETURN tail([])    // []
```

#### reverse()

```cypher
RETURN reverse([1, 2, 3])     // [3, 2, 1]
RETURN reverse(['a', 'b'])    // ['b', 'a']
RETURN reverse([])            // []
```

#### range()

Generates a list of integers within a range.

```cypher
// Basic range (inclusive)
RETURN range(1, 5)            // [1, 2, 3, 4, 5]
RETURN range(0, 3)            // [0, 1, 2, 3]

// With step
RETURN range(0, 10, 2)        // [0, 2, 4, 6, 8, 10]
RETURN range(0, 9, 3)         // [0, 3, 6, 9]

// Negative step (descending)
RETURN range(5, 0, -1)        // [5, 4, 3, 2, 1, 0]
RETURN range(10, 0, -2)       // [10, 8, 6, 4, 2, 0]

// Single element
RETURN range(5, 5)            // [5]

// Step cannot be zero
// RETURN range(1, 5, 0)      // ERROR
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

## Property Access on Lists of Nodes

When you have a list of nodes or relationships, you can access a property directly on the list to get a list of property values:

```cypher
// Get names of all friends in one expression
MATCH (p:Person {name: 'Alice'})
WITH [(p)-[:KNOWS]->(f) | f] as friends
RETURN friends.name    // ['Bob', 'Charlie', 'David']

// Works with any property
MATCH (p:Person)
WITH collect(p) as persons
RETURN persons.age     // [25, 30, 35, ...]
```

This is equivalent to using a list comprehension:

```cypher
// These are equivalent:
WITH friends, friends.name as names1
WITH friends, [f IN friends | f.name] as names2
// names1 == names2
```

!!! note "Null handling"
    If any node in the list doesn't have the property, `null` is included in the result list.

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

#### keys()

Returns a list of all keys in a map, or all property names of a node/relationship.

```cypher
// Map keys
WITH {a: 1, b: 2, c: 3} as m
RETURN keys(m)    // ['a', 'b', 'c']

// Node property keys
MATCH (p:Person {name: 'Alice'})
RETURN keys(p)    // ['name', 'age', 'email', ...]

// Relationship property keys
MATCH ()-[r:KNOWS]->()
RETURN keys(r)    // ['since', ...]
```

#### values()

Returns a list of all values in a map, or all property values of a node/relationship.

```cypher
// Map values
WITH {a: 1, b: 2, c: 3} as m
RETURN values(m)  // [1, 2, 3]

// Node property values
MATCH (p:Person {name: 'Alice'})
RETURN values(p)  // ['Alice', 30, 'alice@example.com', ...]
```

#### Dynamic Key Access

```cypher
// Using variable as key
WITH {name: 'Alice'} as p, 'name' as key
RETURN p[key]  // 'Alice'

// Iterate over all properties
MATCH (p:Person)
WITH p, keys(p) as props
UNWIND props as key
RETURN p.name, key, p[key] as value
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
