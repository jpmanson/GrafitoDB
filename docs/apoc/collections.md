# APOC: Collections, Maps, Text, Utilities

GrafitoDB implements a small, useful subset of APOC functions for text, maps,
and collections. These functions can be used in any Cypher query.

## Text

### apoc.text.join

Joins a list of values into a string with a delimiter.

```cypher
// Basic join
RETURN apoc.text.join(['a', 'b', 'c'], '-')     // 'a-b-c'
RETURN apoc.text.join(['2024', '01', '15'], '/') // '2024/01/15'

// Empty list
RETURN apoc.text.join([], ',')                  // ''

// Single element
RETURN apoc.text.join(['hello'], ',')           // 'hello'

// Null in list returns null
RETURN apoc.text.join(['a', null, 'c'], ',')    // null

// Null inputs
RETURN apoc.text.join(null, ',')                // null
RETURN apoc.text.join(['a', 'b'], null)         // null
```

### apoc.text.split

Splits a string using regex pattern.

```cypher
// Simple split
RETURN apoc.text.split('a,b,c', ',')            // ['a', 'b', 'c']

// Regex pattern
RETURN apoc.text.split('a1b2c3', r'\d+')        // ['a', 'b', 'c']

// Multiple delimiters
RETURN apoc.text.split('a;b|c', r'[;|]')        // ['a', 'b', 'c']

// Null inputs
RETURN apoc.text.split(null, ',')               // null
RETURN apoc.text.split('a,b', null)             // null

// Invalid regex
// RETURN apoc.text.split('abc', '[')           // ERROR - invalid regex
```

### apoc.text.replace

Replaces occurrences matching a regex pattern.

```cypher
// Simple replacement
RETURN apoc.text.replace('a-b-c', '-', '_')     // 'a_b_c'

// Regex pattern
RETURN apoc.text.replace('hello123world', r'\d+', ' ')
// 'hello world'

// Remove matches (replace with empty)
RETURN apoc.text.replace('a1b2c3', r'\d', '')   // 'abc'

// Null inputs
RETURN apoc.text.replace(null, '-', '_')        // null
RETURN apoc.text.replace('abc', null, '_')      // null
RETURN apoc.text.replace('abc', '-', null)      // null

// Invalid regex
// RETURN apoc.text.replace('abc', '[', '_')    // ERROR - invalid regex
```

## Maps

### apoc.map.merge

Merges two maps. Values from the second map overwrite values from the first.

```cypher
// Basic merge
RETURN apoc.map.merge({a: 1}, {b: 2})
// {a: 1, b: 2}

// Overlapping keys (second wins)
RETURN apoc.map.merge({a: 1, b: 2}, {b: 3, c: 4})
// {a: 1, b: 3, c: 4}

// Nested maps are not deep-merged
RETURN apoc.map.merge({a: {x: 1}}, {a: {y: 2}})
// {a: {y: 2}}

// Null inputs
RETURN apoc.map.merge(null, {a: 1})           // null
RETURN apoc.map.merge({a: 1}, null)           // null
```

### apoc.map.clean

Removes specified keys and values from a map.

```cypher
// Remove keys
RETURN apoc.map.clean({a: 1, b: 2, c: 3}, ['a', 'b'], [])
// {c: 3}

// Remove values
RETURN apoc.map.clean({a: 1, b: null, c: 'x'}, [], [null])
// {a: 1, c: 'x'}

// Remove both keys and values
RETURN apoc.map.clean({a: 1, b: null, c: 'x'}, ['a'], [null])
// {c: 'x'}

// Empty remove lists
RETURN apoc.map.clean({a: 1, b: 2}, [], [])   // {a: 1, b: 2}

// Null inputs
RETURN apoc.map.clean(null, ['a'], [])        // null
```

### apoc.map.fromPairs

Creates a map from a list of [key, value] pairs.

```cypher
// Basic usage
RETURN apoc.map.fromPairs([['a', 1], ['b', 2]])
// {a: 1, b: 2}

// Empty list
RETURN apoc.map.fromPairs([])                 // {}

// Duplicate keys (last wins)
RETURN apoc.map.fromPairs([['a', 1], ['a', 2]])
// {a: 2}

// Null input
RETURN apoc.map.fromPairs(null)               // null
```

### apoc.map.removeKey

Returns a new map with the specified key removed.

```cypher
// Remove existing key
RETURN apoc.map.removeKey({a: 1, b: 2}, 'a')
// {b: 2}

// Remove non-existing key (no error)
RETURN apoc.map.removeKey({a: 1, b: 2}, 'c')
// {a: 1, b: 2}

// Null inputs
RETURN apoc.map.removeKey(null, 'a')          // null
RETURN apoc.map.removeKey({a: 1}, null)       // null
```

### apoc.map.get

Gets a value from a map with an optional default.

```cypher
// Key exists
RETURN apoc.map.get({a: 1, b: 2}, 'a')        // 1

// Key not found (no default)
RETURN apoc.map.get({a: 1, b: 2}, 'c')        // null

// Key not found (with default)
RETURN apoc.map.get({a: 1, b: 2}, 'c', 0)     // 0
RETURN apoc.map.get({a: 1}, 'b', 'default')   // 'default'

// Null inputs
RETURN apoc.map.get(null, 'a')                // null (default not returned)
RETURN apoc.map.get({a: 1}, null)             // null

## Convert

### apoc.convert.toMap

Converts various inputs to a map.

```cypher
// From list of pairs
RETURN apoc.convert.toMap([['a', 1], ['b', 2]])
// {a: 1, b: 2}

// From map (returns copy)
RETURN apoc.convert.toMap({a: 1, b: 2})
// {a: 1, b: 2}

// From node (uses properties)
MATCH (p:Person {name: 'Alice'})
RETURN apoc.convert.toMap(p)
// {name: 'Alice', age: 30, ...}

// From relationship
MATCH ()-[r:KNOWS]->()
RETURN apoc.convert.toMap(r)
// {since: 2020, ...}

// From empty list
RETURN apoc.convert.toMap([])                 // {}

// Null input
RETURN apoc.convert.toMap(null)               // null

// Invalid input
// RETURN apoc.convert.toMap([1, 2, 3])       // ERROR - not pairs
// RETURN apoc.convert.toMap([['a']])         // ERROR - pair too short
```

## Collections

### apoc.coll.contains

Checks if a value exists in a list. Returns `null` if the list contains `null` elements and the target is not found.

```cypher
// Value exists
RETURN apoc.coll.contains([1, 2, 3], 2)       // true

// Value not found
RETURN apoc.coll.contains([1, 2, 3], 5)       // false

// List contains null
RETURN apoc.coll.contains([1, 2, null], 5)    // null (unknown if might exist)
RETURN apoc.coll.contains([1, 2, null], 2)    // true

// Null inputs
RETURN apoc.coll.contains(null, 2)            // null
RETURN apoc.coll.contains([1, 2, 3], null)    // null
```

### apoc.coll.toSet

Returns a list of unique values, preserving the original order (first occurrence kept).

```cypher
// Basic deduplication
RETURN apoc.coll.toSet([1, 2, 2, 3])          // [1, 2, 3]
RETURN apoc.coll.toSet(['a', 'b', 'a', 'c'])  // ['a', 'b', 'c']

// Mixed types
RETURN apoc.coll.toSet([1, '1', 1, '1'])      // [1, '1']

// Empty list
RETURN apoc.coll.toSet([])                    // []

// Null input
RETURN apoc.coll.toSet(null)                  // null
```

## Utilities

### apoc.util.compress

Compresses a string or bytes using various compression algorithms.

```cypher
// Default compression (DEFLATE)
RETURN apoc.util.compress('hello world')
// b'x\x9c\xf3H\xcd\xc9\xc9W\x08\xcf\xcf\x00\x00\x12\x8b\x04\x1d'

// GZIP compression
RETURN apoc.util.compress('hello', {compression: 'GZIP'})

// BZ2 compression
RETURN apoc.util.compress('hello', {compression: 'BZ2'})

// XZ/LZMA compression
RETURN apoc.util.compress('hello', {compression: 'XZ'})

// From bytes
RETURN apoc.util.compress(b'hello')

// Null input
RETURN apoc.util.compress(null)               // null
```

Supported algorithms: `DEFLATE` (default), `GZIP`, `BZ2`, `XZ` (or `LZMA`).

!!! note "Case insensitive"
    Compression names are case-insensitive: `gzip`, `GZIP`, `Gzip` all work.


## Notes and Limitations

### List Comprehensions vs filter()/extract()

Some APOC functions like `filter()` and `extract()` are **not available** as function calls because GrafitoDB implements them as native list comprehensions:

```cypher
// Instead of filter(list, condition):
[x IN list WHERE condition]

// Instead of extract(list, expression):
[x IN list | expression]

// Combined:
[x IN list WHERE x > 0 | x * 2]
```

### reduce() Function

`reduce()` is also implemented as a native Cypher expression, not an APOC function:

```cypher
// Native Cypher reduce
RETURN reduce(sum = 0, x IN [1, 2, 3] | sum + x)  // 6
```

See the [Collections documentation](../cypher/collections.md) for full details on list operations.
