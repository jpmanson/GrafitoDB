# String Functions

String manipulation functions in Cypher.

## Type Conversion (Casting)

### toInteger()

Converts a value to an integer.

```cypher
// From string
RETURN toInteger('42')        // 42
RETURN toInteger('3.14')      // 3 (truncates)

// From float
RETURN toInteger(3.9)         // 3 (truncates)

// From boolean
RETURN toInteger(true)        // 1
RETURN toInteger(false)       // 0

// Null handling
RETURN toInteger(null)        // null
```

### toFloat()

Converts a value to a float.

```cypher
// From string
RETURN toFloat('3.14')        // 3.14
RETURN toFloat('2')           // 2.0

// From integer
RETURN toFloat(42)            // 42.0

// From boolean
RETURN toFloat(true)          // 1.0
RETURN toFloat(false)         // 0.0

// Null handling
RETURN toFloat(null)          // null
```

### toString()

Converts any value to a string.

```cypher
// From number
RETURN toString(42)           // '42'
RETURN toString(3.14)         // '3.14'

// From boolean
RETURN toString(true)         // 'true'
RETURN toString(false)        // 'false'

// From temporal types
RETURN toString(date())       // '2024-01-15'
RETURN toString(datetime())   // '2024-01-15T14:30:00.123456'

// Null handling
RETURN toString(null)         // 'null'
```

## Case Conversion

### toUpper()

```cypher
// Convert to uppercase
RETURN toUpper('hello')  // 'HELLO'

// With null
RETURN toUpper(null)  // null
```

### toLower()

```cypher
// Convert to lowercase
RETURN toLower('HELLO')  // 'hello'

// Mixed case
RETURN toLower('Hello World')  // 'hello world'
```

## Trimming

### trim()

```cypher
// Remove leading and trailing whitespace
RETURN trim('  hello  ')  // 'hello'
RETURN trim('\thello\n')  // 'hello'
```

## Substring Extraction

### substring()

```cypher
// substring(text, start, length?)

// From position 0, length 3
RETURN substring('Hello', 0, 3)  // 'Hel'

// From position 1, length 3
RETURN substring('Hello', 1, 3)  // 'ell'

// From position to end (no length)
RETURN substring('Hello', 2)  // 'llo'

// Negative start is not allowed
// substring('Hello', -1)  // ERROR
```

## String Splitting

### split()

```cypher
// Split by delimiter
RETURN split('a,b,c', ',')  // ['a', 'b', 'c']

// Split by space
RETURN split('hello world', ' ')  // ['hello', 'world']

// Split with empty strings
RETURN split('a,,c', ',')  // ['a', '', 'c']

// Wrong type raises error
// RETURN split(123, ',')  // CypherExecutionError
```

## Regular Expressions

### regex()

Returns boolean if pattern matches anywhere in string.

```cypher
// Simple match
RETURN regex('abc', 'a.c')  // true

// Any character
RETURN regex('hello', 'h.*o')  // true

// Character class
RETURN regex('abc', '[aeiou]')  // true

// No match
RETURN regex('abc', '^d')  // false

// Invalid pattern raises error
// RETURN regex('abc', '[')  // CypherExecutionError
```

### matches()

Returns boolean for full string match (implied ^ and $).

```cypher
// Full match
RETURN matches('abc', 'abc')  // true

// Pattern must match entire string
RETURN matches('abc', 'a')  // false
RETURN matches('abc', 'a.*')  // true

// Digit pattern
RETURN matches('123', '\\d+')  // true
```

## String Concatenation

### Using + Operator

```cypher
// Concatenate with +
WITH 'Hello' as a, 'World' as b
RETURN a + ' ' + b  // 'Hello World'

// With numbers (converted to string)
RETURN 'Count: ' + 42  // 'Count: 42'
```

### apoc.text.join()

```cypher
// Join list with separator
RETURN apoc.text.join(['a', 'b', 'c'], ',')  // 'a,b,c'
RETURN apoc.text.join(['2024', '01', '15'], '-')  // '2024-01-15'
```

## Text Cleanup Functions

### deaccent()

Remove accents from characters.

```cypher
// Remove accents
RETURN deaccent('café')  // 'cafe'
RETURN deaccent('naïve')  // 'naive'
RETURN deaccent('résumé')  // 'resume'

// No accents = no change
RETURN deaccent('hello')  // 'hello'

// Null handling
RETURN deaccent(null)  // null
```

### strip_html()

Remove HTML tags from text.

```cypher
// Strip HTML tags
RETURN strip_html('<p>Hello <b>World</b></p>')  // 'Hello World'
RETURN strip_html('<div>Text</div>')  // 'Text'

// Already plain text = no change
RETURN strip_html('plain text')  // 'plain text'

// Null handling
RETURN strip_html(null)  // null
```

### strip_emoji()

Remove emoji characters from text.

```cypher
// Remove emojis
RETURN strip_emoji('Hello 😀 World 🌍')  // 'Hello  World '
RETURN strip_emoji('Great job 👍👍')  // 'Great job '

// No emojis = no change
RETURN strip_emoji('Hello World')  // 'Hello World'

// Null handling
RETURN strip_emoji(null)  // null
```

### snake_case()

Convert text to snake_case format.

```cypher
// Convert to snake_case
RETURN snake_case('Hello World')  // 'hello_world'
RETURN snake_case('helloWorld')  // 'hello_world'
RETURN snake_case('HelloWorld')  // 'hello_world'
RETURN snake_case('hello-world')  // 'hello_world'
RETURN snake_case('hello_world')  // 'hello_world'

// Multiple words
RETURN snake_case('The Quick Brown Fox')  // 'the_quick_brown_fox'

// Null handling
RETURN snake_case(null)  // null
```

## Text Similarity Functions

### levenshtein()

Calculate Levenshtein (edit) distance between two strings.

```cypher
// Exact match
RETURN levenshtein('hello', 'hello')  // 0

// One character different
RETURN levenshtein('hello', 'hallo')  // 1

// Two edits needed
RETURN levenshtein('kitten', 'sitting')  // 3

// Case sensitive
RETURN levenshtein('Hello', 'hello')  // 1

// Use with deaccent for fuzzy matching
RETURN levenshtein(deaccent('café'), 'cafe')  // 0

// Null handling
RETURN levenshtein(null, 'hello')  // null
RETURN levenshtein('hello', null)  // null
```

Common use case - fuzzy search:

```cypher
// Find similar names
MATCH (p:Person)
WHERE levenshtein(deaccent(p.name), 'john') <= 2
RETURN p.name
```

### jaccard()

Calculate Jaccard similarity coefficient between two strings (based on character bigrams).

```cypher
// Similar strings
RETURN jaccard('hello', 'hello')  // 1.0
RETURN jaccard('hello', 'hallo')  // ~0.6

// Different strings
RETURN jaccard('hello', 'world')  // 0.0

// Partial similarity
RETURN jaccard('night', 'nacht')  // ~0.3

// Null handling
RETURN jaccard(null, 'hello')  // null
```

Use for similarity matching:

```cypher
// Find potential duplicates
MATCH (p1:Person), (p2:Person)
WHERE p1.id < p2.id
  AND jaccard(p1.name, p2.name) > 0.8
RETURN p1.name, p2.name, jaccard(p1.name, p2.name) as similarity
```

## String Inspection

### size() with Strings

```cypher
// String length
RETURN size('Hello')  // 5
RETURN size('')  // 0
```

### starts WITH / ends WITH / contains

In WHERE clause:

```cypher
// Starts with
MATCH (p:Person)
WHERE p.name STARTS WITH 'Al'
RETURN p.name

// Ends with
MATCH (p:Person)
WHERE p.email ENDS WITH '@company.com'
RETURN p.name

// Contains
MATCH (p:Person)
WHERE p.bio CONTAINS 'engineer'
RETURN p.name
```

## APOC String Functions

### Replace

```cypher
RETURN apoc.text.replace('hello-world', '-', '_')  // 'hello_world'
```

## Error Handling

All string functions return `null` for `null` input:

```cypher
RETURN toUpper(null)      // null
RETURN trim(null)         // null
RETURN substring(null, 0) // null
RETURN split(null, ',')   // null
RETURN deaccent(null)     // null
RETURN strip_html(null)   // null
RETURN strip_emoji(null)  // null
RETURN snake_case(null)   // null
RETURN levenshtein(null, 'a')  // null
RETURN jaccard(null, 'a')    // null
```

Invalid argument types raise `CypherExecutionError`:

```cypher
// These raise errors:
RETURN substring('abc', 0, -1)  // Negative length
RETURN split(123, ',')          // Number instead of string
RETURN regex('abc', '[')        // Invalid regex pattern
```

## Common Use Cases

### Normalizing Input

```cypher
// Clean user input for search
WITH '  John DOE  ' as raw
RETURN snake_case(deaccent(trim(raw))) as clean
// 'john_doe'
```

### Email Domain Extraction

```cypher
WITH 'alice@company.com' as email
RETURN split(email, '@')[1] as domain
// 'company.com'
```

### Name Formatting

```cypher
// Format: Last, First
MATCH (p:Person)
RETURN substring(p.lastName, 0, 1) + '. ' + p.firstName as display
```

### Fuzzy Matching

```cypher
// Find similar names with multiple strategies
MATCH (p:Person)
WITH p, deaccent(p.name) as normalized
WHERE levenshtein(normalized, 'johnson') <= 2
   OR jaccard(normalized, 'johnson') > 0.7
RETURN p.name
```

### Content Cleaning

```cypher
// Clean article content for indexing
MATCH (a:Article)
SET a.clean_content = strip_emoji(strip_html(a.raw_content))
```

### Duplicate Detection

```cypher
// Find potential duplicate companies
MATCH (c1:Company), (c2:Company)
WHERE c1.id < c2.id
  AND jaccard(
    snake_case(deaccent(c1.name)),
    snake_case(deaccent(c2.name))
  ) > 0.85
RETURN c1.name, c2.name
```
