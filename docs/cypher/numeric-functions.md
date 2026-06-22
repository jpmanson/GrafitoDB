# Numeric Functions

Grafito's Cypher supports the standard mathematical functions and operators.
All functions return `null` when given a `null` argument.

## Arithmetic Operators

| Operator | Meaning | Example | Result |
|----------|---------|---------|--------|
| `+` | Addition | `RETURN 2 + 3 AS r` | `5` |
| `-` | Subtraction | `RETURN 5 - 2 AS r` | `3` |
| `*` | Multiplication | `RETURN 4 * 3 AS r` | `12` |
| `/` | Division | `RETURN 10 / 3 AS r` | `3` |
| `%` | Modulo | `RETURN 10 % 3 AS r` | `1` |
| `^` | Exponentiation | `RETURN 2 ^ 10 AS r` | `1024.0` |

Division of two integers performs **integer division** (`10 / 3` is `3`); if
either operand is a float the result is a float (`10.0 / 4` is `2.5`).
Exponentiation always returns a float and is right-associative
(`2 ^ 2 ^ 3` is `2 ^ 8` = `256.0`).

```cypher
RETURN 2 + 3 * 4 AS r        // 14  (multiplication binds tighter)
RETURN (2 + 3) * 4 AS r      // 20
```

Dividing or taking the modulo by zero raises an error.

## Rounding and Sign

### abs()

Absolute value.

```cypher
RETURN abs(-5) AS r   // 5
```

### ceil() / floor() / round()

`ceil()` rounds up, `floor()` rounds down, and `round()` rounds half **away from
zero** (so `round(2.5)` is `3.0` and `round(-2.5)` is `-3.0`). All return floats.

```cypher
RETURN ceil(1.2) AS r    // 2.0
RETURN floor(1.8) AS r   // 1.0
RETURN round(2.5) AS r   // 3.0
```

### sign()

Returns `-1`, `0`, or `1` for negative, zero, or positive numbers.

```cypher
RETURN sign(-3) AS r   // -1
```

## Powers, Roots and Logarithms

### sqrt()

Square root. Raises an error for negative arguments.

```cypher
RETURN sqrt(9) AS r   // 3.0
```

### exp() / log() / log10()

`exp(x)` is e raised to `x`; `log(x)` is the natural logarithm; `log10(x)` is the
base-10 logarithm. `log()` and `log10()` require a positive argument.

```cypher
RETURN exp(1) AS r     // 2.718281828459045
RETURN log10(1000) AS r  // 3.0
```

## Trigonometry

`sin()`, `cos()`, `tan()`, `cot()`, `asin()`, `acos()`, `atan()`, and
`atan2(y, x)` operate in radians. `asin()` and `acos()` require an argument in
`[-1, 1]`. `degrees()` and `radians()` convert between units, and `haversin()`
returns `(1 - cos(x)) / 2`.

```cypher
RETURN sin(0) AS r           // 0.0
RETURN asin(1) AS r          // 1.5707963267948966
RETURN atan2(1, 1) AS r      // 0.7853981633974483
RETURN degrees(pi()) AS r    // 180.0
RETURN radians(180) AS r     // 3.141592653589793
```

## Constants and Randomness

### e() / pi()

```cypher
RETURN e() AS r    // 2.718281828459045
RETURN pi() AS r   // 3.141592653589793
```

### rand()

Returns a random float in `[0, 1)`.

```cypher
RETURN rand() AS r
```

## Numeric Casting

`toInteger()` and `toFloat()` convert numbers and numeric strings; see
[String Functions](string-functions.md#type-conversion-casting) for details.
For element-wise list casting, `toIntegerList()` / `toFloatList()` turn each
convertible element and leave non-convertible ones as `null`.

```cypher
RETURN toInteger('42') AS r          // 42
RETURN toIntegerList(['1', '2', 'x']) AS r  // [1, 2, null]
```

## Null Propagation

Every numeric function returns `null` when any argument is `null`, so they
compose safely over optional data:

```cypher
MATCH (n:Product)
RETURN n.name AS name, round(sqrt(n.score)) AS rating
```
