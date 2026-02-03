# Temporal and Spatial Types

Working with time and space in Cypher.

## Temporal Types

### Date

```cypher
// Create date
RETURN date('2024-01-15') as d

// From components
RETURN date({year: 2024, month: 1, day: 15}) as d

// Current date
RETURN date() as today
```

### Time

```cypher
// Create time
RETURN time('14:30:00') as t
RETURN time('14:30:00.123') as t
RETURN time('14:30:00+01:00') as t  // with timezone

// Current time
RETURN time() as now
```

### LocalTime

Time without timezone information.

```cypher
// Create local time
RETURN localtime('14:30:00') as t
RETURN localtime('14:30:00.123') as t

// Current local time
RETURN localtime() as now

// From string with timezone (timezone is discarded)
RETURN localtime('14:30:00+05:00')  // 14:30:00
```

### DateTime

```cypher
// Full datetime
RETURN datetime('2024-01-15T14:30:00') as dt

// With timezone
RETURN datetime('2024-01-15T14:30:00Z') as dt  // UTC
RETURN datetime('2024-01-15T14:30:00+05:00') as dt

// Current datetime
RETURN datetime() as now
```

### LocalDateTime

DateTime without timezone information.

```cypher
// From ISO string (without timezone)
RETURN localdatetime('2024-01-15T14:30:00') as ldt

// Current local datetime
RETURN localdatetime() as now

// From string with timezone (timezone is discarded)
RETURN localdatetime('2024-01-15T14:30:00+05:00')  // 2024-01-15T14:30:00
```

!!! note "Timezone handling"
    When parsing a string with timezone using `localdatetime()`, the timezone offset is discarded and the local time is preserved. Use `datetime()` if you need to preserve timezone information.

### Duration

```cypher
// ISO duration format
RETURN duration('P1Y2M3DT4H5M6S') as dur  // 1 year, 2 months, 3 days, 4 hours...
RETURN duration('P1D') as oneDay
RETURN duration('PT2H30M') as twoHalfHours

// From components
RETURN duration({days: 1, hours: 2}) as dur
```

## Temporal Operations

### Comparisons

```cypher
// Compare dates
WITH date('2024-01-15') as d1, date('2024-01-20') as d2
RETURN d1 < d2  // true

// Range queries
MATCH (p:Person)
WHERE p.birthday >= date('1990-01-01')
  AND p.birthday <= date('1999-12-31')
RETURN p.name
```

### Arithmetic

```cypher
// Add duration to date
RETURN date('2024-01-15') + duration('P1M')  // 2024-02-15

// Subtract from datetime
RETURN datetime() - duration('P7D')  // 1 week ago

// Duration between dates
RETURN duration.between(
  date('2024-01-01'),
  date('2024-01-15')
)  // P14D
```

### Truncation

```cypher
// Truncate to year/month/day
RETURN date.truncate('year', date('2024-06-15'))   // 2024-01-01
RETURN date.truncate('month', date('2024-06-15'))  // 2024-06-01

// Truncate datetime
RETURN datetime.truncate('hour', datetime())  // Current hour, minute=0, second=0
```

### Accessing Components

```cypher
WITH date('2024-01-15') as d
RETURN d.year   // 2024
RETURN d.month  // 1
RETURN d.day    // 15
RETURN d.week   // 3 (week of year)
RETURN d.dayOfWeek  // 1 (Monday)
RETURN d.quarter    // 1
```

### Converting Between Types

```cypher
// Date to datetime
RETURN datetime(date('2024-01-15'))  // 2024-01-15T00:00:00

// Datetime to date
RETURN date(datetime('2024-01-15T14:30:00'))  // 2024-01-15

// String to temporal
RETURN date('2024-01-15')
RETURN time('14:30:00')
```

## Spatial Types

### Point (2D)

```cypher
// Create point with x, y
RETURN point({x: 10, y: 20}) as p

// Create geographic point
RETURN point({longitude: -74.006, latitude: 40.7128}) as nyc
```

### Distance

```cypher
// Euclidean distance
WITH point({x: 0, y: 0}) as p1, point({x: 3, y: 4}) as p2
RETURN distance(p1, p2)  // 5.0

// Geographic distance (in meters)
WITH
  point({longitude: -74.006, latitude: 40.7128}) as nyc,
  point({longitude: -118.2437, latitude: 34.0522}) as la
RETURN distance(nyc, la) as meters
```

### Accessing Components

```cypher
WITH point({x: 10, y: 20}) as p
RETURN p.x, p.y

WITH point({longitude: -74.006, latitude: 40.7128}) as p
RETURN p.longitude, p.latitude
```

## Common Use Cases

### Age Calculation

```cypher
MATCH (p:Person)
RETURN p.name,
       duration.between(p.birthday, date()).years as age
```

### Recent Activity

```cypher
// Users active in last 7 days
MATCH (u:User)
WHERE u.lastLogin > datetime() - duration('P7D')
RETURN u.name
```

### Geographic Search

```cypher
// Find locations within radius
MATCH (store:Store)
WITH store, point({longitude: store.lon, latitude: store.lat}) as storePoint
WITH store, storePoint,
     point({longitude: -74.006, latitude: 40.7128}) as center
WHERE distance(storePoint, center) < 5000  // within 5km
RETURN store.name
```

### Time-Based Aggregation

```cypher
// Group by month
MATCH (o:Order)
WITH date.truncate('month', o.createdAt) as month, count(*) as orders
RETURN month, orders
ORDER BY month
```
