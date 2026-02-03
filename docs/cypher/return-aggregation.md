# RETURN and Aggregation

Transform and aggregate query results.

## RETURN Clause

### Returning Nodes

```cypher
MATCH (p:Person)
RETURN p
```

### Returning Properties

```cypher
// Single property
MATCH (p:Person)
RETURN p.name

// Multiple properties
MATCH (p:Person)
RETURN p.name, p.age, p.email
```

### Aliases

```cypher
// Using AS keyword
MATCH (p:Person)
RETURN p.name AS personName, p.age AS years

// Implicit alias (without AS - not recommended)
MATCH (p:Person)
RETURN p.name personName
```

### Distinct Results

```cypher
// Remove duplicates
MATCH (p:Person)-[:KNOWS]->(friend)
RETURN DISTINCT friend.name
```

### Expressions

```cypher
// Computed values
MATCH (p:Person)
RETURN p.name, p.age + 1 AS nextYearAge

// String concatenation
MATCH (p:Person)
RETURN p.firstName + ' ' + p.lastName AS fullName
```

## Aggregation Functions

### COUNT

```cypher
// Count all matches
MATCH (p:Person)
RETURN COUNT(p) AS personCount

// Count non-NULL values
MATCH (p:Person)
RETURN COUNT(p.email) AS withEmail

// Count all rows (including NULLs)
MATCH (p:Person)
RETURN COUNT(*) AS total

// Count distinct
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN COUNT(DISTINCT c.name) AS companyCount
```

### SUM

```cypher
// Total of all ages
MATCH (p:Person)
RETURN SUM(p.age) AS totalAge

// Total salary by department
MATCH (e:Employee)
RETURN e.department, SUM(e.salary) AS totalSalary
```

### AVG

```cypher
// Average age
MATCH (p:Person)
RETURN AVG(p.age) AS averageAge

// Average by group
MATCH (p:Person)
RETURN p.city, AVG(p.age) AS avgAge
```

### MIN / MAX

```cypher
// Minimum and maximum
MATCH (p:Person)
RETURN MIN(p.age) AS youngest, MAX(p.age) AS oldest

// By city
MATCH (p:Person)
RETURN p.city, MIN(p.age), MAX(p.age)
```

### COLLECT

```cypher
// Collect into list
MATCH (p:Person)
RETURN COLLECT(p.name) AS allNames

// Collect with filter
MATCH (p:Person)
WHERE p.city = 'NYC'
RETURN COLLECT(p.name) AS nyNames

// Nested collect
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name, COLLECT(p.name) AS employees
```

### Standard Deviation

```cypher
// Sample standard deviation
MATCH (p:Person)
RETURN stdDev(p.age) AS ageStdDev

// Population standard deviation
MATCH (p:Person)
RETURN stdDevP(p.age) AS agePopulationStdDev
```

### Percentiles

```cypher
// Continuous percentile (interpolated)
MATCH (p:Person)
RETURN percentileCont(p.age, 0.5) AS medianAge

// Discrete percentile (actual value)
MATCH (p:Person)
RETURN percentileDisc(p.age, 0.9) AS p90Age
```

## Grouping Results

### Implicit Grouping

```cypher
// Group by company
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name, AVG(p.age) AS avgAge

// Multiple group keys
MATCH (p:Person)
RETURN p.city, p.department, COUNT(*) AS count
```

### WITH for Pre-aggregation

```cypher
// Filter after aggregation
MATCH (p:Person)
WITH p.city AS city, COUNT(*) AS cityCount
WHERE cityCount > 10
RETURN city, cityCount
```

## Advanced Return Patterns

### Conditional Values

```cypher
// Using CASE
MATCH (p:Person)
RETURN p.name,
CASE
  WHEN p.age < 18 THEN 'minor'
  WHEN p.age < 65 THEN 'adult'
  ELSE 'senior'
END AS category
```

### Complex Expressions

```cypher
// Computed columns
MATCH (p:Person)
RETURN
  p.name,
  p.age,
  p.age * 12 AS ageInMonths,
  date().year - p.birthYear AS calculatedAge
```

### Returning Maps

```cypher
// Create map from properties
MATCH (p:Person)
RETURN {
  name: p.name,
  age: p.age,
  city: p.city
} AS personMap
```

### Returning Paths

```cypher
// Return full path
MATCH p = (a:Person)-[:KNOWS*1..3]->(b:Person)
RETURN p, length(p) AS hops

// Path nodes and relationships
MATCH p = (a:Person)-[:KNOWS]->(b:Person)
RETURN nodes(p) AS pathNodes, relationships(p) AS pathRels
```

### Path Functions

#### nodes()

Returns a list of all nodes in a path.

```cypher
MATCH p = (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b:Person)
RETURN nodes(p) as pathNodes
// [(:Person {name: 'Alice'}), (:Person {...}), ...]

// Count nodes in path
MATCH p = shortestPath((a)-[:KNOWS*]-(b))
RETURN size(nodes(p)) as nodeCount
```

#### relationships()

Returns a list of all relationships in a path.

```cypher
MATCH p = (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b:Person)
RETURN relationships(p) as pathRels
// [(:KNOWS {...}), (:KNOWS {...}), ...]

// Get relationship types
MATCH p = (a)-[*1..3]-(b)
RETURN [r IN relationships(p) | type(r)] as relTypes
```

#### length()

Returns the number of relationships in a path.

```cypher
MATCH p = (a)-[:KNOWS*]->(b)
RETURN length(p) as hops
```

## Common Use Cases

### Dashboard Metrics

```cypher
// User statistics
MATCH (u:User)
RETURN
  COUNT(*) AS totalUsers,
  COUNT(u.verifiedEmail) AS verifiedUsers,
  AVG(u.loginCount) AS avgLogins,
  MAX(u.lastLogin) AS mostRecentLogin
```

### Reports

```cypher
// Monthly signups
MATCH (u:User)
WITH date.truncate('month', u.createdAt) AS month, COUNT(*) AS signups
RETURN month, signups
ORDER BY month
```

### Network Analysis

```cypher
// Most connected people
MATCH (p:Person)
WITH p, COUNT { (p)-[:KNOWS]-() } AS connections
RETURN p.name, connections
ORDER BY connections DESC
LIMIT 10
```
