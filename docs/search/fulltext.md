# Full-Text Search (FTS5)

GrafitoDB provides full-text search capabilities using SQLite's FTS5 extension.

## Prerequisites

FTS5 must be enabled in your SQLite build:

```python
from grafito import GrafitoDatabase

db = GrafitoDatabase(':memory:')

if not db.has_fts5():
    raise RuntimeError("FTS5 not available. Install SQLite with FTS5 support.")
```

## Creating Text Indexes

### Node Text Index

```python
# Index specific properties on nodes with a label
db.create_text_index(
    entity_type='node',
    label='Person',
    properties=['name', 'bio', 'description']
)
```

### Relationship Text Index

```python
# Index properties on relationships
db.create_text_index(
    entity_type='relationship',
    rel_type='COMMENT',
    properties=['text', 'title']
)
```

### Multiple Indexes

```python
# Create indexes for different entity types
db.create_text_index('node', 'Article', ['title', 'content', 'tags'])
db.create_text_index('node', 'Product', ['name', 'description'])
db.create_text_index('relationship', 'REVIEW', ['comment'])
```

## Building the Index

After configuring indexes, rebuild to index existing data:

```python
# Index all existing data
db.rebuild_text_index()

# Future inserts/updates are indexed automatically
```

## Searching

### Basic Search

```python
# Search across all indexed text
results = db.text_search('python developer', k=10)

for result in results:
    print(f"Score: {result.score}")
    print(f"Node: {result.node}")
    print(f"Matched properties: {result.matched_properties}")
```

### Filtered Search

```python
# Search only within specific labels
results = db.text_search(
    query='machine learning',
    k=10,
    labels=['Article', 'Paper']
)

# Search with property filter
results = db.text_search(
    query='database',
    k=10,
    labels=['Article'],
    properties={'status': 'published'}
)
```

### Search Results

```python
results = db.text_search('graph database', k=5)

for r in results:
    print(f"ID: {r.node.id}")
    print(f"Labels: {r.node.labels}")
    print(f"BM25 Score: {r.score}")
    print(f"Properties: {r.node.properties}")
```

## Query Syntax

### Basic Terms

```python
# Single term
db.text_search('python')

# Multiple terms (AND implied)
db.text_search('python graph')

# Exact phrase
db.text_search('"machine learning"')
```

### Boolean Operators

```python
# AND (default)
db.text_search('python AND graph')

# OR
db.text_search('python OR javascript')

# NOT
db.text_search('database NOT sql')

# Combined
db.text_search('(python OR javascript) AND web')
```

### Prefix Matching

```python
# Match words starting with 'dev'
db.text_search('dev*')

# Match 'graph' prefix
db.text_search('graph* database')
```

### NEAR Operator

```python
# Terms within 10 words of each other
db.text_search('python NEAR/10 database')

# Terms within 5 words
db.text_search('machine NEAR/5 learning')
```

## BM25 Ranking

FTS5 uses BM25 ranking by default. Scores are higher for better matches.

```python
results = db.text_search('python', k=10)

# Results sorted by BM25 score (descending)
for i, r in enumerate(results, 1):
    print(f"{i}. {r.node.properties['title']} (score: {r.score:.3f})")
```

## Integration with Cypher

```python
# Hybrid: Text search + graph pattern
results = db.text_search('database', k=20)
node_ids = [r.node.id for r in results]

# Then query relationships
cypher_results = db.execute("""
    MATCH (n)-[:AUTHORED]->(author:Person)
    WHERE id(n) IN $node_ids
    RETURN n.title, author.name
""", {'node_ids': node_ids})
```

## Best Practices

### 1. Choose Properties Wisely

```python
# Index meaningful text fields
db.create_text_index('node', 'Article', [
    'title',      # Good: descriptive
    'content',    # Good: searchable body
    # 'id',       # Skip: not meaningful for search
    # 'url',      # Skip: usually not helpful
])
```

### 2. Rebuild After Bulk Import

```python
# After importing large dataset
db.rebuild_text_index()
```

### 3. Combine with Property Filters

```python
# Narrow search space first
results = db.text_search(
    'python',
    k=10,
    labels=['Article'],
    properties={'published': True, 'language': 'en'}
)
```

### 4. Handle Empty Results

```python
results = db.text_search('xyzabc123', k=10)

if not results:
    print("No matches found")
    # Fall back to different query or show suggestions
```

## Limitations

1. **FTS5 Required**: Not available in all SQLite builds
2. **Tokenization**: Uses SQLite's default tokenizer (unicode61)
3. **No Fuzzy Matching**: Exact and prefix matches only
4. **Single Language**: Best for one language per index

## Troubleshooting

### No Results

```python
# Check if FTS5 is available
print(db.has_fts5())

# Check if index exists
print(db.list_text_indexes())

# Rebuild index
db.rebuild_text_index()
```

### Poor Ranking

```python
# BM25 scores can be negative (lower is better in FTS5)
# GrafitoDB normalizes to positive scores
# Check raw scores to debug
results = db.text_search('term', k=10)
for r in results:
    print(f"Score: {r.score}")
```
