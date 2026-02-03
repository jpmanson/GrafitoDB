# Semantic/Vector Search

GrafitoDB supports semantic search using vector embeddings and approximate nearest neighbor (ANN) search.

## Overview

Vector search allows you to:
- Find semantically similar content
- Implement recommendation systems
- Build RAG (Retrieval-Augmented Generation) pipelines
- Combine semantic and keyword search

## Creating Vector Indexes

### Basic FAISS Index

```python
# Create a flat (exact) index
db.create_vector_index(
    name='articles_vec',
    dim=384,                    # Embedding dimension
    backend='faiss',
    method='flat',
    options={'metric': 'l2'}    # L2 distance
)
```

### IVF Index (Approximate)

```python
# Faster search, approximate results
db.create_vector_index(
    name='articles_vec',
    dim=384,
    backend='faiss',
    method='ivf_flat',
    options={
        'metric': 'l2',
        'nlist': 100,      # Number of clusters
        'nprobe': 10       # Clusters to search
    }
)
```

### HNSW Index

```python
# Graph-based ANN (good balance)
db.create_vector_index(
    name='articles_vec',
    dim=384,
    backend='faiss',
    method='hnsw',
    options={
        'metric': 'l2',
        'hnsw_m': 16,           # Connections per node
        'ef_construction': 200,  # Build-time search depth
        'ef_search': 64          # Query-time search depth
    }
)
```

### Persistent Index

```python
# Save index to disk
db.create_vector_index(
    name='articles_vec',
    dim=384,
    backend='faiss',
    method='flat',
    options={
        'metric': 'l2',
        'index_path': '.grafito/indexes/articles.faiss'
    }
)
```

## Adding Embeddings

### Single Embedding

```python
# Get embedding from your model
embedding = model.encode("Python graph databases")  # [0.1, -0.2, ...]

# Upsert into index
db.upsert_embedding(
    node_id=article.id,
    vector=embedding.tolist(),
    index='articles_vec'
)
```

### Batch Upsert

```python
# Efficient batch insertion
with db:
    for article in articles:
        embedding = model.encode(article['content'])
        db.upsert_embedding(
            node_id=article['id'],
            vector=embedding.tolist(),
            index='articles_vec'
        )
```

### With Stored Vectors

```python
# Also store raw vectors in SQLite
db.create_vector_index(
    name='articles_vec',
    dim=384,
    backend='faiss',
    method='flat',
    options={'store_embeddings': True}  # Persist in SQLite
)
```

## Searching

### Basic Semantic Search

```python
# Encode query
query = "How to build graph applications"
query_vec = model.encode(query).tolist()

# Search
results = db.semantic_search(
    query_vector=query_vec,
    k=10,
    index='articles_vec'
)

for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"Title: {r.node.properties['title']}")
```

### Filtered Search

```python
# Search within specific labels
results = db.semantic_search(
    query_vector=query_vec,
    k=10,
    index='articles_vec',
    labels=['Article', 'Tutorial']
)

# Search with property filter
results = db.semantic_search(
    query_vector=query_vec,
    k=10,
    index='articles_vec',
    labels=['Article'],
    properties={'published': True}
)
```

### With Reranking

```python
# Use custom reranker
def my_reranker(query_vector, candidates):
    # candidates: [{"id": int, "vector": [...], "score": float, "node": Node}, ...]
    # Return re-ranked list
    return [{"id": c["id"], "score": c["score"] * 1.1} for c in candidates]

# Register and use
db.register_reranker('custom', my_reranker)
results = db.semantic_search(
    query_vector=query_vec,
    k=10,
    index='articles_vec',
    reranker='custom'
)
```

## Cypher Integration

### Vector Search Procedure

```python
results = db.execute("""
    CALL db.vector.search('articles_vec', $query_vec, 10, {labels: ['Article']})
    YIELD node, score
    RETURN node.title, score
""", {'query_vec': query_vec})
```

### Formatting Vectors for Cypher

```python
from grafito.cypher import format_vector_literal

# Format vector for Cypher query
vector_str = format_vector_literal(query_vec, precision=8)

cypher = f"""
    CALL db.vector.search('articles_vec', {vector_str}, 5)
    YIELD node, score
    RETURN node.title, score
"""

results = db.execute(cypher)
```

## Distance Metrics

| Metric | Use Case | Backend Support |
|--------|----------|-----------------|
| `l2` | Euclidean distance | All |
| `ip` | Inner product (for normalized vectors) | All |
| `cosine` | Cosine similarity | FAISS, usearch |

```python
# Cosine similarity (for normalized embeddings)
db.create_vector_index(
    name='articles_vec',
    dim=384,
    backend='faiss',
    method='flat',
    options={'metric': 'ip'}  # For normalized vectors
)
```

## Default k Values

```python
# Global default
db = GrafitoDatabase(':memory:', default_top_k=20)

# Per-index default
db.create_vector_index(
    name='articles_vec',
    dim=384,
    backend='faiss',
    method='flat',
    options={'metric': 'l2', 'default_k': 5}
)

# Uses index default (5)
results = db.semantic_search(query_vec, index='articles_vec')

# Override default
results = db.semantic_search(query_vec, k=10, index='articles_vec')
```

## Best Practices

### 1. Choose Right Backend

```python
# Small dataset (<10K): Brute force or flat
# Medium (10K-100K): IVF or HNSW
# Large (>100K): HNSW with persistence
```

### 2. Normalize for Cosine

```python
import numpy as np

# Normalize embeddings for cosine similarity
embedding = model.encode(text)
embedding = embedding / np.linalg.norm(embedding)

db.upsert_embedding(node_id, embedding.tolist(), index='articles_vec')
```

### 3. Batch Operations

```python
# Build index in batches
batch_size = 1000
for i in range(0, len(articles), batch_size):
    batch = articles[i:i+batch_size]
    with db:
        for article in batch:
            emb = model.encode(article['content'])
            db.upsert_embedding(article['id'], emb.tolist(), 'articles_vec')
```

### 4. Hybrid Search

```python
# Combine keyword + semantic
keyword_results = db.text_search('python graph', k=20)
semantic_results = db.semantic_search(query_vec, k=20, index='articles_vec')

# Merge and deduplicate
all_ids = set()
for r in keyword_results + semantic_results:
    all_ids.add(r.node.id)
```

## Troubleshooting

### Index Not Found

```python
# List available indexes
print(db.list_vector_indexes())
```

### Wrong Dimension

```python
# Check dimension mismatch
# Error: "Vector dimension 768 does not match index dimension 384"
# Solution: Create index with correct dimension or resize embeddings
```

### Empty Results

```python
# Check if embeddings exist
results = db.execute("SELECT COUNT(*) FROM vector_entries WHERE index_name = 'articles_vec'")

# Rebuild if needed
for node in db.match_nodes(labels=['Article']):
    emb = model.encode(node.properties['content'])
    db.upsert_embedding(node.id, emb.tolist(), 'articles_vec')
```
