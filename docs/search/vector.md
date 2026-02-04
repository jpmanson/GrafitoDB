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

## Similarity Metrics

Different metrics measure similarity in different ways:

| Metric | Range | Interpretation | Best For |
|--------|-------|----------------|----------|
| **Cosine Similarity** | [-1, 1] | 1 = identical direction, 0 = orthogonal, -1 = opposite | Text embeddings, normalized vectors |
| **L2 Distance** | [0, ∞) | 0 = identical, larger = more different | General purpose, spatial data |
| **Inner Product** | (-∞, ∞) | Higher = more similar | Normalized vectors (equivalent to cosine) |

### Cosine Similarity (Recommended for Text)

Best for text embeddings and semantic search:

```python
# Option 1: HNSWlib with cosine (via indexer)
from grafito.indexers import HNSWlibIndexer

indexer = HNSWlibIndexer(
    options={"metric": "cosine"},
    embedding_function=embedder
)
db.create_vector_index("my_index", indexer=indexer)

# Option 2: Annoy with angular (same as cosine)
from grafito.indexers import AnnoyIndexer

indexer = AnnoyIndexer(
    options={"metric": "angular"},
    embedding_function=embedder
)

# Option 3: FAISS with inner product + normalized embeddings
embedder = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    normalize_embeddings=True  # Normalize to unit length
)
indexer = FAISSIndexer(
    method="flat",
    options={"metric": "ip"},  # Inner product with normalized = cosine
    embedding_function=embedder
)
```

**Why cosine for text?**
- Focuses on direction, not magnitude
- Robust to document length differences
- Standard in NLP and semantic search

### L2 Distance

Euclidean distance, good for spatial embeddings:

```python
from grafito.indexers import FAISSIndexer

indexer = FAISSIndexer(
    method="flat",
    options={"metric": "l2"},
    embedding_function=embedder
)
```

**When to use:**
- Image embeddings
- Spatial data
- When magnitude matters

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

## Advanced Features

### 1. String Queries (Auto-Embedding)

Pass strings directly instead of vectors when the index has an embedding function:

```python
# String query - automatically embedded
results = db.semantic_search(
    "machine learning algorithms",  # String, not vector!
    k=5,
    index="docs_vec"
)
```

**Requirements**: The vector index must have an associated embedding function.

### 2. Built-in Reranking

Improve precision with exact reranking of candidate results:

```python
# Get more candidates, then rerank with exact distances
results = db.semantic_search(
    query_vector,
    k=10,
    index="docs_vec",
    rerank=True,  # Rerank using exact distances
    candidate_multiplier=3  # Fetch 3x candidates before reranking
)
```

**How it works:**
1. Fetch `k * candidate_multiplier` candidates using approximate index
2. Recompute exact distances for all candidates
3. Return top `k` results after reranking

### 3. Batch Operations

Efficiently insert multiple embeddings:

```python
# Batch insert
db.upsert_embeddings(
    node_ids=[1, 2, 3, 4, 5],
    vectors=[vec1, vec2, vec3, vec4, vec5],
    index="docs_vec"
)
```

### 4. Index Management

```python
# List all vector indexes
indexes = db.list_vector_indexes()
for idx in indexes:
    print(f"{idx['name']}: {idx['dim']}D, {idx['backend']}, {idx['method']}")

# Drop an index
db.drop_vector_index("old_index")

# Get index statistics
stats = db.get_vector_index_stats("docs_vec")
print(f"Total vectors: {stats['count']}")
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

### 5. Embed the Right Properties

Choose properties that best represent semantic meaning:

```python
# Good: Embed rich text content
text = f"{paper['title']}. {paper['abstract']}"

# Avoid: Embedding IDs, dates, or non-semantic data
# Bad: text = f"{paper['id']}"  # No semantic meaning
```

### 6. Index Tuning

For HNSW indexes, balance accuracy vs. performance:

```python
from grafito.indexers import HNSWlibIndexer

# Higher accuracy, slower, more memory
indexer = HNSWlibIndexer(
    options={
        "metric": "cosine",
        "M": 32,               # More connections
        "ef_construction": 400, # Better build quality
        "ef_search": 100,       # Better search quality
    },
    embedding_function=embedder
)

# Faster, less accurate, less memory
indexer = HNSWlibIndexer(
    options={
        "metric": "cosine",
        "M": 8,
        "ef_construction": 100,
        "ef_search": 20,
    },
    embedding_function=embedder
)
```

### 7. Monitor Index Performance

```python
import time

start = time.time()
results = db.semantic_search(query_vector, k=10, index="my_index")
elapsed = time.time() - start

print(f"Search took {elapsed*1000:.2f}ms")
print(f"Throughput: {1/elapsed:.1f} queries/sec")
```

### 8. Combine Semantic + Structural Queries

```python
# Don't just use semantic search - leverage the graph!
results = db.semantic_search("machine learning", k=20)

# Then use graph structure
for result in results:
    node = result["node"]

    # Check connectivity
    connections = db.get_neighbors(node.id)

    # Check paths to important nodes
    has_path = db.find_shortest_path(node.id, important_node_id)

    if len(connections) > 5 and has_path:
        # Node is well-connected and relevant
        print(f"Highly relevant: {node.properties['title']}")
```

### 9. Version Your Embeddings

Track which embedding model was used:

```python
# Store model info in properties
node = db.create_node(
    labels=["Document"],
    properties={
        "text": "...",
        "embedding_model": "text-embedding-3-small",
        "embedding_version": "2024-01-01"
    }
)
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

## API Reference

### Database Methods

#### `create_vector_index(name, indexer=None, dim=None, backend=None, method=None, options=None)`

Create a new vector index.

```python
# With indexer object (recommended)
from grafito.indexers import HNSWlibIndexer

indexer = HNSWlibIndexer(options={"metric": "cosine"}, embedding_function=embedder)
db.create_vector_index("my_index", indexer=indexer)

# Manual specification
db.create_vector_index(
    "my_index",
    dim=384,
    backend="faiss",
    method="flat",
    options={"metric": "l2"}
)
```

#### `upsert_embedding(node_id, vector, index="default")`

Insert or update embedding for a node.

```python
db.upsert_embedding(node_id, vector, index="my_index")
```

#### `upsert_embeddings(node_ids, vectors, index="default")`

Batch insert/update embeddings.

```python
db.upsert_embeddings([1, 2, 3], [vec1, vec2, vec3], index="my_index")
```

#### `semantic_search(vector, k=None, index="default", labels=None, properties=None, rerank=False, reranker=None, candidate_multiplier=None)`

Search for nearest neighbors.

```python
results = db.semantic_search(
    vector=query_vector,           # or string if index has embedding function
    k=10,                          # number of results
    index="my_index",              # index name
    labels=["Paper"],              # filter by node labels
    properties={"year": 2023},     # filter by properties
    rerank=True,                   # rerank with exact distances
    reranker="my_reranker",        # custom reranker name
    candidate_multiplier=3         # fetch 3x candidates for reranking
)

# Returns: [{"node": Node, "score": float}, ...]
```

#### `list_vector_indexes()`

List all vector indexes.

```python
indexes = db.list_vector_indexes()
# Returns: [{"name": str, "dim": int, "backend": str, "method": str, ...}, ...]
```

#### `drop_vector_index(name)`

Delete a vector index.

```python
db.drop_vector_index("my_index")
```

#### `register_reranker(name, reranker_fn)`

Register a custom reranking function.

```python
def my_reranker(query_vec, candidate_vecs, candidate_ids):
    # Return sorted [(id, score), ...]
    pass

db.register_reranker("my_reranker", my_reranker)
```

### Cypher Syntax

#### `CALL db.vector.search(index, vector, k, options)`

```cypher
// Basic usage
CALL db.vector.search('index_name', $query_vector, 10)
YIELD node, score
RETURN node, score

// With options
CALL db.vector.search('index_name', $query_vector, 10, {
    labels: ['Label1', 'Label2'],
    properties: {key: 'value'},
    rerank: true,
    candidate_multiplier: 3
})
YIELD node, score
RETURN node, score
```

**Options map:**
- `labels`: List of label filters
- `properties`: Map of property filters
- `rerank`: Boolean, enable reranking
- `candidate_multiplier`: Integer, multiplier for candidate fetching
