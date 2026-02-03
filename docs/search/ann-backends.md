# ANN Backends

GrafitoDB supports multiple Approximate Nearest Neighbor (ANN) backends for vector search.

## Available Backends

| Backend | Install | Best For | Persistence |
|---------|---------|----------|-------------|
| **FAISS** | `pip install grafitodb[faiss]` | Production, most features | ✅ Yes |
| **Annoy** | `pip install grafitodb[annoy]` | Large datasets, memory-mapped | ✅ Yes |
| **LEANN** | `pip install grafitodb[leann]` | Fast builds, small datasets | ✅ Yes |
| **HNSWlib** | `pip install grafitodb[hnswlib]` | High recall | ⚠️ Soft delete |
| **USearch** | `pip install grafitodb[usearch]` | Modern, fast | ✅ Yes |
| **Voyager** | `pip install grafitodb[voyager]` | Spotify's library | ✅ Yes |
| **Brute Force** | Built-in | Small datasets, exact search | ❌ No |

## FAISS

Facebook AI Similarity Search. Most feature-complete backend.

### Installation

```bash
pip install grafito[faiss]
# Or with conda:
conda install -c pytorch faiss-cpu
```

### Methods

```python
# Flat (exact search)
db.create_vector_index(
    name='docs',
    dim=384,
    backend='faiss',
    method='flat',
    options={'metric': 'l2'}
)

# IVF (inverted file index)
db.create_vector_index(
    name='docs',
    dim=384,
    backend='faiss',
    method='ivf_flat',
    options={'nlist': 100, 'nprobe': 10}
)

# HNSW (hierarchical navigable small world)
db.create_vector_index(
    name='docs',
    dim=384,
    backend='faiss',
    method='hnsw',
    options={'hnsw_m': 16, 'ef_construction': 200}
)
```

### Persistence

```python
db.create_vector_index(
    name='docs',
    dim=384,
    backend='faiss',
    method='flat',
    options={'index_path': 'indexes/docs.faiss'}
)
```

## Annoy

Spotify's Approximate Nearest Neighbors Oh Yeah.

### Installation

```bash
pip install grafito[annoy]
```

### Characteristics

- Memory-mapped indexes (shareable between processes)
- Good for read-heavy workloads
- Static indexes (rebuild to add vectors)

```python
db.create_vector_index(
    name='docs',
    dim=384,
    backend='annoy',
    method='annoy',
    options={
        'metric': 'angular',  # or 'euclidean', 'manhattan'
        'n_trees': 50,        # More trees = more accurate, slower build
        'index_path': 'indexes/docs.annoy'
    }
)
```

## LEANN

Lightweight Efficient Approximate Nearest Neighbors.

### Installation

```bash
pip install grafito[leann]
```

### Characteristics

- Fast index building
- Good for small to medium datasets
- Auto-build control

```python
db.create_vector_index(
    name='docs',
    dim=384,
    backend='leann',
    method='leann',
    options={
        'metric': 'l2',
        'auto_build': False,  # Disable auto-build
        'index_path': 'indexes/docs.leann'
    }
)

# Add embeddings...

# Manual rebuild
db.rebuild_vector_index('docs')
```

## HNSWlib

Hierarchical Navigable Small World implementation.

### Installation

```bash
pip install grafito[hnswlib]
```

### Characteristics

- High recall rates
- Supports incremental adds
- Soft deletes (need rebuild to purge)

```python
db.create_vector_index(
    name='docs',
    dim=384,
    backend='hnswlib',
    method='hnswlib',
    options={
        'metric': 'l2',
        'M': 16,
        'ef_construction': 200,
        'ef': 50
    }
)
```

!!! note
    HNSWlib uses soft deletes. Call `rebuild_vector_index()` to fully remove deleted vectors.

## USearch

Modern FAISS alternative by Unum.

### Installation

```bash
pip install grafito[usearch]
```

### Characteristics

- Faster than FAISS for many workloads
- Smaller memory footprint
- Native persistence

```python
db.create_vector_index(
    name='docs',
    dim=384,
    backend='usearch',
    method='usearch',
    options={
        'metric': 'cos',
        'connectivity': 16,
        'expansion_add': 128,
        'expansion_search': 64
    }
)
```

## Voyager

Spotify's latest ANN library.

### Installation

```bash
pip install grafito[voyager]
```

### Characteristics

- Multi-threaded index building
- Good for large-scale search
- EFIL (Enhanced Forward Index Layout)

```python
db.create_vector_index(
    name='docs',
    dim=384,
    backend='voyager',
    method='voyager',
    options={
        'space': 'cosine',  # or 'l2', 'ip'
        'M': 16,
        'ef_construction': 200,
        'index_path': 'indexes/docs.voyager'
    }
)
```

## Brute Force

Built-in exact search for small datasets.

```python
# No extra installation needed
db.create_vector_index(
    name='docs',
    dim=384,
    backend='bruteforce',
    method='bruteforce',
    options={'metric': 'l2'}
)
```

Use for:
- Datasets < 1000 vectors
- When exact results are required
- Testing and debugging

## Backend Comparison

| Feature | FAISS | Annoy | LEANN | HNSWlib | USearch | Voyager |
|---------|-------|-------|-------|---------|---------|---------|
| Incremental adds | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| GPU support | ✅* | ❌ | ❌ | ❌ | ❌ | ❌ |
| Memory mapped | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Cosine similarity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Inner product | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| L2 distance | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

*Requires faiss-gpu (conda)

## Choosing a Backend

### Small Dataset (< 10K vectors)

```python
# Brute force or FAISS flat
db.create_vector_index(..., backend='bruteforce', method='bruteforce')
# or
db.create_vector_index(..., backend='faiss', method='flat')
```

### Medium Dataset (10K - 100K)

```python
# FAISS IVF or HNSW
db.create_vector_index(..., backend='faiss', method='ivf_flat')
# or
db.create_vector_index(..., backend='faiss', method='hnsw')
```

### Large Dataset (> 100K)

```python
# Annoy (memory-mapped) or Voyager
db.create_vector_index(..., backend='annoy', method='annoy')
# or
db.create_vector_index(..., backend='voyager', method='voyager')
```

### Read-Heavy Workloads

```python
# Annoy with memory mapping
db.create_vector_index(..., backend='annoy', method='annoy',
                       options={'index_path': 'shared.annoy'})
```

### Write-Heavy Workloads

```python
# FAISS HNSW or HNSWlib
db.create_vector_index(..., backend='faiss', method='hnsw')
```

## Checking Available Backends

```python
from grafito.integrations import available_vector_backends

print(available_vector_backends())
# ['bruteforce', 'faiss', 'annoy', 'leann', 'hnswlib', 'usearch', 'voyager']
```
