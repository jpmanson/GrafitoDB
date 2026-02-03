# Installation

## Requirements

- Python 3.11 or higher
- SQLite 3 with FTS5 support (for full-text search features)

## Basic Installation

Install GrafitoDB from PyPI:

```bash
pip install grafitodb
```

Or using `uv`:

```bash
uv pip install grafito
```

## Development Installation

To install GrafitoDB in development mode with all test dependencies:

```bash
git clone <repository-url>
cd grafito
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv pip install -e ".[dev]"
```

## Optional Dependencies

GrafitoDB has a modular design with optional extras for specific features:

### All Extras

```bash
pip install grafito[all]
```

!!! note
    `grafito[all]` may fail on some OS/Python combinations depending on native wheels
    for optional backends. In that case, install only the extras you need.

### Vector Search Backends

Choose one or more ANN (Approximate Nearest Neighbors) backends:

| Backend | Installation | Best For |
|---------|-------------|----------|
| FAISS | `pip install grafitodb[faiss]` | CPU-optimized, most features |
| Annoy | `pip install grafitodb[annoy]` | Memory-mapped large indexes |
| LEANN | `pip install grafitodb[leann]` | Lightweight, fast builds |
| HNSWlib | `pip install grafitodb[hnswlib]` | High-recall search |
| USearch | `pip install grafitodb[usearch]` | Modern alternative to FAISS |
| Voyager | `pip install grafitodb[voyager]` | Spotify's ANN library |

### Other Integrations

```bash
# RDF/Turtle support
pip install grafitodb[rdf]

# Visualization with PyVis
pip install grafitodb[viz]

# BM25 text search enhancements
pip install grafitodb[bm25s]
```

## Verifying Installation

Test your installation:

```python
from grafito import GrafitoDatabase

# Create an in-memory database
db = GrafitoDatabase(':memory:')

# Create a test node
node = db.create_node(labels=['Test'], properties={'message': 'Hello, GrafitoDB!'})
print(f"Created node: {node.id}")

# Check FTS5 availability
print(f"FTS5 available: {db.has_fts5()}")

db.close()
```

## Troubleshooting

### FTS5 Not Available

If `db.has_fts5()` returns `False`, your SQLite build doesn't include FTS5:

**macOS:**
```bash
brew install sqlite3
# Or use Python from conda:
conda install sqlite
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libsqlite3-dev
```

**Windows:**
Use Python from [python.org](https://python.org) or Conda, which typically include FTS5.

### FAISS Installation Issues

FAISS requires native compilation. If installation fails:

```bash
# Use conda instead
conda install -c pytorch faiss-cpu

# Then install grafitodb without the faiss extra
pip install grafito
```

### Import Errors with Optional Dependencies

If you get import errors for optional features:

```python
# Check available backends
from grafito.integrations import available_viz_backends, available_vector_backends

print("Visualization:", available_viz_backends())
print("Vector:", available_vector_backends())
```
