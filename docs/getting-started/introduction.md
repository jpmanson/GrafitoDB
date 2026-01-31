# Introduction

**Grafito** is a fast, embeddable property graph database for Python — no server required,
SQLite-backed, Cypher-powered, with optional semantic search capabilities.

## What is Grafito?

Grafito implements the **Property Graph Model** (Neo4j-compatible concepts) on top of SQLite,
providing a lightweight yet powerful graph database that runs directly in your Python application.

### Key Features

| Feature | Description |
|---------|-------------|
| **No Server Required** | Embeddable database that runs in-process with your application |
| **SQLite-Backed** | Reliable storage with ACID transactions |
| **Cypher Query Language** | Full Cypher parser and executor for declarative queries |
| **Multi-Labeled Nodes** | Nodes can have multiple labels (e.g., `Person`, `Employee`, `Manager`) |
| **Rich Properties** | JSON-serializable properties on nodes and relationships |
| **Full-Text Search** | BM25-ranked keyword search via SQLite FTS5 |
| **Semantic Search** | Vector similarity search with multiple ANN backends |
| **NetworkX Compatible** | Export to/import from NetworkX for graph algorithms |
| **RDF/Turtle Support** | Import and export RDF data |
| **Visualization** | Export to PyVis, D2, Mermaid, and Graphviz |

## When to Use Grafito

Grafito is ideal for:

- **Prototyping** graph-based applications without infrastructure overhead
- **Educational** purposes — learning graph databases and Cypher
- **Small to Medium Scale** — graphs up to 100K+ nodes
- **Embedded Applications** — shipping a graph database with your Python app
- **Testing** — graph algorithms without heavy dependencies
- **Hybrid Workflows** — combining structured and semantic search

## Quick Comparison

| Feature | Grafito | Neo4j | NetworkX |
|---------|---------|-------|----------|
| Embeddable | ✅ Yes | ❌ No | ✅ Yes |
| Persistent | ✅ Yes | ✅ Yes | ❌ No |
| Cypher Support | ✅ Full | ✅ Full | ❌ No |
| Server Required | ❌ No | ✅ Yes | ❌ No |
| Vector Search | ✅ Built-in | Plugin | ❌ No |
| Full-Text Search | ✅ FTS5 | ✅ Native | ❌ No |

## Philosophy

Grafito follows these design principles:

1. **Simplicity First**: Simple installation, simple API, simple mental model
2. **Python-Native**: First-class Python integration with familiar patterns
3. **Standards-Based**: Cypher queries, SQLite storage, NetworkX compatibility
4. **Extensible**: Plugin architecture for embeddings, visualization, and more
5. **Production-Ready**: ACID transactions, indexes, constraints, and testing utilities

## Next Steps

- **[Installation](installation.md)** — Get Grafito up and running
- **[Quick Start](quickstart.md)** — Your first graph in 5 minutes
- **[Core Concepts](concepts.md)** — Understanding the property graph model