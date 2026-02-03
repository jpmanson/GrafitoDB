# Advanced: Architecture

GrafitoDB is a SQLite-backed property graph database with a Cypher-like query
engine and pluggable indexes.

## High-Level Components

- **Storage**: SQLite database, JSON properties stored as text.
- **Schema**: Tables for nodes, labels, relationships, indexes, constraints.
- **Cypher Engine**: Parser + executor + evaluator for graph queries.
- **Indexes**:
  - Property indexes
  - Full-text search (FTS5)
  - Vector/ANN backends
- **Integrations**: Optional helpers for NetworkX, RDF, visualization.

## Core Modules

- `grafito/database.py`: Main API surface (`GrafitoDatabase`).
- `grafito/schema.py`: SQLite schema and initialization.
- `grafito/cypher/`: Lexer, parser, AST, and execution engine.
- `grafito/indexers/`: Property index metadata and helpers.
- `grafito/text_index/`: SQLite FTS and optional BM25S integration.
- `grafito/vector_index/`: ANN backends and persistence.
- `grafito/embedding_functions/`: Embedding provider integrations.

## Execution Flow (Cypher)

1) Parse query into an AST.
2) Execute clauses in order, carrying intermediate results.
3) Evaluate expressions and functions with the evaluator.
4) Apply filters and ordering.
5) Return result rows.

## Traversal

Traversal and shortest-path functionality are implemented in
`grafito/query.py` (BFS/DFS) and used by the database API.

## Index Lifecycle

Vector indexes and text indexes are registered in the database instance,
stored as metadata, and rebuilt/loaded on demand.
