---
type: ADR
title: Add optional vector search
description: Why semantic search is layered on top as an optional capability.
status: accepted
tags: [search, embeddings, architecture]
timestamp: 2026-03-18T14:00:00Z
---

# Context

Keyword and pattern queries can't answer questions phrased by *meaning*. Users
want to retrieve nodes whose text is conceptually related to a query, not just
lexically matching it.

# Decision

We add **[semantic search](/glossary/semantic-search.md)** as an *optional*
layer: embeddings are stored alongside the [property graph](/glossary/property-graph.md)
and queried through pluggable approximate-nearest-neighbour backends. It stays
optional so the core database keeps zero required dependencies, consistent with
[the SQLite decision](/decisions/0001-use-sqlite.md).

# Consequences

- Meaning-based retrieval complements the [Cypher subset](/decisions/0002-cypher-subset.md)
  and full-text search.
- Users choose an embedding model and an ANN backend per their needs.
- Embeddings can be rebuilt from source text at any time.

# Citations

- [Efficient and robust approximate nearest neighbor search (HNSW)](https://arxiv.org/abs/1603.09320)
- https://github.com/facebookresearch/faiss
