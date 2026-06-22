---
type: Term
title: Semantic search
description: Retrieval by meaning using vector embeddings rather than keyword matching.
tags: [glossary, search, embeddings]
timestamp: 2026-01-05T00:00:00Z
---

# Definition

**Semantic search** retrieves items by *meaning* instead of exact words. Each
piece of text is converted into a vector embedding; a query is embedded the same
way, and the nearest vectors are returned. A search for "how do I make a query
faster" can surface a document about query performance even if it never uses
those exact words.

In this project it is layered over the [property graph](/glossary/property-graph.md)
as an optional capability — see [the vector search decision](/decisions/0003-vector-search.md) —
and complements pattern queries written in [Cypher](/glossary/cypher.md).

# Citations

- [Efficient and robust approximate nearest neighbor search (HNSW)](https://arxiv.org/abs/1603.09320)
