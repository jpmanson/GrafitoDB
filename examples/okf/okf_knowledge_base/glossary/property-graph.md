---
type: Term
title: Property graph
description: A data model of labeled nodes and typed, directed relationships, both carrying properties.
tags: [glossary, data-model]
timestamp: 2026-01-05T00:00:00Z
---

# Definition

A **property graph** represents data as *nodes* and *relationships*. Nodes carry
one or more labels and a set of key/value properties; relationships are directed,
have a single type, and may also carry properties.

This is the model popularized by Neo4j and the one this project implements. It
contrasts with the triple-based RDF model, where everything is decomposed into
subject–predicate–object statements.

Queried with [Cypher](/glossary/cypher.md); enriched with
[semantic search](/glossary/semantic-search.md).

# Citations

- [Property Graph Model overview](https://neo4j.com/docs/getting-started/appendix/graphdb-concepts/)
