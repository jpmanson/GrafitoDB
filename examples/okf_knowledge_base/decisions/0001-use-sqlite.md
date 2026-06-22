---
type: ADR
title: Use SQLite as the storage engine
description: Why the database is built on SQLite rather than a custom store or a server.
status: accepted
tags: [storage, architecture, embedded]
timestamp: 2026-01-12T10:00:00Z
---

# Context

We need durable storage for a [property graph](/glossary/property-graph.md)
that runs embedded inside a Python process, with zero operational overhead and
no separate server to deploy.

# Decision

We store nodes, relationships, and labels in **SQLite**. Properties are kept as
JSON columns, which gives us schema-free flexibility while still allowing
expression indexes on individual fields.

# Consequences

- No server to run: the whole database is a single file (or in-memory).
- We inherit SQLite's transactions and durability for free.
- Heavy write concurrency is limited to one writer at a time — acceptable for
  the embedded, small-to-medium graphs we target.
- This decision is the foundation that [the Cypher subset](/decisions/0002-cypher-subset.md)
  and [vector search](/decisions/0003-vector-search.md) build on.

# Citations

- https://www.sqlite.org/whentouse.html
- [SQLite JSON1 extension](https://www.sqlite.org/json1.html)
