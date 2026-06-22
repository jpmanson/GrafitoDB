---
type: ADR
title: Implement a Cypher subset
description: Why we expose a subset of Cypher instead of a bespoke query API.
status: accepted
tags: [query, cypher, architecture]
timestamp: 2026-02-03T09:30:00Z
---

# Context

Building on [the SQLite storage decision](/decisions/0001-use-sqlite.md), we
needed a query surface for pattern matching and traversal over the graph.

# Decision

We implement a practical subset of **[Cypher](/glossary/cypher.md)** — the
declarative query language popularized by Neo4j — rather than inventing a new
API. Familiarity lowers the learning curve and lets users port existing queries.

# Consequences

- Users already fluent in Cypher are productive immediately.
- We deliberately do **not** target full Neo4j parity; unsupported constructs
  fail loudly rather than silently misbehaving.
- Slow patterns are a real risk; see the
  [slow query runbook](/runbooks/slow-queries.md) for triage guidance.

# Citations

- [openCypher specification](https://opencypher.org/)
- https://neo4j.com/docs/cypher-manual/current/
