---
type: Term
title: Cypher
description: A declarative query language for property graphs.
tags: [glossary, query]
timestamp: 2026-01-05T00:00:00Z
---

# Definition

**Cypher** is a declarative, pattern-oriented query language for
[property graphs](/glossary/property-graph.md). Patterns are written as ASCII-art:
`(a:Person)-[:KNOWS]->(b:Person)` matches a person who knows another person.

It supports reading (`MATCH`), writing (`CREATE`, `MERGE`, `SET`, `DELETE`),
aggregation, and variable-length traversal. This project implements a practical
subset of it — see [the decision record](/decisions/0002-cypher-subset.md).

# Citations

- [openCypher specification](https://opencypher.org/)
