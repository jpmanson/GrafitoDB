---
type: Playbook
title: Triaging a slow graph query
description: Steps to diagnose and mitigate a query that is taking too long.
tags: [oncall, performance, query]
timestamp: 2026-04-02T08:15:00Z
---

# Trigger

A [Cypher](/glossary/cypher.md) query runs far longer than expected, or a
request times out waiting on the database.

# Steps

1. Identify the pattern. Unbounded variable-length paths (`[:REL*..]`) are the
   most common culprit — they can explode combinatorially.
2. Add a hop bound (`[:REL*1..3]`) or a `LIMIT` to constrain the traversal.
3. Check whether the matched labels and filtered properties are indexed; if not,
   create a property index so the planner can prune candidates early.
4. For repeated lookups by a key, confirm a uniqueness or property index exists.
5. If the query is fundamentally a similarity question rather than a traversal,
   it may belong in [semantic search](/glossary/semantic-search.md) instead — see
   [the vector search decision](/decisions/0003-vector-search.md).

# Notes

This runbook assumes the embedded, single-writer model established in
[the SQLite decision](/decisions/0001-use-sqlite.md): long-running reads do not
block other readers, but a slow write holds the write lock.

# Citations

- https://www.sqlite.org/queryplanner.html
- [EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html)
