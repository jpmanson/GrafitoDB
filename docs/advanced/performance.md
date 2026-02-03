# Advanced: Performance

GrafitoDB performance depends on SQLite configuration, index usage, and query
patterns. This page summarizes practical tips.

## General Tips

- Use transactions for bulk inserts (`db.begin_transaction()` / `db.commit()`).
- Keep properties lean; avoid large blobs in node properties.
- Prefer targeted labels and relationship types to reduce scan size.

## Indexing

- Create property indexes for frequently filtered properties.
- Use full-text search for text-heavy filtering instead of `CONTAINS`.
- Use vector indexes for semantic search at scale.

## Text Search (FTS5)

- Ensure your SQLite build supports FTS5.
- Rebuild the FTS index after bulk updates if needed.

## Vector Indexing

- Choose an ANN backend appropriate for your dataset size.
- Persist vector indexes on disk for faster startups.
- Rebuild indexes after large batch updates.

## Query Planning

- Filter early (labels/types/properties) and avoid wide graph traversals.
- Keep shortest-path searches bounded where possible.

## Operational Notes

- For large workloads, use a file-backed database instead of `:memory:`.
- Consider WAL mode in SQLite for better concurrent reads.
