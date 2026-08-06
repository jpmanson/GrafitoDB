# Bulk Document Ingest

`index_documents()` turns row-shaped data — a HuggingFace dataset, a DataFrame's
`to_dict("records")`, a JSONL file — into nodes, relationships, and embeddings in
one call.

```python
rows = [
    {"id": "1", "text": "Graph databases store nodes and edges", "year": 2024},
    {"id": "2", "text": "Vector search finds nearest neighbours", "year": 2025},
]

db.create_vector_index("default", dim=384, embedding_function=embedder)
report = db.index_documents(rows, label="Document")

print(report)  # IndexReport(2 created, 0 updated, 0 relationships, 2 embedded)
```

Every key other than the relationships field becomes a node property, so `year`
above is queryable immediately. Set `copy_attributes=False` to keep only `id` and
`text`, or pass `properties=["id", "year"]` for an explicit allowlist.

## Relationships

Point `relationships_key` at the field holding each row's outgoing edges. Both
shapes work:

```python
rows = [
    {"id": "1", "text": "...", "links": ["2", "3"]},
    {"id": "2", "text": "...", "links": [
        {"id": "3", "type": "CITES", "properties": {"weight": 2}},
    ]},
    {"id": "3", "text": "...", "links": []},
]

report = db.index_documents(rows, label="Paper", relationships_key="links")
```

A bare id uses `default_rel_type` (`RELATED_TO`); the mapping form takes `type`
and `properties`.

Nodes are created in a first pass and relationships in a second, so a row may
reference a document that appears later in `rows` — forward references resolve.
Targets that name an id no row supplied are skipped and collected:

```python
if report.unresolved:
    print(f"{len(report.unresolved)} dangling references: {report.unresolved[:5]}")
```

A dataset slice legitimately references rows outside the slice, so this is not an
error. But a long `unresolved` list usually means `id_key` does not match the ids
used in the relationship field.

## Identity and Re-ingest

`id_key` (default `"id"`) gives rows an external identity. When `upsert=True`
(the default), re-ingesting a row with a known id updates that node instead of
creating a duplicate:

```python
db.index_documents(rows, label="Paper")           # 3 created
db.index_documents(updated_rows, label="Paper")   # 0 created, 3 updated
```

Pass `upsert=False` to always create, or `id_key=None` for data with no identity
— note that relationships cannot be resolved without an id key.

`report.ids` maps external ids to node ids, which is how you connect this data to
the rest of the graph afterwards:

```python
report = db.index_documents(rows, label="Paper")
db.create_relationship(report.ids["1"], author_node.id, "AUTHORED_BY")
```

## Embedding

Text is embedded in batches using the vector index's embedding function:

```python
db.index_documents(rows, label="Paper", index="papers_vec", batch_size=512)
```

- `batch_size` controls how many texts go to the embedder per call. Larger
  batches are faster for API-backed embedders and use more memory for local ones.
- Rows with no `text_key`, or with blank text, are stored as nodes but not
  embedded.
- `index=None` skips embedding entirely — useful for a structure-only load.

## Full-Text

Pass `configure_fts=True` to register a full-text index over `label`/`text_key`
at the same time, so the documents are reachable by both retrieval paths:

```python
db.index_documents(rows, label="Paper", configure_fts=True)

db.semantic_search("nearest neighbours", k=10)
db.text_search("nearest neighbours", k=10)
```

It is off by default because it creates an index you did not ask for. You can
also call `create_text_index()` yourself at any point — FTS5 indexes existing
rows when created, so the order does not matter:

```python
db.index_documents(rows, label="Paper")
db.create_text_index("node", "Paper", ["text"])
```

## End to End

```python
db = GrafitoDatabase("papers.db")
db.create_vector_index("default", dim=384, embedding_function=embedder)

report = db.index_documents(
    dataset,                      # any iterable of dicts
    label="Paper",
    text_key="abstract",
    relationships_key="citations",
    default_rel_type="CITES",
    batch_size=512,
)
db.create_text_index("node", "Paper", ["abstract"])

# Retrieve as a graph, then rank within the result
sub = db.semantic_subgraph("retrieval augmented generation", k=50, expand=1)
for hit in db.centrality("pagerank", graph=sub.to_networkx(), limit=10):
    print(hit["node"].properties["title"], hit["score"])
```

## API Reference

::: grafito.ingest_report.IndexReport
