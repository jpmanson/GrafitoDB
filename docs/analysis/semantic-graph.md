# Materialised Semantic Graph

`create_semantic_graph()` turns the vector index into relationships: every
indexed node gets edges to its `k` nearest neighbours, so similarity becomes
something you can traverse with Cypher and cluster with community detection.

```python
report = db.create_semantic_graph(
    index="default",
    rel_type="SEMANTIC_SIMILAR",
    k=15,
    min_score=0.1,
)
print(report)  # SemanticGraphReport(742 edges, 100 nodes)
```

## When to Use It — and When Not To

!!! warning "Materialised edges are a cache, and caches go stale"
    This writes `k` edges per node into the same table as your domain
    relationships. At 100k nodes with `k=15` that is ~750k rows. Those edges:

    - **dominate unqualified traversals.** `MATCH (a)-[]->(b)` now sweeps
      similarity noise, and every centrality or community result is meaningless
      unless it excludes them.
    - **go stale silently.** They reflect the embeddings as of the build.
      Nothing updates them when a vector changes.
    - **need maintenance forever.** Every ingest leaves the graph a little more
      out of date until you rebuild.

    Before reaching for this, check whether a query answers the same question:

    ```cypher
    -- No stored edges, never stale
    CALL db.vector.search('default', 'transformers', 15) YIELD node, score
    RETURN node, score
    ```

    [`SIMILAR()` and `db.vector.search`](../cypher/vector-search.md) cover most
    of what materialised neighbours are used for, and
    [`semantic_subgraph()`](subgraphs.md) covers most of the rest.

The case that genuinely needs materialised edges is **community detection over
similarity**. Modularity algorithms need edges to exist; there is no way to run
Louvain against an ANN index:

```python
db.create_semantic_graph(k=15, min_score=0.3)

for community in db.communities(
    "louvain", rel_types=["SEMANTIC_SIMILAR"], weight_property="score", seed=42
):
    print(community.size, [n.properties["title"] for n in community.nodes[:3]])
```

Multi-hop similarity patterns are the other — "things like this, and things like
*those*", which an ANN query cannot express:

```cypher
MATCH p=(a:Doc {id: 'd1'})-[:SEMANTIC_SIMILAR*1..2]-(b:Doc)
RETURN DISTINCT b
```

## Controlling the Edge Count

`min_score` is the most effective lever — far more so than `k`, because it cuts
the long tail of weak neighbours that inflate the graph without adding signal:

```python
db.create_semantic_graph(k=15, min_score=0.5)   # far fewer edges than the 0.1 default
```

It defaults to `0.1`. That is lower than the `SIMILAR()` default of `0.5`, on
purpose: `SIMILAR()` answers "is this similar?", where a permissive threshold
gives wrong answers, whereas here `k` already bounds the result and `min_score`
only trims the tail.

On an `l2` index the default is refused rather than guessed — those scores are
negated distances (`<= 0`), so any positive threshold would build an empty graph:

```python
db.create_semantic_graph(k=15)                  # DatabaseError on an l2 index
db.create_semantic_graph(k=15, min_score=-0.5)  # explicit, and meaningful
```

`undirected=True` (the default) emits one edge per pair instead of one per
direction, roughly halving the count. Query it with an undirected pattern:

```cypher
MATCH (a)-[:SEMANTIC_SIMILAR]-(b)   -- note: no arrow
```

`labels` restricts which nodes participate, and `max_edges` caps the build
outright:

```python
db.create_semantic_graph(k=15, min_score=0.3, labels=["Article"], max_edges=100_000)
```

## Provenance and Rebuilding

Every generated edge carries `score`, `index`, `generated_by`, and
`generated_at`. Those last two are what make rebuilds safe: `replace=True` (the
default) deletes only edges this method created **from the same index**, so both
a hand-made `SEMANTIC_SIMILAR` relationship and edges generated from a different
vector index survive a rebuild.

```python
db.create_semantic_graph(k=15, min_score=0.3)      # idempotent: re-running replaces
db.drop_semantic_graph()                           # generated edges only
db.drop_semantic_graph(index="papers_vec")         # ...from one index only
```

!!! note "The rebuild is atomic"
    Neighbours are computed first — the slow part, minutes on a large corpus —
    and the old edges are swapped for the new ones inside a single short
    transaction. A failure mid-build leaves the previous graph intact, and
    concurrent readers never observe a window where the semantic graph is
    missing.

## Incremental Updates

After adding documents, link the new ones without rebuilding everything:

```python
db.index_documents(new_rows, label="Doc")
report = db.refresh_semantic_graph(k=15, min_score=0.3)
print(report)  # SemanticGraphReport(30 edges, 2 nodes, 100 skipped)
```

`refresh_semantic_graph()` only processes nodes that have no *generated* edge for
that `rel_type` and `index` yet. Hand-made relationships of the same type do not
count as "already linked", and neither do edges generated from a different vector
index — otherwise a single manual edge would exclude its endpoints from the
semantic graph forever.

It will **not** notice that an existing node's neighbourhood changed — new
documents can be a better match for old ones than what is currently stored.
Only a full rebuild fixes that, so schedule one periodically if your corpus
keeps growing.

## Keeping Analysis Honest

Once these edges exist, every graph analysis needs to say whether it wants them:

```python
# Structure of the domain
db.centrality("pagerank", exclude_rel_types=["SEMANTIC_SIMILAR"])

# Structure of the embedding space
db.communities("louvain", rel_types=["SEMANTIC_SIMILAR"], weight_property="score")
```

The same applies to [subgraph expansion](subgraphs.md) — a single hop through
similarity edges reaches a large part of the database:

```python
db.semantic_subgraph("agents", k=20, expand=1, exclude_rel_types=["SEMANTIC_SIMILAR"])
```

## API Reference

::: grafito.ingest_report.SemanticGraphReport
