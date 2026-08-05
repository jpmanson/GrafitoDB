# Search Results as Subgraphs

`semantic_search()` returns a ranked list. `semantic_subgraph()` returns the same
hits *plus how they connect* — the thing a graph database knows that a vector
store does not.

```python
sub = db.semantic_subgraph("autonomous agents", k=30, expand=1)

print(len(sub))                 # nodes in the subgraph
print(sub.relationships)        # every edge among them
print(sub.seeds)                # the ranked hits that seeded it
```

## Why a Subgraph

A top-k list treats results as independent, which they usually are not. Given
the same 30 hits, the subgraph tells you that eight of them cite one another and
form a cluster, that two are isolated, and which one sits in the middle. That is
directly useful for three things: visualising a result, ranking within a result,
and packing an LLM prompt with context that has structure rather than just
relevance.

## Expansion and Provenance

`expand` pulls in the neighbourhood of each hit:

```python
sub = db.semantic_subgraph("autonomous agents", k=20, expand=2)
```

Every node records where it came from:

```python
sub.scores    # {node_id: retrieval_score}  — seeds only
sub.hops      # {node_id: distance}         — 0 for seeds, 1+ for expanded
```

This is what keeps the result explainable. Without it, a subgraph is an
undifferentiated blob and there is no way to distinguish a strong direct match
from something two hops away that happened to be adjacent.

All relationships between selected nodes are returned, not only the ones
traversed — two seeds that link to each other show that link even if neither was
reached from the other.

## Controlling Expansion

```python
sub = db.semantic_subgraph(
    "autonomous agents",
    k=20,
    expand=2,
    direction="out",                        # "both" (default), "out", "in"
    rel_types=["CITES"],                    # traverse only these
    exclude_rel_types=["SEMANTIC_SIMILAR"], # never traverse these
    labels=["Article"],                     # only expand into these labels
    max_nodes=500,                          # stop once this large
)
```

!!! warning "Expansion is exponential in dense graphs"
    One hop from a hub node can pull in thousands of nodes; `expand=2` through
    that hub reaches most of the database. `max_nodes` is the guard — expansion
    stops once the subgraph reaches that size, which shows up as missing hops
    rather than a hang.

    `exclude_rel_types` is the sharper tool: derived edges like similarity links
    connect everything to everything, so a single hop through them is already a
    near-full scan.

Retrieval filters and expansion filters are separate. `filter_labels` and
`filter_props` constrain *which hits seed* the subgraph; `labels` and
`rel_types` constrain *where expansion goes*:

```python
sub = db.semantic_subgraph(
    "autonomous agents",
    filter_props={"published": True},   # seed only from published articles
    labels=["Article"],                 # but expand only into articles too
    expand=1,
)
```

## Lexical and Explicit Seeds

`text_subgraph()` is the FTS5/BM25 counterpart, taking the same expansion
arguments:

```python
sub = db.text_subgraph("attention mechanism", k=20, expand=1)
```

`subgraph()` takes seeds directly — node ids, `Node` objects, or search hits —
so any retrieval strategy can feed it, including a hybrid fusion of your own:

```python
sub = db.subgraph([node_id_1, node_id_2], expand=2)
sub = db.subgraph(my_fused_hits, expand=1)   # [{"node": Node, "score": ...}]
```

## Visualising

`to_networkx()` carries the provenance into the graph, so a viewer can size or
colour nodes by score and hop distance:

```python
from grafito.integrations.viz import export_graph

sub = db.semantic_subgraph("autonomous agents", k=30, expand=1)
export_graph(sub.to_networkx(), "context.html", backend="cytoscape")
```

Node attributes: `labels`, `properties`, `uri`, `score`, `hops`.
Edge attributes: `type`, `properties`, `uri`.

## Ranking Within a Result

Both [graph algorithms](graph-algorithms.md) accept `graph=`, so a subgraph can
be ranked on its own terms:

```python
sub = db.semantic_subgraph("autonomous agents", k=50, expand=1)
graph = sub.to_networkx()

central = db.centrality("pagerank", graph=graph, limit=10)
clusters = db.communities("louvain", graph=graph, seed=42)
```

This answers "what is important among the things that matched?" — a different
and often more useful question than global centrality, and the basis for
graph-aware reranking: a node with a middling vector score that everything else
in the result set points at is usually worth surfacing.

## API Reference

::: grafito.subgraph.Subgraph
