# Graph Algorithms

Centrality and community detection run directly against the database. Both are
NetworkX under the hood — GrafitoDB's contribution is scoping the analysis to
the right subgraph and resolving results back to nodes.

## Centrality

```python
for hit in db.centrality("pagerank", limit=10):
    print(hit["node"].properties["name"], hit["score"])
```

Results come back sorted, highest first, as `{"node": Node, "score": float}`.

| `kind` | Answers | Cost |
| --- | --- | --- |
| `pagerank` | Which nodes are reachable from many important nodes? | Fast |
| `degree` | Which nodes have the most connections? | Fast |
| `in_degree` / `out_degree` | Directed variants (require `directed=True`) | Fast |
| `betweenness` | Which nodes sit on the most shortest paths? | **O(V·E)** |
| `closeness` / `harmonic` | Which nodes are near everything else? | Expensive |
| `eigenvector` | Like PageRank, undirected-flavoured | Moderate |

Degree and PageRank usually agree; `betweenness` is the one that disagrees, and
that disagreement is the point — it finds brokers and bridges that are weakly
connected but structurally critical:

```python
# The article that connects two otherwise separate literatures
db.centrality("betweenness", directed=False, limit=5)
```

On graphs past a few thousand nodes, `betweenness` gets slow. Pass NetworkX's
sampling parameter through, or use `pagerank`:

```python
db.centrality("betweenness", k=200)  # approximate, sampled from 200 sources
```

## Scoping the Analysis

Every algorithm accepts the same filters, which build the graph the analysis
runs on:

```python
db.centrality(
    "pagerank",
    rel_types=["CITES"],                   # only these edge types
    exclude_rel_types=["SEMANTIC_SIMILAR"],# never these
    labels=["Article"],                    # only these node labels
    directed=True,
    weight_property="weight",              # relationship property to weight by
)
```

!!! warning "Exclude derived edges before analysing"
    Bulk-generated edges — similarity links, containment, anything produced by a
    batch job rather than the domain — will dominate any centrality or community
    result. There are usually far more of them than real edges, and they are
    distributed by embedding geometry rather than meaning.

    ```python
    # Meaningless: similarity edges swamp the citation structure
    db.centrality("pagerank")

    # Meaningful
    db.centrality("pagerank", rel_types=["CITES"])
    ```

    The filtering has to happen before the algorithm runs. Post-filtering the
    results does not undo the effect the extra edges had on the scores.

`weight_property` reads through to relationship properties, so
`weight_property="score"` finds `{"score": 0.9}` on the relationship. Edges
missing it get weight 1.0. It is rejected for `degree`-family measures, which
count edges and cannot honour a weight.

## Communities

```python
for community in db.communities("louvain", seed=42):
    names = [n.properties["name"] for n in community.nodes]
    print(f"community {community.id} ({community.size}): {names}")
```

Returns `Community` objects, largest first.

| `algorithm` | Notes |
| --- | --- |
| `louvain` | Default. Fast, good quality, randomised — pass `seed`. |
| `greedy` | Deterministic modularity maximisation. Slower on large graphs. |
| `lpa` | Label propagation. Fastest, noisiest, randomised. |

`resolution` tunes granularity for `louvain` and `greedy` — higher values
produce more, smaller communities:

```python
db.communities("louvain", resolution=1.5, min_size=3, seed=42)
```

Two properties worth internalising:

- **Direction is dropped.** Modularity is defined on undirected graphs. Parallel
  edges collapse into one weighted edge, so two `KNOWS` edges between the same
  pair count as weight 2.
- **Community ids are positions, not identities.** They are indexes into the
  returned list. Re-running after an edit will renumber them, and `louvain`/`lpa`
  are randomised — pass `seed` for reproducibility.

Use `min_size` to drop singletons, which are noise in most datasets.

## Composing with Retrieval

Both methods accept a pre-built NetworkX graph via `graph=`, which is how they
compose with [subgraphs](subgraphs.md) — retrieve first, then rank *within* the
result rather than across the whole database:

```python
sub = db.semantic_subgraph("autonomous agents", k=50, expand=1)
central = db.centrality("pagerank", graph=sub.to_networkx(), limit=10)
```

That is a different question from global PageRank: not "what is important in
this database?" but "what is important *among the things that matched?*".

## Exporting the Analysis Graph

`to_analysis_graph()` returns the filtered NetworkX graph itself, for anything
these wrappers do not cover:

```python
import networkx as nx

graph = db.to_analysis_graph(rel_types=["CITES"], weight_property="weight")
nx.diameter(graph.to_undirected())

# Many NetworkX algorithms reject multigraphs; collapse parallel edges first
nx.average_clustering(nx.Graph(graph))
```

It differs from [`to_networkx()`](../integrations/networkx.md), which mirrors the
entire database unfiltered.

## API Reference

::: grafito.algorithms.Community

::: grafito.algorithms.compute_centrality

::: grafito.algorithms.detect_communities
