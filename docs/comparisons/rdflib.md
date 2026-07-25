# GrafitoDB vs. rdflib

[rdflib](https://rdflib.readthedocs.io) is the de-facto RDF toolkit for Python: a
triplestore with SPARQL, many serialization formats, and full Semantic-Web
interoperability. GrafitoDB is a **property-graph database** on top of SQLite, queried
with Cypher. They solve overlapping but different problems — this page explains when
each is the better tool, backed by a reproducible benchmark.

!!! tip "Short version"
    Use **rdflib** when you need RDF *as a standard*: SPARQL, named graphs, reasoning,
    Linked-Data interchange. Use **GrafitoDB** when you need a *database for a property
    graph*: durable persistence, bounded memory, and fast indexed lookups/traversals.
    The two interoperate through Grafito's [RDF integration](../integrations/rdf.md).

## Feature comparison

| Capability | rdflib | GrafitoDB |
| --- | --- | --- |
| Data model | RDF triples | Property graph (nodes/edges with properties) |
| Query language | SPARQL 1.1 | Cypher (+ programmatic API) |
| SPARQL over the data | ✅ native | ✅ via [`query_sparql`](../integrations/rdf.md#sparql-queries) (delegated) |
| RDF import/export | ✅ 20+ formats | ✅ Turtle, JSON-LD, N-Triples, RDF/XML, N-Quads, N3, TriG |
| Edge properties | ⚠️ needs reification (~4 triples/edge) | ✅ first-class (one row) |
| Persistence | in-memory (plugins for disk stores) | ✅ single SQLite file, transactional |
| Memory footprint | whole graph in RAM | ✅ streamed from disk |
| Property indexes | — | ✅ used by Cypher/`match_nodes` |
| Full-text & vector search | — | ✅ FTS5 + ANN backends |
| Named graphs / quads | ✅ | — |
| Reasoning (RDFS/OWL) | ✅ (plugins) | — |
| Graph isomorphism / diff | ✅ | ✅ via [`graph_diff`](../integrations/rdf.md#comparing-two-graphs) (delegated) |

As a **pure RDF engine, rdflib is more capable** (native SPARQL, named graphs,
reasoning). As a **database for a property graph, GrafitoDB is substantially more
efficient**, as the benchmark below shows.

## Benchmark

The same knowledge graph (`Person` nodes with `name/age/city/mbox`, `KNOWS` edges with a
`since` property) is stored in both engines and queried with each engine's native
language — **Cypher** for Grafito, **SPARQL** for rdflib. Edges are modelled identically
(the reified form Grafito's exporter produces). Median of 5 runs.

The full, reproducible script lives in
[`playground/rdf_vs_grafito/`](https://github.com/jpmanson/GrafitoDB/tree/main/playground/rdf_vs_grafito).

### 20,000 nodes / ~200,000 edges

| Metric | GrafitoDB | rdflib | Winner |
| --- | --- | --- | --- |
| Build (ingest) | 26.0 s | 23.4 s | rdflib 1.1× |
| On-disk size | 31.9 MB | 24.1 MB | rdflib 1.3× |
| Reopen / reload | **0.4 ms** | 10.4 s | **Grafito** |
| Memory resident | **~0 (on disk)** | 885 MB | **Grafito** |
| Point lookup | **0.08 ms** | 1.04 ms | **Grafito 12×** |
| 1-hop neighbours | **0.14 ms** | 1.33 ms | **Grafito 10×** |
| 2-hop | **0.49 ms** | 2.40 ms | **Grafito 5×** |
| Edge-property filter | **2.37 s** | 3.47 s | **Grafito 1.5×** |
| Degree top-k (aggregation) | 2.25 s | 1.61 s | rdflib 1.4× |

Point lookups are **constant-time** across scales (0.08–0.09 ms at both 5k and 20k
nodes) because Grafito uses its property index; rdflib scans/filters, so its cost grows
with the dataset.

### Reading the results

**GrafitoDB wins decisively on:**

- **Lookups and traversals** — 5–12× faster, and constant-time thanks to indexes.
- **Reopen** — the data is already on disk (~0.4 ms); rdflib must re-parse the whole
  serialization (10 s at 20k nodes).
- **Memory** — Grafito streams from SQLite (~0 resident); rdflib keeps every triple in
  RAM (885 MB for ~900k triples). This gap grows linearly and is structural.
- **Edge-property queries** — native edge properties vs. RDF reification overhead.

**rdflib wins on:**

- **On-disk size** — Turtle/N-Triples are more compact than a SQLite file with indexes.
- **Aggregations** like degree top-k (SPARQL ~1.4×).
- **Bulk build at large scale** — Grafito's insert path scales worse than linear.

## When to use which

- **Choose rdflib** if RDF *is the point*: you consume/publish Linked Data, need SPARQL
  features Cypher lacks, work with named graphs or ontologies/reasoning, or exchange data
  across the Semantic-Web ecosystem.
- **Choose GrafitoDB** if you need a *database*: durable and transactional storage,
  bounded memory on large graphs, fast indexed lookups and multi-hop traversals, first-class
  edge properties, or built-in full-text and vector search.
- **Use both**: model and query in Grafito, then `export_*` / `import_*` to interoperate
  with RDF tools, or run one-off `query_sparql` queries when a SPARQL feature is handy.
  See the [RDF integration guide](../integrations/rdf.md).
