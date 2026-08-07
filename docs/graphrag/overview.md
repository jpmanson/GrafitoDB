# GraphRAG

Retrieval-augmented generation where the retrieval step uses the graph, not just
vector distance. This page is the map: what GrafitoDB provides at each stage of
the pipeline, and which of the three abstraction levels to work at.

## What the Graph Adds

Plain RAG retrieves the top-k chunks nearest a query embedding. That misses two
things the graph knows:

- **The thing that answers the question is often adjacent to the match, not the
  match itself.** A question about a decision retrieves the decision; the
  rationale lives in the document it supersedes.
- **Results are related to each other.** Eight of thirty hits citing one another
  is a signal — about which one is central, and about what to include when the
  budget only fits five.

Everything below is machinery for those two: following edges from what matched,
and using the structure among results.

## Pick a Level

Three layers cover the same pipeline at increasing levels of opinion. Working at
the wrong one is the most common way to make this harder than it is.

| Level | Use when | Entry point |
| --- | --- | --- |
| **Graph** | Your data is already a graph, or you want full control of retrieval and expansion | [`semantic_subgraph()`](../analysis/subgraphs.md) |
| **Documents** | You have files or long text to chunk, retrieve at passage level, and cite | [`DocumentIngestor`](../search/document-chunking.md) |
| **OKF** | You want a governed knowledge base: provenance, trust levels, supersession, review workflow | [`OKFBundle`](../integrations/okf.md) |

They stack rather than compete — OKF is built on documents, documents on the
graph — but you should pick one as your primary interface and drop down only
when you need to.

### Graph

You hold the nodes and edges; GrafitoDB supplies retrieval and expansion over
them. Nothing is written on your behalf, so the labels, properties and
relationship types stay exactly what you designed — and everything else on this
page (`semantic_search`, `hybrid_subgraph`, `communities`, Cypher vector search)
works directly on them.

Pick this level when the entities *are* the unit of retrieval: papers, tickets,
people, products. The cost is that chunking, reading order, ids that survive
re-ingest and citation offsets are yours to build if the corpus is long text.

### Documents

[`DocumentIngestor`](../search/document-chunking.md) turns text into a managed
passage graph and gives back the four verbs a passage-level RAG needs:

| Verb | What you get |
| --- | --- |
| `ingest(text, document_key=…)` | `Document` → `DocumentVersion` → `Chunk` nodes, plus a `Section` tree (`HAS_SECTION`) when the text is markdown-shaped and `hierarchy` is on. Embeds in one batched call if `embed_index` is set |
| `search()` / `hybrid_search()` | Top-k over *this* helper's passages only — filtered to the managed nodes, the active generation and `embed_role=passage`, so an unrelated `Chunk` label in the same database cannot leak in |
| `expand(hit.node, window=1)` | The neighbouring passages by reading order (`global_seq`), optionally the section path above them (`include_ancestors=True`) |
| `pack(ctx, max_tokens=…)` | A budgeted string plus `PackedSegment`s carrying `node_id`, `char_start`/`char_end` and `section_path` — the offsets a citation needs |

What you are really buying is the bookkeeping around those verbs:

- **Ownership.** Every node it writes carries `managed_by="grafito.document"`,
  `owner_document_id`, `generation` and `role`. Deletes and rebuilds are scoped
  to those markers, so the helper never touches nodes it did not create. A
  pre-existing node passed as `parent_id=` is attached, never owned or deleted —
  that is how an OKF concept gets passages hung off it.
- **Idempotent re-ingest.** `ingest()` fingerprints the text (plus the embed
  flag and the view set). Unchanged input returns `IngestResult(skipped=True)`
  without re-embedding. Changed input builds the new generation as `BUILDING`,
  flips it to `ACTIVE`, then garbage-collects the previous one — searches never
  observe a half-written document.
- **Reading order as structure.** `global_seq` on each passage, and by default a
  `Chunk_i -[:NEXT_PASSAGE]-> Chunk_{i+1}` chain for Cypher and visualisation.
  `expand()` windows by `global_seq`, not by those edges.
- **A navigable table of contents.** `toc(document_key)` returns titles, levels
  and `node_key`s without bodies; `load_sections()` fetches the bodies for the
  keys you chose; `tree_select()` lets a model do the choosing. This is the
  cheap path when the document is long but well-structured — no embedding of the
  whole thing required.

Chunker choice, multi-view indexing, enrichment and the overlap-dedup rules
behind `pack()` are all on the [Document → Graph
Chunking](../search/document-chunking.md) page; that page is the reference, this
one is the map.

Drop down to the graph level when you want a traversal the four verbs do not
express — the passage nodes are ordinary nodes, and `db.subgraph()`,
`db.path_context()` and Cypher all reach them.

### OKF

Everything at the document level, plus governance: frontmatter, citations,
layers, trust levels, supersession and a review workflow. Filters like
`min_trust` hold through *expansion*, not just retrieval, which is the part that
is tedious to rebuild by hand. It is the heaviest level, and the right one only
when the corpus needs to be curated rather than merely indexed — see
[OKF](../integrations/okf.md).

## The Pipeline

### 1. Ingest

| Tool | For |
| --- | --- |
| [`db.index_documents()`](../search/bulk-ingest.md) | Row-shaped data: HuggingFace datasets, DataFrames, JSONL. Nodes, edges and batched embeddings in one call |
| [`DocumentIngestor.ingest()`](../search/document-chunking.md) | Long text that needs chunking, with a section tree and stable ids across re-ingests |
| [Chunkers](../search/document-chunking.md#choosing-a-chunker) | `fixed`, `recursive`, `markdown`, `semantic`, plus a `chonkie` adapter |
| `OKFBundle.add_concept()` | Governed concepts with frontmatter, citations and layers |

Register an FTS index alongside the vector index if you want hybrid retrieval —
`db.create_text_index("node", "Doc", ["text"])`.

### 2. Retrieve

| Tool | For |
| --- | --- |
| `db.semantic_search()` | Vector top-k |
| `db.text_search()` | FTS5/BM25 |
| [`CALL db.vector.search`](../cypher/vector-search.md) | Vector search *inside* a Cypher pattern — seeds a traversal from the ANN index |
| [`SIMILAR()` / `VECTOR_SCORE()`](../cypher/vector-search.md) | Constrain the far end of a pattern by similarity |
| `db.hybrid_search()` | Vector + lexical fused with RRF. Usually the right default: the two modes fail differently |
| `DocumentIngestor.hybrid_search()` | The same fusion at passage level, with ownership filters |
| `OKFBundle.search()` | Retrieval with governance filters (layer, trust, superseded) |

The Cypher-level tools are what make retrieval structural rather than a
pre-filter. Both endpoints of a path can be seeded semantically:

```cypher
CALL db.vector.search('papers_vec', 'chatgpt', 10) YIELD node AS a
CALL db.vector.search('papers_vec', 'anthropic', 10) YIELD node AS b
MATCH p=(a)-[:CITES*1..3]->(b)
RETURN p LIMIT 10
```

### 3. Expand

| Tool | For |
| --- | --- |
| [`db.hybrid_subgraph()`](../analysis/subgraphs.md) | Fused hits plus their neighbourhood, with `hops`/`scores` provenance |
| `db.semantic_subgraph()` / `db.text_subgraph()` | The same, seeded by one retrieval mode |
| `db.subgraph()` | The same, from any seeds — including a fusion of your own |
| [`db.path_context()`](../analysis/subgraphs.md#context-from-routes-not-rankings) | The routes *between* concepts, when the answer is what connects them |
| `DocumentIngestor.expand()` | Passage-level: siblings, parents, and surrounding sections |
| `OKFBundle.context()` | Expansion governed by filters, so a `min_trust` guarantee holds through links |

Expansion is where the guard rails matter. One hop from a hub reaches most of the
database; `exclude_rel_types` and `max_nodes` are not optional in a real corpus.

### 4. Rerank

| Tool | For |
| --- | --- |
| `LexicalReranker` | Offline, no dependencies. A sane default |
| `CrossEncoderReranker` | Local cross-encoder. The usual quality winner |
| `CohereReranker`, `VoyageReranker`, `JinaReranker` | Hosted reranking APIs |
| `rrf_fuse()` | Reciprocal rank fusion across retrieval strategies |
| `db.centrality(graph=...)` | Graph-aware: rank by position within the retrieved subgraph |

All live in `grafito.okf.rerank` except `rrf_fuse` (`grafito.document.hybrid`).
Rerankers plug into `OKFBundle.context(rerank=...)`, `semantic_search(reranker=...)`
and `CALL db.vector.search(..., {reranker: 'name'})`.

Graph-aware reranking is worth trying but not worth assuming: a node with a
middling vector score that everything else in the result set points at is often
the right answer — and often noise. Measure it (see [below](#what-is-missing)).

### 5. Pack

| Tool | For |
| --- | --- |
| `OKFBundle.context()` | Search → expand → rerank → pack to a token budget, with citations and per-block provenance (`via JOINS_WITH`) |
| `DocumentIngestor.pack()` | Passage-level packing with citations |
| `Subgraph` | Raw material when you want to build the prompt yourself |

`context()` is the most complete one-shot path: it names the relationship that
pulled each block in, so the model can cite structure and not just text, and its
filters govern expansion as well as retrieval.

### 6. Serve to a Model

| Tool | For |
| --- | --- |
| `GraphTools` | `graph_schema`, `graph_neighbors`, `text_search`, `vector_search` |
| `CypherTools` | `graph_query` — read-only Cypher escape hatch |
| `DocumentTools` | `document_context`, `document_search`, `document_expand`, `document_toc`, `document_load_sections`; writes opt-in |
| `grafito-mcp` | Serves any of the above over MCP stdio |
| `run_agent()` | In-process agent loop with `OpenAIChat` / `AnthropicChat` |

One-shot `context()` and an agentic loop are not equivalent in cost: measured on
a real model, letting the agent drive retrieval cost ~3.7x the one-shot pack.
That is structural — resent conversation state — not a framework problem. Reach
for the agent when the query genuinely needs several retrieval rounds.

### 7. Understand the Corpus

| Tool | For |
| --- | --- |
| [`db.communities()`](../analysis/graph-algorithms.md) | Thematic clusters, optionally labelled with `label_terms` |
| `db.centrality()` | Which documents are structurally central |
| [`db.create_semantic_graph()`](../analysis/semantic-graph.md) | Materialise similarity as edges — the prerequisite for clustering by similarity |
| `export_graph()` | Visualise a subgraph (`grafito.integrations.viz`) |

## Recipes

### Passage-level RAG with citations

```python
db.create_vector_index("passages", dim=384, embedding_function=embedder)
ingestor = DocumentIngestor(db, embed_index="passages", configure_fts=True)
ingestor.ingest(Path("adr-042.md").read_text(), document_key="adr-042")

hits = ingestor.hybrid_search("why did we drop the queue?", k=10)
pack = ingestor.pack(ingestor.expand(hits[0].node, window=1), max_tokens=4000)
print(pack.text)
```

`expand()` takes one centre passage, not the whole hit list — window each hit you
want to keep, then pack the union.

### Retrieve as a graph, rank within the result

```python
sub = db.semantic_subgraph("autonomous agents", k=50, expand=1,
                           exclude_rel_types=["SEMANTIC_SIMILAR"])
top = db.centrality("pagerank", graph=sub.to_networkx(), limit=10)
```

### Governed context for a prompt

```python
bundle = OKFBundle.load("kb/", embed=embedder)
pack = bundle.context(
    "retry policy for payments",
    budget_tokens=4000,
    expand_hops=1,
    min_trust="human-reviewed",
    rerank=CrossEncoderReranker(),
)
prompt = str(pack)
```

### Thematic map of a corpus

```python
db.create_semantic_graph(k=15, min_score=0.3, symmetrize="mutual")
for c in db.communities("louvain", rel_types=["SEMANTIC_SIMILAR"],
                        weight_property="score", seed=42, label_terms=4):
    print(f"[{c.label}] — {c.size} documents")
```

`symmetrize="mutual"` keeps only reciprocated neighbours, which cuts the
cross-cluster edges that blur communities — at the cost of needing a larger `k`.

### How two concepts connect

```python
sub = db.path_context(
    ["retry policy", "payment provider outage"],
    max_hops=3,
    exclude_rel_types=["SEMANTIC_SIMILAR"],
)
for route in sub.paths:
    print(" → ".join(db.get_node(n).properties["title"] for n in route))
```

An empty result means they are not connected within `max_hops` — which is an
answer, and one a top-k search would have hidden behind plausible-looking hits.

## What Is Missing

Retrieval quality is still unmeasured. There is an
[invariant harness](https://github.com/jpmanson/GrafitoDB/tree/main/tests/invariants)
covering the *structural* properties — that a rebuild is atomic, that a refresh
converges, that no node is stranded — and it has caught real defects. But the
knobs on this page (`expand` depth, reranker choice, `min_score`, RRF weights,
`symmetrize`, whether graph-aware reranking helps at all) change retrieval
*quality* in ways nothing here measures.

Build a small labelled set of queries and expected documents for your corpus
before tuning. Without one you are guessing, and the guesses tend to favour
whatever is most elaborate rather than what works.
