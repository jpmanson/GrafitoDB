# Open Knowledge Format (OKF)

[Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
(OKF) is an open, human- and agent-friendly format for representing *knowledge*:
a directory tree of markdown files with YAML frontmatter. There is no schema
registry and no required tooling — if you can `git clone` a repo, you can ship
an OKF bundle.

GrafitoDB can **import** an OKF bundle into a queryable property graph and
**export** a graph back to an OKF bundle. This makes OKF a durable, diffable,
human-readable storage layer while GrafitoDB provides Cypher, full-text, and
semantic search on top — a natural fit for **agent memory**: persist knowledge
as markdown in git, index and query it at runtime.

!!! tip "Start with `OKFBundle`"
    [`OKFBundle`](#high-level-api-okfbundle) is the recommended high-level entry
    point: load a bundle, navigate concepts/links/citations, search by meaning,
    assemble grounded [`context()`](#grounded-context-for-agents-context) for an
    agent prompt, and write memory back — with the raw graph one attribute away
    via `bundle.db`. The functions below are the low-level layer it delegates to.

## Prerequisites

```bash
# PyYAML is included by default
pip install grafito
```

## How concepts map to the graph

| OKF | GrafitoDB |
|-----|-----------|
| Concept (a `.md` file) | Node |
| `type` (frontmatter, required) | Node label |
| `title`, `description`, `resource`, `tags`, `timestamp`, extra keys | Node properties |
| Concept ID (e.g. `tables/orders`) | Node `uri` (`okf:tables/orders`) |
| Markdown body | `body` property (feeds full-text search) |
| Markdown link `[x](/tables/y.md)` | Relationship (`LINKS_TO` by default) |
| Link under `# Citations` | `CITES` relationship (to a concept or a `Reference` node) |
| `index.md` / `log.md` | Skipped (reserved, derivable) |

Concepts without a `type` and links to not-yet-written concepts fall back to the
generic `Concept` label (permissive consumption — broken links are tolerated,
not errors). A file whose frontmatter is not valid YAML does not abort the
import either: its full text is kept as the body and its ID is reported under
the summary's `malformed` key. The whole import runs in a single transaction,
so large bundles load fast and a hard failure leaves the database untouched.

## Importing a bundle

```python
from grafito import GrafitoDatabase

db = GrafitoDatabase(":memory:")
summary = db.import_okf_bundle("path/to/bundle")
print(summary)  # {'nodes': 8, 'relationships': 9, 'stubs': 3, 'skipped': 6}
```

Once imported, the knowledge is fully queryable:

```python
# Cypher: what does the Orders table link to?
db.execute("""
    MATCH (a {title: 'Orders'})-[:LINKS_TO]->(b)
    RETURN b.title AS target
""")

# Full-text search over titles, descriptions, and bodies
db.text_search("customer", k=5)
```

### Semantic search

Pass an embedding function to embed each concept into a vector index at import
time. Concepts are embedded from their `title`, `description`, and `body`, so
you can query the bundle by meaning rather than keywords:

```python
from grafito.embedding_functions import SentenceTransformerEmbeddingFunction

embedder = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
db.import_okf_bundle("path/to/bundle", embed=embedder)

# Query by meaning; the index already knows how to embed the query text
db.semantic_search("how do customers pay for orders", index="okf", k=5)
```

Relevant import options:

| Argument | Default | Description |
|----------|---------|-------------|
| `embed` | `None` | An `EmbeddingFunction`; when set, concepts are embedded for semantic search. |
| `embed_index` | `"okf"` | Name of the vector index created for concept embeddings. |
| `embed_fields` | `("title", "description", "body")` | Concept fields concatenated into the embedded document. |
| `embed_backend` | `"bruteforce"` | Vector index backend (the default needs no extra dependencies). |
| `embed_options` | `None` | Extra options for the vector index — `{"store_embeddings": True}` persists the vectors in the database for reuse across sessions; `{"index_path": ...}` places a file-backed index (faiss/hnswlib/...). |

The summary dict reports the number of `embedded` concepts. This pairs full-text
(`text_search`) and vector (`semantic_search`) retrieval over the same imported
bundle — useful for hybrid agent-memory workflows.

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `link_type` | `"LINKS_TO"` | Relationship type created for intra-bundle markdown links. |
| `configure_fts` | `True` | Configure full-text search over `title`/`description`/`body` (best-effort; skipped if SQLite lacks FTS5). |
| `uri_prefix` | `"okf:"` | Prefix prepended to each concept ID to form the node `uri`. |
| `progress_every` | `None` | Print a progress line every N concept files (and per phase) — for large bundles. |
| `progress` | `None` | Callback `(phase, count)` invoked instead of printing; phases are `concepts`, `links`, `citations`, `embedded`, `done`. |

The import runs in a single transaction and creates an expression index on
`concept_id`, so concept lookups stay fast as bundles grow.

The returned summary dict reports `nodes`, `relationships`, `stubs` (nodes
created for links whose target is not in the bundle), `skipped`
(`index.md`/`log.md` files), and `malformed` (concept IDs whose frontmatter
was not valid YAML and was imported as plain body text).

## Validating a bundle

`validate_okf_bundle` is the linter counterpart to the importer's permissive
consumption: it checks a bundle against the OKF v0.1 conformance rules
(SPEC §9) without importing anything and without stopping at the first bad
file:

```python
from grafito.okf import validate_okf_bundle

report = validate_okf_bundle("path/to/bundle")
report["conformant"]   # True when there are no errors
report["errors"]       # [{'path', 'error'}]  — missing frontmatter block,
                       # unparseable YAML, missing/empty required `type`
report["warnings"]     # [{'path', 'warning'}] — broken intra-bundle links,
                       # frontmatter in a non-root index.md
```

Errors are conformance failures; warnings are soft guidance a consumer must
tolerate (a broken link may simply be not-yet-written knowledge).

## Exporting a bundle

The inverse operation serializes the graph back to OKF markdown:

```python
db.export_okf_bundle("out/bundle", write_viz=True)
# {'concepts': 8, 'skipped': 0, 'viz': True}
```

This writes:

- one markdown file per node (label → `type`, properties → frontmatter, `body`
  → markdown body);
- per-directory `index.md` files for progressive disclosure — the root index
  lists child directories under `# Subdirectories`, and each directory groups
  its concepts by `type`;
- an optional self-contained `viz.html` graph viewer (`write_viz=True`).

Stub nodes (created for broken links during import) are **not** exported. Nodes
created programmatically without a stored `body` get a synthesized `# Links`
section listing their outgoing relationships.

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `uri_prefix` | `"okf:"` | Prefix used to recover concept IDs from node URIs. Should match the import value. |
| `write_index` | `True` | Generate per-directory `index.md` files. |
| `write_viz` | `False` | Also emit a self-contained `viz.html` at the bundle root. |
| `prune` | `False` | Delete concept `.md` files that no longer correspond to a node (directories left empty are removed). `log.md` and non-markdown files are never touched. `OKFBundle.save()` prunes by default so removals round-trip. |

## Round-trip and agent memory

Import → query/enrich → export is lossless for the graph structure and
preserves unknown frontmatter keys:

```python
db = GrafitoDatabase(":memory:")
db.import_okf_bundle("bundle")

# ... query, traverse, or add knowledge via Cypher / the programmatic API ...
db.execute("CREATE (n:Playbook {title: 'New runbook', body: 'Steps...'})")

# Persist the enriched knowledge back to markdown (commit it to git)
db.export_okf_bundle("bundle")
```

!!! note "Multi-label nodes"
    OKF concepts have a single `type`. When a node has several labels, the
    exporter uses the first label as `type`. See `todo/okf/IMPROVEMENTS.md` for
    the open design question on representing multi-label nodes.

## High-level API: `OKFBundle`

The functions above are the low-level layer. `grafito.okf.OKFBundle` is an
OKF-flavored façade over them: it speaks concepts/links/citations/layers instead
of nodes/relationships, while exposing the full graph via `bundle.db`.

```python
from grafito.okf import OKFBundle

kb = OKFBundle.load("examples/okf/okf_knowledge_base", embed=embedder)

kb.layers()                                  # {'decisions': 3, 'glossary': 3, 'runbooks': 1}
kb.index()                                    # root index.md, in memory (subdirs)
kb.index("decisions")                         # a directory's listing: title+description, no bodies
c = kb.concept("decisions/0003-vector-search")
c.title                                       # 'Add optional vector search'
c.links()                                     # [Concept, ...] outgoing LINKS_TO
c.cites()                                     # [{'url'|'concept', 'anchor'}, ...]

kb.search("how do I make a query run faster", k=3)        # semantic / text / hybrid
kb.search("make it faster", layer="decisions")            # scoped to a layer
kb.search("vector similarity", mode="hybrid")             # RRF fusion of FTS + vector
# hybrid degrades to text-only when the bundle was loaded without embed=

kb.db.execute("MATCH (n) RETURN count(n)")    # escape hatch: full graph power
kb.save("out/bundle", write_viz=True)         # round-trip back to markdown
```

### Grounded context for agents: `context()`

`search()` returns ranked hits; `context()` turns them into a prompt. It is the
framework-agnostic bridge to *any* agent loop — no LangChain, LlamaIndex, or SDK
required. Given a question it:

1. **seeds** retrieval with `search()` (semantic / text / hybrid);
2. **graph-expands** — follows each hit's outgoing `LINKS_TO` within
   `expand_hops`, so the pack carries linked context the embedding alone would
   miss (the GraphRAG edge over a flat vector store);
3. **packs** the concepts into a token budget as titled, cited blocks, greedily
   in priority order (the top hit is never dropped — it is truncated if it alone
   exceeds the budget).

```python
pack = kb.context("how do I make a query run faster", budget_tokens=2000)

str(pack)          # prompt-ready text (same as pack.text)
pack.citations     # [{'url'|'concept', 'anchor', 'cited_by': [...]}, ...] — deduped
pack.concepts      # the Concepts that made it into the budget, in order
pack.hits          # the seed search Hits (scores/provenance)
pack.tokens        # estimated token count of the packed text
pack.truncated     # True if anything was dropped/cut to fit

prompt = f"Answer using only this context:\n\n{pack}"   # drops straight into a prompt
```

| Argument | Default | Description |
|----------|---------|-------------|
| `budget_tokens` | `2000` | Token budget for the packed text. |
| `k` | `8` | Seed hits to retrieve before expansion. |
| `mode` | `"auto"` | `"semantic"` / `"text"` / `"hybrid"` / `"auto"`. |
| `type`, `layer` | `None` | Restrict retrieval to a concept type / directory layer. |
| `expand_hops` | `1` | Outgoing `LINKS_TO` hops to graph-expand (`0` disables). |
| `include_citations` | `True` | Render `Sources:` lines and collect `pack.citations`. |
| `token_counter` | heuristic | Callable `str -> int`; default ≈ 4 chars/token. Pass your model's tokenizer for exact budgeting. |
| `rerank` | `None` | An optional reranker (see below). |

### Reranking

A bi-encoder (embedding) retrieves cheaply but coarsely. A **reranker** re-scores
candidates against the query *text* — the standard RAG precision step. In
`context()` it matters most: graph expansion deliberately pulls in loosely
related neighbours, and the reranker decides which of them deserve the token
budget. It runs over the seed + expanded pool **before** packing.

A `Reranker` is **any callable** `(query, candidates) -> [(concept, score), ...]`
(most relevant first) — inject your own, or use one of the bundled ones:

```python
from grafito.okf import (
    LexicalReranker,        # dependency-free (query-term overlap); offline default
    CrossEncoderReranker,   # local HuggingFace cross-encoder (sentence-transformers)
    CohereReranker,         # Cohere rerank API
    VoyageReranker,         # Voyage AI rerank API
    JinaReranker,           # Jina AI rerank API
)

kb.context(question, rerank=LexicalReranker())                          # offline, no deps
kb.context(question, rerank=CrossEncoderReranker("BAAI/bge-reranker-base"))  # local HF
kb.context(question, rerank=CohereReranker())                          # needs COHERE_API_KEY

# Custom: any matching callable works — no subclassing required.
def my_reranker(query, candidates):
    return sorted(((c, score(query, c)) for c in candidates), key=lambda p: -p[1])

kb.context(question, rerank=my_reranker)
```

The API rerankers (`Cohere`/`Voyage`/`Jina`) need `httpx` and read their API key
from the matching environment variable (e.g. `COHERE_API_KEY`) or an explicit
`api_key=`. Requests use a 30s timeout (`timeout=`), and the instances are
context managers (`close()` releases the HTTP client). `CrossEncoderReranker`
needs `sentence-transformers` but runs offline. A reranker may return a subset
(e.g. its own `top_n`); `context()` packs exactly the order and subset it
returns.

Mutating a bundle (agent-memory write path):

```python
kb.add_concept("notes/idea", type="Note", title="An idea",
               body="# Notes\n...", tags=["draft"])   # embedded + FTS-indexed
kb.update_concept("notes/idea", body="# Notes\nRevised…",
                  status="reviewed")                   # partial update; re-embeds
kb.update_concept("notes/idea", description=None)      # None removes a field
kb.link("notes/idea", "decisions/0001-use-sqlite", anchor="builds on")
kb.cite("notes/idea", "https://example.com/paper", anchor="source")
kb.remove_concept("notes/old")
kb.save()                                              # persist to markdown
```

`update_concept` changes only the fields you pass (including `type`, which
relabels the node, and any producer-defined frontmatter key); the FTS index and
the vector embedding follow automatically. `save()` mirrors the graph to disk:
files for removed concepts are pruned so `remove_concept` round-trips (pass
`prune=False` to only add/overwrite).

Round-trip note: `save()` writes each concept's stored `body` verbatim. For a
concept created **without** a body, `link`/`cite` edges are synthesized into
`# Links` / `# Citations` sections on export (so they round-trip). For a concept
**with** a body, include the links/citations in that body if you want them in the
markdown — the edges remain queryable in the graph regardless.

### Persistent reuse across sessions

`load()` parses markdown and (optionally) embeds every concept — work you only
want to do once. Back the bundle with a database *file* and persist the
embeddings, then later sessions `open()` the file directly: no markdown
parsing, no re-embedding.

```python
# Session 1 — import once, persist graph + embeddings.
kb = OKFBundle.load(
    "path/to/bundle",
    db=GrafitoDatabase("kb.db"),
    embed=embedder,
    embed_options={"store_embeddings": True},
)

# Session 2+ — open the database file; the vector index rehydrates from it.
kb = OKFBundle.open(GrafitoDatabase("kb.db"), source_path="path/to/bundle")
kb.search("how do I make a query run faster")   # semantic, no re-embedding
```

Pass `embed=` to `open()` only when the embedding function is a custom one the
registry cannot rebuild by name (built-ins such as the SentenceTransformer
function are rehydrated automatically from the index metadata). `source_path`
is optional; it sets the default `save()` target. For very large indexes,
prefer a file-backed ANN backend via `embed_backend="hnswlib"` (or `faiss`)
plus `embed_options={"index_path": "kb.hnswlib"}`.

Materializing the directory tree and history (opt-in) lets you traverse the
hierarchy as a graph and query the changelog:

```python
kb = OKFBundle.load("bundle", directory_nodes=True, import_log=True)

kb.children()                 # {'subdirs': ['decisions', ...], 'concepts': [...]}
kb.children("decisions")      # one level down, via CONTAINS edges
kb.log()                      # all log.md entries, newest first
kb.log("decisions/0001-use-sqlite")   # entries that mention this concept
```

`directory_nodes=True` adds `Directory` nodes + `CONTAINS` edges (root → subdir →
concept); `import_log=True` adds `LogEntry` nodes linked to mentioned concepts via
`MENTIONS`. Both are synthesized/derived and are skipped on export.

Design notes:

- **Delegates, never duplicates** — `load`/`save` call `import_okf_bundle` /
  `export_okf_bundle`; the low-level API stays the canonical implementation.
- **`search()` unifies** grafito's text and vector results into a single `Hit`
  (`hit.concept`, `hit.score`, `hit.via`); `mode="auto"` uses vectors when the
  bundle was loaded with `embed=`, else full-text.
- **`context()` is framework-agnostic** — it returns prompt-ready text plus
  citations, not a framework-specific object; the `rerank=` hook is any callable.
- **`Concept`** is a thin view; `concept.node` is the raw grafito node.
- **Lookups scale** — `concept()`, `concepts()`, `layers()`, `index()` and
  `len()` filter in SQL (backed by an expression index on `concept_id`); only
  matching nodes are hydrated. `layer=` accepts nested paths
  (`"references/joins"`), matching any concept below that directory.
- Captures `okf_version` from the root `index.md` (lost by the low-level import).

## Examples

Both runnable examples use the `OKFBundle` façade. `examples/okf/okf_import.py` is a
short intro (load → concept/links → search → save with `viz.html`) over the
tabular sample bundle in `examples/okf/okf_bundle/`:

```bash
python examples/okf/okf_import.py
```

OKF shines on *narrative*, cross-linked knowledge rather than tabular data.
`examples/okf/okf_knowledge_base/` is a small engineering knowledge base —
architecture decision records, an on-call runbook, and glossary terms, all
cross-linked with citations. The script walks the full façade — index/traversal,
the directory tree, aggregation via the `kb.execute` escape hatch, semantic
search, grounded `context()` assembly with a reranker, the agent-memory write
path, and visualization (it retrieves a "slow query" runbook for the query
*"how do I make a query run faster"*, which never uses those words):

```bash
python examples/okf/okf_knowledge_base.py
```
