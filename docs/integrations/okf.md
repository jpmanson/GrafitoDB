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
not errors).

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

The summary dict reports the number of `embedded` concepts. This pairs full-text
(`text_search`) and vector (`semantic_search`) retrieval over the same imported
bundle — useful for hybrid agent-memory workflows.

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `link_type` | `"LINKS_TO"` | Relationship type created for intra-bundle markdown links. |
| `configure_fts` | `True` | Configure full-text search over `title`/`description`/`body` (best-effort; skipped if SQLite lacks FTS5). |
| `uri_prefix` | `"okf:"` | Prefix prepended to each concept ID to form the node `uri`. |

The returned summary dict reports `nodes`, `relationships`, `stubs` (nodes
created for links whose target is not in the bundle), and `skipped`
(`index.md`/`log.md` files).

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

kb = OKFBundle.load("examples/okf_knowledge_base", embed=embedder)

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

kb.db.execute("MATCH (n) RETURN count(n)")    # escape hatch: full graph power
kb.save("out/bundle", write_viz=True)         # round-trip back to markdown
```

Mutating a bundle (agent-memory write path):

```python
kb.add_concept("notes/idea", type="Note", title="An idea",
               body="# Notes\n...", tags=["draft"])   # embedded + FTS-indexed
kb.link("notes/idea", "decisions/0001-use-sqlite", anchor="builds on")
kb.cite("notes/idea", "https://example.com/paper", anchor="source")
kb.remove_concept("notes/old")
kb.save()                                              # persist to markdown
```

Round-trip note: `save()` writes each concept's stored `body` verbatim. For a
concept created **without** a body, `link`/`cite` edges are synthesized into
`# Links` / `# Citations` sections on export (so they round-trip). For a concept
**with** a body, include the links/citations in that body if you want them in the
markdown — the edges remain queryable in the graph regardless.

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
- **`Concept`** is a thin view; `concept.node` is the raw grafito node.
- Captures `okf_version` from the root `index.md` (lost by the low-level import).

## Examples

Both runnable examples use the `OKFBundle` façade. `examples/okf_import.py` is a
short intro (load → concept/links → search → save with `viz.html`) over the
tabular sample bundle in `examples/okf_bundle/`:

```bash
python examples/okf_import.py
```

OKF shines on *narrative*, cross-linked knowledge rather than tabular data.
`examples/okf_knowledge_base/` is a small engineering knowledge base —
architecture decision records, an on-call runbook, and glossary terms, all
cross-linked with citations. The script walks the full façade — index/traversal,
the directory tree, aggregation via the `kb.execute` escape hatch, semantic
search, the agent-memory write path, and visualization (it retrieves a "slow
query" runbook for the
query *"how do I make a query run faster"*, which never uses those words):

```bash
python examples/okf_knowledge_base.py
```
