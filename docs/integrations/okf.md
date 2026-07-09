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
| `typed_links` | `False` | Derive the relationship type from the heading a link sits under — a link under `# Joins with` becomes a `JOINS_WITH` relationship. Links before any heading, under `# Links`, or under headings that don't normalize to a valid type keep `link_type`. |
| `wikilinks` | `False` | Also resolve Obsidian-style `[[Note]]` links (see [Obsidian wikilinks](#obsidian-wikilinks)). |
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

### Obsidian wikilinks

An Obsidian vault is already an OKF bundle without a plugin — directories of
markdown files with YAML frontmatter. What Obsidian adds is its own link
syntax: `[[Note]]`, `[[Note|Alias]]`, `[[Note#Heading]]`. Pass
`wikilinks=True` to resolve those alongside plain markdown links:

```python
db.import_okf_bundle("path/to/obsidian-vault", wikilinks=True)
```

Obsidian links by note title, not by path, so resolution is vault-wide rather
than relative to the linking file:

1. An exact concept-ID match first — `[[decisions/0001-use-sqlite]]` works
   like a normal absolute markdown link.
2. Otherwise, a case-insensitive match against every concept's *basename*
   (filename without the directory) — `[[0001-use-sqlite]]` resolves the same
   way, from any file in the vault.
3. A basename shared by more than one concept is **ambiguous** and is
   skipped rather than guessed (no relationship, no stub).
4. A target matching nothing becomes a stub keyed by the literal link text —
   Obsidian's own "red link" (not-yet-written note) convention. If a note
   with that exact title is added later, the stub is promoted just like a
   broken markdown link's would be (see [Incremental import](#incremental-import)).

`[[Note|Alias]]` uses `Alias` as the relationship's `anchor` (falling back to
`Note` without one); the `#Heading` fragment is dropped, same as for markdown
links. Wikilinks participate in `typed_links` the same way markdown links do.
Only the main body is scanned — a `[[Note]]` under `# Citations` is not
picked up (citations resolve by URL or markdown link, not by title).

### Incremental import

Re-importing a bundle normally reparses and re-embeds every file. For a large
or frequently-updated bundle, pass `incremental=True` to make re-imports cheap:

```python
db.import_okf_bundle("path/to/bundle", embed=embedder, incremental=True)
# ... edit a couple of files in the bundle directory ...
summary = db.import_okf_bundle("path/to/bundle", embed=embedder, incremental=True)
print(summary["unchanged"], summary["updated"], summary["nodes"])
```

Each concept's raw file content is hashed (`okf_hash`, stored on the node and
excluded from exported frontmatter) and compared against the hash recorded on
its last import:

- **Unchanged** files are skipped entirely — no re-parsing, no re-embedding,
  no relationship churn. This is the main cost saved on a re-import.
- **Changed** files are updated *in place*: the same node ID is kept (so
  relationships from unchanged concepts pointing at it stay valid), its own
  outgoing links/citations are regenerated, and it is re-embedded. Edges added
  by the trust model (`OKFBundle.supersede`/`conflicts_with` —
  `SUPERSEDES`/`CONFLICTS_WITH`) are left untouched.
- **New** files are created as usual. A link that previously created a stub
  (SPEC §5.3) is promoted in place when the target file is later added,
  rather than creating a duplicate node.
- Pass `prune=True` (requires `incremental=True`) to also delete nodes whose
  concept file was removed from the bundle since the last import — mirrors
  the exporter's `prune` option.

The summary dict gains `unchanged`, `updated`, and `pruned` counts, and
`references`/`stubs` count only genuinely new nodes (existing `Reference`/stub
nodes are reused, not duplicated).

`directory_nodes`/`import_log` are not incremental-aware: combining them with
`incremental=True` duplicates directory/log nodes on every re-import.

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

### Layered linting: `lint_okf_bundle`

`validate_okf_bundle` only checks hard SPEC conformance. `lint_okf_bundle`
wraps it and adds two more layers — a three-tier model (Core / Profile /
Hygiene):

```python
from grafito.okf import lint_okf_bundle

report = lint_okf_bundle("path/to/bundle", profile="profile.yaml")
report["conformant"]   # Core has no errors AND no Profile rule severity="error" fired
report["core"]         # {"errors", "warnings"} — identical to validate_okf_bundle
report["profile"]      # [{"path", "rule", "message", "severity"}, ...]
report["hygiene"]      # [{"path", "rule", "message"}, ...] — always advisory
```

**Profile** rules are bundle-specific and come from a manifest — a dict, or a
path to a YAML file:

```yaml
rules:
  - id: adr-requires-status
    applies_to: ADR            # a type name, a list of types, or "*" (default)
    require_field: status      # missing, or empty ("" / [] / {}) unless non_empty: false
    severity: error             # "error" blocks `conformant`; "warning" (default) doesn't
  - id: title-max-length
    field: title
    max_length: 80
    severity: warning
```

A rule may combine more than one check: `require_field` (+ `non_empty`),
`forbid_field`, `max_length` (+ `field`), `allowed_values` (+ `field`),
`pattern` (a regex, + `field`).

**Hygiene** is a fixed set of best-practice checks for a *knowledge graph*
specifically — not customizable, and never blocks `conformant`:
`missing-title`, `missing-description`, `short-body` (main body under
`short_body_chars`, default 40, excluding citations), `orphan-concept` (no
intra-bundle links in or out — disconnected from the graph), and
`duplicate-title` (two concepts sharing a title).

`mode="audit"` (default) is the human-facing report — all three layers.
`mode="validate"` drops Hygiene, for a CI gate that only cares about
conformance:

```python
report = lint_okf_bundle("path/to/bundle", profile="profile.yaml", mode="validate")
assert report["conformant"], report["core"]["errors"] + report["profile"]
```

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
created programmatically without a stored `body` get synthesized link sections
from their outgoing relationships: `LINKS_TO` edges under `# Links`, and every
other type under a heading derived from it (`JOINS_WITH` → `# Joins with`), so
typed relationships round-trip through markdown when re-imported with
`typed_links=True`.

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `uri_prefix` | `"okf:"` | Prefix used to recover concept IDs from node URIs. Should match the import value. |
| `write_index` | `True` | Generate per-directory `index.md` files. |
| `write_viz` | `False` | Also emit a self-contained `viz.html` at the bundle root. |
| `write_log` | `True` | Regenerate per-scope `log.md` files from the graph's `LogEntry` nodes (imported history plus `log_entry`/autolog additions). Scopes without entries are left alone — an existing `log.md` is never blanked. |
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
    exporter uses the first label as `type`; representing multi-label nodes is
    an open design question with no OKF-side convention yet.

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
c.links()                                     # [Concept, ...] any outgoing link type
c.links(type="JOINS_WITH")                    # restrict to one type (typed_links bundles)
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
2. **graph-expands** — follows each hit's outgoing links (any relationship
   type except `CITES`, including typed links) within `expand_hops`, so the
   pack carries linked context the embedding alone would miss (the GraphRAG
   edge over a flat vector store);
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
| `expand_hops` | `1` | Outgoing link hops to graph-expand (`0` disables); follows any relationship type except `CITES`. |
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

### Trust model: `supersede()` and `conflicts_with()`

An agent writing memory unattended can silently overwrite a correct claim with
a hallucinated one if edits always land in place. `supersede()` and
`conflicts_with()` give the write path the **append-only-on-meaning** discipline
from the OKF trust model: corrections create a new concept and link it to the
old one, rather than rewriting the old one's meaning.

```python
new = kb.add_concept("decisions/0002-use-hnswlib", type="Decision",
                      title="Use hnswlib for ANN", body="# Context\n...")
kb.supersede("decisions/0001-use-bruteforce", new, note="scaled past 50k vectors")

kb.conflicts_with("glossary/latency", "glossary/throughput",
                   note="one source defines these interchangeably")
```

`supersede(old, new)` sets `status="superseded"` / `superseded_by` on `old`,
appends to `supersedes` on `new`, and links `new -[:SUPERSEDES]-> old` (typed,
so it round-trips with `typed_links=True`). It does **not** delete or rewrite
`old` — the retracted claim stays inspectable via `kb.concept(old_id)`, `git
blame`, and `kb.log()`.  `conflicts_with(a, b)` is the softer, symmetric
sibling for when new information contradicts an existing concept without
strong enough evidence to supersede it outright: it links both concepts via
`CONFLICTS_WITH` (both directions — a conflict has no natural direction)
without changing either.

`search()` and `context()` exclude `status="superseded"` concepts by default
(`include_superseded=True` opts back in), so retrieval never hands an agent a
retracted claim as if it were current truth — while `concepts()`/`concept()`
still surface them for provenance and history browsing.

```python
kb.concept("decisions/0001-use-bruteforce").is_superseded   # True
kb.concept("decisions/0001-use-bruteforce").superseded_by   # 'decisions/0002-use-hnswlib'
new.supersedes                                               # ['decisions/0001-use-bruteforce']
kb.concept("glossary/latency").conflicts()                   # [Concept('glossary/throughput')]
```

### Review queue: `propose()`, `approve()`, `reject()`

`add_concept()` always writes immediately — right for a human curating the
bundle, or a pipeline you trust. An autonomous agent proposing *new* facts is
a different trust level: it might be duplicating something that already
exists, or contradicting it under a different id. `propose()` is the
agent-facing entry point that gates on that:

```python
result = kb.propose("decisions/0004-use-hnswlib", type="Decision",
                     title="Use hnswlib for ANN", body="# Context\n...")

if isinstance(result, Proposal):
    result.similar          # [{'concept_id', 'title', 'score', 'via'}, ...]
    kb.approve(result)       # materializes it (same node id) — or:
    kb.reject(result, note="duplicate of 0003")
```

Three modes, controlled by `auto_approve`:

- `None` (default, **conditional**): searches the bundle for concepts similar
  to the proposal. With a vector index it auto-approves unless a hit scores
  at or above `similarity_threshold` (cosine similarity, default `0.85`);
  without one (text-only), there's no comparable numeric scale, so *any* FTS
  hit at all triggers review. Nothing to compare against (no
  title/description/body, or no hits) auto-approves.
- `True`: always writes immediately, like `add_concept` — returns a `Concept`.
- `False`: always stages it, regardless of similarity — returns a `Proposal`.

A staged proposal is a real graph node (`pending_reviews()` survives process
restarts) but is invisible to `concept()`, `concepts()`, `search()`,
iteration, and `save()` until `approve()`d — it can't leak into retrieval or
round-trip to markdown while undecided. It carries no links/citations of its
own; wire those up with `link()`/`cite()` after approval. An id collision
with an existing concept always raises — that's a hard conflict, not a
similarity judgment call, so it isn't staged for review either.

```python
kb.pending_reviews()                    # [Proposal, ...], ordered by id
kb.approve("decisions/0004-use-hnswlib")  # or an id string
kb.reject("decisions/0004-use-hnswlib", note="duplicate of 0003")
```

### Changelog: `log_entry()` and autolog

An agent that writes memory should also leave a history. `log_entry()` appends
a changelog entry (a `LogEntry` node, SPEC §7) that `save()` serializes to the
scope's `log.md` — and `autolog=True` at load time does it automatically for
every `add_concept` / `update_concept` / `remove_concept`:

```python
kb = OKFBundle.load("bundle", import_log=True, autolog=True)

kb.add_concept("notes/idea", type="Note", title="An idea", body="...")
kb.update_concept("notes/idea", description="Refined.")
kb.log_entry("Consolidated duplicate notes.", kind="Update",
             concepts=["notes/idea"])          # manual entry, MENTIONS the concept

kb.log()                    # entries newest first, including the imported history
kb.save()                   # regenerates log.md per scope (git-diffable history)
```

Autolog entries embed a markdown link to the concept
(`**Creation**: Created [An idea](/notes/idea.md).`), so `MENTIONS` edges
survive markdown round-trips. Load with `import_log=True` when the bundle
already has a `log.md` so new entries extend the history instead of replacing
it on `save()`.

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

If the bundle's markdown keeps changing across sessions, `load()` forwards
unknown keyword arguments to `import_okf_bundle`, so `incremental=True` (see
[Incremental import](#incremental-import)) works the same way: `OKFBundle.load("bundle",
db=GrafitoDatabase("kb.db"), embed=embedder, incremental=True)` re-parses and
re-embeds only what changed since the last `load()`.

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

## Agentic GraphRAG: `grafito.okf.agent`

Where `context()` is *one-shot* GraphRAG, `grafito.okf.agent` lets the **model
drive the exploration itself** through OpenAI-style tool calls:

- **`BundleTools`** — the bundle façade as function tools: `browse`
  (progressive disclosure), `search` (hybrid), `open` (full concept + typed
  edges), `follow` (graph traversal by relationship type), `history`
  (changelog), and `remember` (write a linked, embedded, autologged note back
  into the bundle). Schemas + dispatch, framework-free: the same tools drop
  into LangGraph, CrewAI, or an MCP server unchanged. Tool errors come back as
  `{"error": ...}` for the model to react to instead of killing the loop.
- **`run_agent(kb, question, chat=...)`** — a minimal tool-calling loop.
- **`Chat`** — the model contract: *any callable*
  `(messages, tools) -> assistant message` in OpenAI chat format. Grafito
  never imports an LLM SDK — the client is injected, like `rerank=`.
- **`OpenAIChat`** — the bundled convenience for any OpenAI-compatible
  endpoint (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, ...); needs `httpx`,
  reads `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`.
- **`AnthropicChat`** — Claude via the official `anthropic` SDK
  (`pip install grafito[anthropic]`). Translates the loop's OpenAI format to
  the Anthropic Messages API (system prompt, `input_schema` tools,
  `tool_use`/`tool_result` blocks), runs with adaptive thinking and preserves
  thinking blocks across turns. Defaults to `claude-opus-4-8`; credentials
  resolve from `ANTHROPIC_API_KEY` (or an `ant auth login` profile), and
  `ANTHROPIC_MODEL` overrides the model.

```python
from grafito.okf import OKFBundle, OpenAIChat, run_agent

kb = OKFBundle.load("bundle", embed=embedder, autolog=True)
answer = run_agent(kb, "why did we pick SQLite?", chat=OpenAIChat())
kb.save()   # the agent's remember()ed notes + changelog land in git
```

For a non-OpenAI-format provider, the adapter is a few lines — e.g. litellm:

```python
import litellm

def chat(messages, tools):
    response = litellm.completion(model="anthropic/claude-sonnet-5",
                                  messages=messages, tools=tools)
    return response.choices[0].message.model_dump()

run_agent(kb, question, chat=chat)
```

`examples/okf/okf_agent.py` is the runnable end-to-end walkthrough (explore →
answer with concept citations → remember a note → save bundle + `log.md`):

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1   # e.g. Ollama
export OPENAI_MODEL=llama3.1
python examples/okf/okf_agent.py
```
