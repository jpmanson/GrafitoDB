# Document → Graph Chunking

Grafito can index long documents as **many passage nodes** (still **one vector per node**), with graph links for navigation after semantic search.

Full design: `todo/DOCUMENT_CHUNKING_HELPER.md`.

## Quick start

```python
from grafito import GrafitoDatabase
from grafito.document import DocumentIngestor, MarkdownChunker
from grafito.embedding_functions import SentenceTransformerEmbeddingFunction

db = GrafitoDatabase(":memory:")
embedder = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
db.create_vector_index("docs_chunks", embedding_function=embedder)

ing = DocumentIngestor(
    db,
    chunker=MarkdownChunker(max_chars=1200, overlap=150),
    embed_index="docs_chunks",
)

result = ing.ingest(
    open("runbook.md").read(),
    document_key="runbooks/slow-queries",
    title="Slow queries",
    embed=True,
)

hits = ing.search("connection pool timeouts", k=5)
ctx = ing.expand(hits[0].node, window=1, include_ancestors=True)
packed = ing.pack(ctx, max_chars=4000)
print(packed.text)

# Section tree (titles / keys — no full bodies)
for sec in ing.toc("runbooks/slow-queries"):
    print(sec.title, sec.node_key, [c.title for c in sec.children])
```

## Behaviour (MVP)

| Topic | Behaviour |
|-------|-----------|
| Chunkers | `FixedChunker` (chars/tokens), `MarkdownChunker` (headings + overflow; **preambles** are passages) |
| Hierarchy | `hierarchy="auto"` (default): with `MarkdownChunker`, writes `:Section` tree (`HAS_SECTION`) + passages under sections |
| Ownership | Managed nodes: `managed_by=grafito.document`, `generation`, `owner_document_id` |
| Replace | Generational: new `BUILDING` → `ACTIVE`; previous generation GC’d |
| External parent | `parent_id=` attach without owning/deleting the parent (e.g. OKF Concept) |
| Search | Filters managed + active generation + `embed_role=passage`; `diversify_by_document=` |
| Expand | `global_seq` window; optional `include_ancestors=True` (section path) |
| ToC | `toc(document_key)` / `load_sections(document_key, node_keys=[…])` |
| Pack | Budget (`max_chars`, or `max_tokens` with optional `token_counter`), overlap merge via `char_start`/`char_end` |

### Pack budgets

- `max_chars=…` — hard character budget.
- `max_tokens=…, token_counter=fn` — exact token budget via your counter.
- `max_tokens=…` **without** counter — **rough estimate** `tokens ≈ ceil(len/4)` (same idea as design §7.2). Prefer a real counter for production LLM windows.

### Overlap dedup and `store_full_text`

Pack merges overlapping passages using offsets. **Best quality** when the parent keeps the full document body (`store_full_text=True`, default): merge slices `parent.text[char_start:char_end]`.

With `store_full_text=False`, merge still runs by **stitching passage texts** from offsets. That works if `char_start`/`char_end` stay consistent with each passage’s `text`; it is weaker if offsets drift. If you use chunk **overlap > 0**, keep `store_full_text=True` unless you accept stitch-based reconstruction.

## Core prerequisites used

- `GrafitoDatabase.upsert_embeddings_batch` / `remove_embeddings_batch` (single ANN mutate + single persist)
- `delete_node` best-effort removes embeddings from registered indexes

### Hierarchy notes

- **Section** = structure (`title`, `level`, `node_key`, optional `summary`); not the full body.
- **Passages** remain the only default embed targets; set `embed_section_summaries=True` to also embed sections that have a `summary` property.
- `hierarchy=False` forces flat passages only (even with `MarkdownChunker`).
- `hierarchy=True` forces the markdown tree builder (best with markdown-shaped text).
- Headings inside fenced code (`` ``` `` / ``~~~``) are **not** treated as sections (shell `#` comments stay in the passage body).

## Non-goals (yet)

Dual multi-view indexing, hybrid RRF helper, Chonkie adapter, tree/agent retrieve — see design Phase 3.
