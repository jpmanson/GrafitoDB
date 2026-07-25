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
ctx = ing.expand(hits[0].node, window=1)
packed = ing.pack(ctx, max_chars=4000)
print(packed.text)
```

## Behaviour (MVP)

| Topic | Behaviour |
|-------|-----------|
| Chunkers | `FixedChunker` (chars/tokens), `MarkdownChunker` (headings + overflow; **preambles** are passages) |
| Ownership | Managed nodes: `managed_by=grafito.document`, `generation`, `owner_document_id` |
| Replace | Generational: new `BUILDING` → `ACTIVE`; previous generation GC’d |
| External parent | `parent_id=` attach without owning/deleting the parent (e.g. OKF Concept) |
| Search | Filters managed + active generation + `embed_role=passage` |
| Expand | `global_seq` window (not hop-by-hop NEXT) |
| Pack | Budget (`max_chars` / `max_tokens`+counter), overlap merge via `char_start`/`char_end` |

## Core prerequisites used

- `GrafitoDatabase.upsert_embeddings_batch` / `remove_embeddings_batch` (single ANN mutate + single persist)
- `delete_node` best-effort removes embeddings from registered indexes

## Non-goals (yet)

Section tree ToC, dual multi-view indexing, hybrid RRF helper, Chonkie adapter — see design Phase 2–3.
