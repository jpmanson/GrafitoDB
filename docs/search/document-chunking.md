# Document → Graph Chunking

Grafito can index long documents as **many passage nodes** (still **one vector per node**), with graph links for navigation after semantic search.

Related: [Vector Search](vector.md) · [Hybrid Search](hybrid.md) · [Embeddings](../embeddings/overview.md)

## Runnable examples

**Script (offline, no optional ML deps):**

```bash
python examples/semantic/document_chunking.py
```

**Notebook for Colab / class (PDF → chunk → search + PyVis graphs):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jpmanson/GrafitoDB/blob/main/examples/semantic/pdf_chunking_colab.ipynb)

```text
examples/semantic/pdf_chunking_colab.ipynb
```

The notebook walks students through extraction, hierarchy, embeddings, search,
expand/pack, and hybrid RRF while rendering the graph after each major step.

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

## Behaviour

| Topic | Behaviour |
|-------|-----------|
| Chunkers | `FixedChunker` (chars/tokens + word boundary), `RecursiveChunker` (hierarchical separators, LC-style), `MarkdownChunker` (headings + overflow; **preambles** are passages), `SemanticBreakpointChunker`, `ChonkieChunker` |
| Hierarchy | `hierarchy="auto"` (default): with `MarkdownChunker`, writes `:Section` tree (`HAS_SECTION`) + passages under sections |
| Ownership | Managed nodes: `managed_by=grafito.document`, `generation`, `owner_document_id` |
| Replace | Generational: new `BUILDING` → `ACTIVE`; previous generation GC’d |
| External parent | `parent_id=` attach without owning/deleting the parent (e.g. OKF Concept) |
| Search | Filters managed + active generation + `embed_role=passage`; `diversify_by_document=` |
| Expand | `global_seq` window; optional `include_ancestors=True` (section path) |
| Reading chain | By default each passage links to the next: `Chunk_i -[:NEXT_PASSAGE]-> Chunk_{i+1}` (`write_next_passage=True`). Disable with `write_next_passage=False`. `expand` still uses `global_seq`, not these edges. |
| ToC | `toc(document_key)` / `load_sections(document_key, node_keys=[…])` |
| Pack | Budget (`max_chars`, or `max_tokens` with optional `token_counter`), overlap merge via `char_start`/`char_end` |

### Pack budgets

- `max_chars=…` — hard character budget.
- `max_tokens=…, token_counter=fn` — exact token budget via your counter.
- `max_tokens=…` **without** counter — **rough estimate** `tokens ≈ ceil(len/4)`. Prefer a real counter for production LLM windows.

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

## Advanced retrieval

### Hybrid search (RRF)

```python
ing = DocumentIngestor(..., embed_index="docs_chunks", configure_fts=True)
hits = ing.hybrid_search(
    "connection pool timeout",
    k=5,
    vector_k=20,
    fts_k=20,
    rrf_k=60,
    vector_weight=1.0,
    fts_weight=1.0,
)
```

FTS must index passage `text` (`configure_fts=True` or manual `create_text_index`). Scores from vector and BM25 are **not** summed raw — fused with Reciprocal Rank Fusion.

### Tree select (agentic ToC path)

```python
def my_llm(prompt: str) -> str:
    # return JSON list of node_key strings
    return '["0001", "0003"]'

keys = ing.tree_select("runbooks/x", "how do we rotate secrets?", my_llm)
sections = ing.load_sections("runbooks/x", keys)
```

### Choosing a chunker

| Chunker | Best for | Notes |
|---------|----------|--------|
| **`RecursiveChunker`** | Plain prose / mixed text without reliable headings | Default-of-choice for unstructured bodies (LangChain `RecursiveCharacterTextSplitter` style: `\n\n` → `\n` → ` ` → hard cut). Exact `char_start`/`char_end`. |
| **`MarkdownChunker`** | Markdown / docs with ATX headings | Structure-aware + section tree; large sections overflow via `FixedChunker`. |
| **`FixedChunker`** | Uniform windows, token budgets | Sliding window; `boundary="word"` avoids mid-word cuts. Use `unit="tokens"` + `counter=` when needed. |
| **`SemanticBreakpointChunker`** | Topic shifts in long prose | Needs an embedder; more expensive. |
| **`ChonkieChunker`** | External recipes (token-aware, late chunking, …) | Thin adapter around a Chonkie chunker instance or `from_recipe`. |

```python
from grafito.document import RecursiveChunker

# Plain text / OKF body without markdown structure
ing = DocumentIngestor(
    db,
    chunker=RecursiveChunker(max_size=1200, overlap=150),
    hierarchy=False,
    embed_index="docs_chunks",
)

# Language-aware separators (python, markdown, html, javascript, …)
code_chunker = RecursiveChunker.from_language("python", max_size=800, overlap=50)
```

### Semantic breakpoint chunker

```python
from grafito.document import SemanticBreakpointChunker

ing = DocumentIngestor(
    db,
    chunker=SemanticBreakpointChunker(embedder, threshold=0.3, min_chars=200, max_chars=1200),
    hierarchy=False,  # flat passages
    embed_index="docs_chunks",
)
```

### Cypher query strings

With an embedding function on the index:

```cypher
CALL db.vector.search('docs_chunks', 'connection pool', 5)
YIELD node, score
RETURN node.text, score
```

### Enrichment

```python
from grafito.document import TitleContextEnricher
ing = DocumentIngestor(..., enricher=TitleContextEnricher())
```

Sets `context` on passages so embeddings use title/section situating text.

### Dual multi-view indexing

Index one document under more than one segmentation and fuse them at query time.
Each view's passages carry a `view` property and their own `global_seq` reading
order; `diversify_by_span=True` keeps only the best-scored hit among overlapping
character spans, so a passage indexed under two views does not appear twice.

```python
ing = DocumentIngestor(db, chunker=MarkdownChunker(), embed_index="docs_chunks")

# Opt-in: index the same document as a hierarchy AND fixed windows.
ing.ingest(text, document_key="doc", views=["hierarchy", "fixed"])

hits = ing.search("connection pool", k=5, diversify_by_span=True)  # fused, no dup spans
only_fixed = ing.search("connection pool", k=5, views=["fixed"])   # restrict to one view
```

Default is a single view derived from the chunker — dual views double embedding
cost, so reach for them only when one granularity is not enough. `replace()`
rebuilds every view (generational GC), and the fingerprint includes the view set.

### Chonkie adapter

Reuse [Chonkie](https://github.com/chonkie-ai/chonkie)'s token-aware chunkers
(extra: `pip install 'grafito[document-chonkie]'`).

```python
from grafito.document import ChonkieChunker

# Build from a short recipe name (imports chonkie lazily)...
chunker = ChonkieChunker.from_recipe("recursive", chunk_size=512)
# ...or wrap an existing Chonkie chunker instance.
from chonkie import TokenChunker
chunker = ChonkieChunker(TokenChunker(chunk_size=512))

ing = DocumentIngestor(db, chunker=chunker, embed_index="docs_chunks", hierarchy=False)
```

Chonkie offsets and `token_count` map onto `ChunkSpec`.

## OKF long-body chunking (opt-in)

For OKF bundles, `enable_body_chunking()` indexes any concept whose `body`
reaches a threshold as passage nodes on a **separate** vector index, while short
concepts keep their single title+description+body vector. The concept node is
never mutated with helper bookkeeping — each chunked concept gets a dedicated
managed `Document` (flagged `okf_auto`, so it never round-trips as a concept)
linked via `HAS_PASSAGES`.

```python
from grafito.okf import OKFBundle

kb = OKFBundle.load("kb/", embed=my_embedder)
kb.enable_body_chunking(threshold=2000)  # chars; separate index "okf_chunks"

kb.add_concept("runbooks/slow", type="Runbook", body=long_markdown)  # auto-chunked
kb.add_concept("glossary/term", type="Term", body="Short definition.")  # single node

hits = kb.search_passages("connection pool timeouts", k=5)
hits = kb.search_passages("...", k=5, concept_id="runbooks/slow")  # scope to one concept
```

A chunked long concept is embedded twice (concept vector on the main index,
passages on the chunk index) — the intended cost of passage-level retrieval.
