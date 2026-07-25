#!/usr/bin/env python3
"""Document → graph chunking demo (DocumentIngestor).

Ingests a long markdown runbook as section + passage nodes, embeds passages,
then shows vector search, expand/pack, ToC, and hybrid search (if FTS5 is available).

Runs with a tiny built-in embedder (no FAISS / sentence-transformers required).
For production, swap in SentenceTransformerEmbeddingFunction or OpenAI, etc.

Usage (from repo root)::

    python examples/semantic/document_chunking.py
"""

from __future__ import annotations

import re

from grafito import GrafitoDatabase
from grafito.document import DocumentIngestor, MarkdownChunker, TitleContextEnricher
from grafito.embedding_functions.base import EmbeddingFunction

# Sample runbook with a fenced shell block (headings inside fences must NOT become sections).
RUNBOOK = """# Triaging slow graph queries

When query latency spikes, work through these steps before scaling hardware.

## Check connection pool

Connection pool exhaustion often looks like random timeouts under load.

Symptoms:
- intermittent `timeout waiting for connection`
- rising queue depth while CPU is idle

```bash
# apt update   <-- this is NOT a section heading
# check pool size
curl -s localhost:8080/metrics | grep pool_
```

## Inspect Cypher plans

Look for cartesian products and unbounded variable-length paths.

```cypher
// Prefer bounded patterns
MATCH (a:Person)-[:KNOWS*1..3]->(b)
RETURN a, b
LIMIT 100
```

## Escalate

If pool and plans look fine, capture a profile and open an issue with
`document_key`, sample Cypher, and wall-clock timings.
"""


class ToyEmbedder(EmbeddingFunction):
    """Deterministic bag-of-tokens embedder for offline demos."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in input:
            vec = [0.0] * self._dim
            for i, tok in enumerate(re.findall(r"[a-z0-9áéíóúñ_]+", text.lower())):
                vec[hash(tok) % self._dim] += 1.0 + (i % 3) * 0.01
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out

    @staticmethod
    def name() -> str:
        return "toy_document_chunking_example"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    @staticmethod
    def build_from_config(config: dict) -> ToyEmbedder:
        return ToyEmbedder(dim=int(config.get("dim", 32)))

    def get_config(self) -> dict:
        return {"dim": self._dim}

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    @property
    def dimension(self) -> int:
        return self._dim


def main() -> None:
    print("=== Document chunking example ===\n")
    db = GrafitoDatabase(":memory:")
    embedder = ToyEmbedder()
    db.create_vector_index(
        "docs_chunks",
        dim=embedder.dimension,
        backend="bruteforce",
        embedding_function=embedder,
        options={"metric": "cosine"},
    )

    ing = DocumentIngestor(
        db,
        chunker=MarkdownChunker(max_chars=400, overlap=40),
        embed_index="docs_chunks",
        configure_fts=db.has_fts5(),
        enricher=TitleContextEnricher(),
    )

    result = ing.ingest(
        RUNBOOK,
        document_key="runbooks/slow-queries",
        title="Triaging slow graph queries",
        source="examples/semantic/document_chunking.py",
        embed=True,
    )
    print(
        f"Ingested: hierarchy={result.hierarchy} "
        f"sections={result.n_sections} passages={result.n_passages} "
        f"generation={result.generation}"
    )

    print("\n--- Table of contents ---")
    for sec in ing.toc("runbooks/slow-queries"):
        print(f"  [{sec.node_key}] L{sec.level} {sec.title}")
        for child in sec.children:
            print(f"      [{child.node_key}] L{child.level} {child.title}")

    query = "connection pool timeout under load"
    print(f"\n--- Vector search: {query!r} ---")
    hits = ing.search(query, k=3)
    for i, h in enumerate(hits, 1):
        text = (h.node.properties.get("text") or "")[:80].replace("\n", " ")
        print(f"  {i}. score={h.score:.3f} seq={h.global_seq}  {text}…")

    if hits:
        print("\n--- Expand (window=1) + pack ---")
        ctx = ing.expand(hits[0].node, window=1, include_ancestors=True)
        if ctx.section:
            print(f"  section={ctx.section.properties.get('title')!r}")
        if ctx.ancestors:
            print(f"  ancestors={[a.properties.get('title') for a in ctx.ancestors]}")
        packed = ing.pack(ctx, max_chars=1200, include_citations=True)
        print(f"  passages_in_window={len(ctx.passages)} truncated={packed.truncated}")
        print("  --- packed context (first 500 chars) ---")
        print(packed.text[:500] + ("…" if len(packed.text) > 500 else ""))

    if db.has_fts5():
        print("\n--- Hybrid search (RRF) ---")
        for i, h in enumerate(ing.hybrid_search("pool timeout", k=3), 1):
            text = (h.node.properties.get("text") or "")[:70].replace("\n", " ")
            print(f"  {i}. rrf={h.score:.4f}  {text}…")
    else:
        print("\n(FTS5 unavailable — skipping hybrid_search demo)")

    print("\n--- Cypher string vector query ---")
    rows = db.execute(
        """
        CALL db.vector.search('docs_chunks', $q, 2)
        YIELD node, score
        RETURN node.text AS text, score
        """,
        {"q": "cypher plan cartesian"},
    )
    for row in rows:
        text = str(row.get("text") or "")[:70].replace("\n", " ")
        print(f"  score={row.get('score'):.3f}  {text}…")

    # Fence sanity: shell comments must not appear as section titles
    titles: list[str] = []

    def walk(secs) -> None:
        for s in secs:
            titles.append(s.title)
            walk(s.children)

    walk(ing.toc("runbooks/slow-queries"))
    assert "apt update" not in " ".join(titles), "fenced # comments became sections"
    print("\nFence check: OK (no fake sections from bash comments)")
    print("\nDone.")
    db.close()


if __name__ == "__main__":
    main()
