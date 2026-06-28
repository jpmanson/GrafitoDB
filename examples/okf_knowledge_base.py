"""A *narrative* (non-tabular) OKF bundle through the high-level OKFBundle API.

Where OKF shines is curated, cross-linked, evolving knowledge — not tabular
data. This loads a small engineering knowledge base (architecture decision
records, an on-call runbook, glossary terms) as an ``OKFBundle`` and shows the
façade end to end:

1. **Import + index** — concepts, and the in-memory ``index`` for triage.
2. **Traversal** — ``concept.links()``, multi-hop, and the directory tree.
3. **Aggregation** — most-cited sources, via the ``kb.execute`` escape hatch.
4. **Semantic search** — retrieval by meaning, results as a uniform ``Hit``.
5. **Exploiting a hit** — a ``Hit`` carries a full ``Concept`` you can pivot from.
6. **Grounded context for an agent** — ``kb.context`` packs a token-budgeted,
   graph-expanded, cited prompt fragment (framework-agnostic GraphRAG).
7. **Agent-memory write path** — ``add_concept`` / ``link`` / ``cite`` / ``save``.
8. **Visualization** — via ``kb.db`` (the full graph is always reachable).

Run:  python examples/okf_knowledge_base.py
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

from grafito.embedding_functions import EmbeddingFunction
from grafito.okf import OKFBundle

BUNDLE = Path(__file__).parent / "okf_knowledge_base"


class HashingEmbeddingFunction(EmbeddingFunction):
    """Dependency-free fallback embedder (hashed bag-of-words).

    Good enough to demonstrate semantic retrieval offline. In production, swap in
    a real model, e.g. ``SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")``.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []
        for text in input:
            vec = [0.0] * self._dim
            for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
                # md5, not Python's hash(): the latter is randomized per process.
                bucket = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "little")
                vec[bucket % self._dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "hashing_demo"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "ip"]

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(dim=config.get("dim", 256))

    def get_config(self) -> dict:
        return {"dim": self._dim}

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    @property
    def dimension(self) -> int:
        return self._dim


def make_embedder() -> EmbeddingFunction:
    """Use a real model if available, otherwise the offline fallback."""
    try:
        from grafito.embedding_functions import SentenceTransformerEmbeddingFunction

        embedder = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        print("Embedder: SentenceTransformer (all-MiniLM-L6-v2)")
        return embedder
    except Exception:
        print("Embedder: hashing fallback (install sentence-transformers for real semantics)")
        return HashingEmbeddingFunction()


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n {text}\n{'=' * 70}")


def main() -> None:
    # ── Import ───────────────────────────────────────────────────────────────
    # OKFBundle.load delegates to the importer and returns a façade. We embed
    # for semantic search and materialize the directory tree (CONTAINS edges).
    kb = OKFBundle.load(BUNDLE, embed=make_embedder(), directory_nodes=True)

    banner("1. What the import produced")
    s = kb.summary
    print(
        f"  concepts={len(kb)}  relationships={s['relationships']}  "
        f"citations={s['citations']}  references={s['references']}  "
        f"directories={s['directories']}"
    )
    # index() is the in-memory equivalent of index.md: titles + descriptions,
    # no bodies — exactly what you'd skim to decide what to open.
    print("\n  Bundle index (group → concepts), no bodies loaded:")
    for layer in kb.index()["subdirs"]:
        for entry in kb.index(layer)["concepts"]:
            print(f"    [{entry['type']:<8}] {entry['title']}")

    # ── Traversal ────────────────────────────────────────────────────────────
    banner("2. Traversal — following the links plain markdown can only render")

    # 2a. Direct neighbours, in OKF vocabulary.
    vec = kb.concept("decisions/0003-vector-search")
    print(f"'{vec.title}' links directly to:")
    for c in vec.links():
        print(f"  -> {c.title}: {c.description}")

    # 2b. Multi-hop *with hop distance* — the façade doesn't hide the graph, so
    #     we drop to Cypher via kb.execute for what needs it.
    print("\nWithin 2 hops of the slow-query runbook (with distance):")
    rows = kb.execute(
        """
        MATCH p=(r {title: 'Triaging a slow graph query'})-[:LINKS_TO*1..2]->(c)
        RETURN c.title AS title, min(length(p)) AS hops
        ORDER BY hops, title
        """
    )
    for row in rows:
        print(f"  {row['hops']} hop{'s' if row['hops'] > 1 else ' '}  ~ {row['title']}")

    # 2c. The directory tree, traversed as a graph (CONTAINS edges).
    print("\nDirectory tree (via CONTAINS):")
    for layer in kb.children()["subdirs"]:
        kids = [c.id.split("/")[-1] for c in kb.children(layer)["concepts"]]
        print(f"  {layer}/  →  {kids}")

    # ── Aggregation ──────────────────────────────────────────────────────────
    banner("3. Aggregation — which external sources does this KB lean on?")
    rows = kb.execute(
        """
        MATCH (a)-[:CITES]->(r:Reference)
        RETURN r.url AS url, COUNT(*) AS citations
        ORDER BY citations DESC LIMIT 3
        """
    )
    for row in rows:
        print(f"  [{row['citations']}x] {row['url']}")

    # ── Semantic search ──────────────────────────────────────────────────────
    banner("4. Semantic search — retrieval by meaning, not keywords")
    question = "how do I make a query run faster"
    print(f"Question (no keyword overlap with the docs): {question!r}\n")
    hits = kb.search(question, k=3)  # auto → semantic (we loaded with embeddings)
    for hit in hits:
        print(f"  {hit.concept.title:<42} [{hit.concept.type}]  score={hit.score:.3f}  via={hit.via}")

    # ── Exploiting a hit ─────────────────────────────────────────────────────
    banner("5. Exploiting the top hit — a result is a full concept, not a snippet")
    top = hits[0].concept
    print(f"Top hit: {top.title!r}  (type {top.type}, id {top.id})")
    print(f"  description : {top.description}")
    print(f"  tags        : {top.tags}")
    print("  pivot — what this concept relies on:")
    for c in top.links():
        print(f"    -> {c.title}")

    # ── Grounded context for an agent ────────────────────────────────────────
    # kb.context is the bridge to *any* agent loop: it seeds with search, follows
    # the graph (so it pulls in linked context the embedding alone would miss),
    # and packs the result into a token budget — returning prompt-ready text plus
    # the citations that back it. No framework: drop `str(pack)` into your prompt.
    #
    # rerank is an optional, injectable precision step: the seed + graph-expanded
    # pool is re-scored against the query text before budgeting. Here we use the
    # dependency-free LexicalReranker; in production inject a cross-encoder
    # (e.g. CohereReranker) — same one-line hook, no framework lock-in.
    from grafito.okf import LexicalReranker

    banner("6. Grounded context — retrieve + graph-expand + rerank + pack to a budget")
    pack = kb.context(question, k=3, budget_tokens=600, rerank=LexicalReranker())
    print(
        f"  packed {len(pack.concepts)} concepts (~{pack.tokens} tokens, "
        f"truncated={pack.truncated}) from {len(pack.hits)} seed hits"
    )
    print(f"  concepts: {[c.id for c in pack.concepts]}")
    print(f"  citations backing the answer ({len(pack.citations)}):")
    for cit in pack.citations[:3]:
        print(f"    - {cit.get('url') or cit.get('concept')}  (cited_by {cit['cited_by']})")
    print("\n  --- prompt-ready text (first 280 chars) ---")
    print("  " + str(pack)[:280].replace("\n", "\n  ") + " …")

    # ── Agent-memory write path ──────────────────────────────────────────────
    # The same façade writes: add a concept, relate it, cite a source, persist.
    banner("7. Agent memory — add knowledge and save it back to markdown")
    note = kb.add_concept(
        "decisions/0004-result-cache",
        type="ADR",
        title="Cache hot query results",
        description="Memoize expensive read queries.",
        tags=["perf"],
        body="# Context\nHot read queries recur and dominate latency.\n",
    )
    kb.link(note, "decisions/0001-use-sqlite", anchor="builds on")
    kb.cite(note, "https://www.sqlite.org/pragma.html", anchor="SQLite PRAGMA")
    print(f"  added {note.id!r}; it now links to {[c.id for c in note.links()]}")
    out = Path(tempfile.mkdtemp(prefix="okf_kb_"))
    print(f"  saved {kb.save(out)} to {out}")

    # ── Visualization ────────────────────────────────────────────────────────
    banner("8. Visualizing the graph (via kb.db)")
    graph = kb.db.to_networkx()
    print(f"  NetworkX export: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    try:
        from grafito.integrations import save_pyvis_html

        html_path = save_pyvis_html(
            graph, path=str(out / "okf_knowledge_graph.html"), label_attr="title",
            color_by_label=True, physics="spread",
        )
        print(f"  Interactive  : {Path(html_path).resolve()}  (open in a browser)")
    except ImportError:
        print("  (visualization backends not installed — `pip install grafito[viz]`)")

    kb.db.close()


if __name__ == "__main__":
    main()
