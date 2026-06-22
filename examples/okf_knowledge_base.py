"""A *narrative* (non-tabular) Open Knowledge Format bundle in GrafitoDB.

Where OKF shines is curated, cross-linked, evolving knowledge — not tabular
data. This example imports a small engineering knowledge base (architecture
decision records, an on-call runbook, and glossary terms) and walks through
what you actually get back, and what you can do with it:

1. **What the import produced** — every `.md` file becomes a graph *node*; its
   YAML frontmatter becomes node *properties*, the prose becomes a `body`
   property, and every `[link](...)` in the text becomes a typed relationship
   (`LINKS_TO` between docs, `CITES` to external `Reference` nodes).
2. **Relationship traversal** with Cypher over that `LINKS_TO` / `CITES` graph.
3. **Aggregation** over the graph (most-cited external sources).
4. **Semantic search** over the prose — retrieval by meaning, not keywords.
5. **Exploiting a hit** — a search result is a *full node*, so you can read its
   metadata and prose, then pivot straight back into the graph from it.

Run:  python examples/okf_knowledge_base.py
"""

from __future__ import annotations

import re
from pathlib import Path

from grafito import GrafitoDatabase
from grafito.embedding_functions import EmbeddingFunction

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
                vec[hash(token) % self._dim] += 1.0
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
    db = GrafitoDatabase(":memory:")

    # ── Import ───────────────────────────────────────────────────────────────
    # Each .md file under examples/okf_knowledge_base/ is read once and turned
    # into a graph node. The summary tells you exactly what landed in the graph.
    summary = db.import_okf_bundle(str(BUNDLE), embed=make_embedder())
    banner("1. What the import produced")
    print(
        f"  nodes={summary['nodes']}  relationships={summary['relationships']}  "
        f"citations={summary['citations']}  references={summary['references']}  "
        f"embedded={summary['embedded']}"
    )
    print(
        "\n  Reading that back: the 7 markdown documents became 7 graph nodes;\n"
        "  their inline [links] became 21 typed relationships; 11 of those are\n"
        "  CITES edges pointing at 9 distinct external Reference nodes; and all\n"
        "  7 documents were embedded so they can be searched by meaning."
    )

    # Every node carries the doc's frontmatter as properties, PLUS the full
    # markdown prose under `body`. Let's look at the inventory grouped by label
    # (the OKF `type:` field — ADR / Term / Playbook — becomes the node label).
    print("\n  Documents in the bundle (label · title · key metadata):")
    for label in ("ADR", "Term", "Playbook"):
        rows = db.execute(
            f"MATCH (n:{label}) "
            f"RETURN n.title AS title, n.status AS status, n.tags AS tags, "
            f"n.timestamp AS ts ORDER BY title"
        )
        for r in rows:
            meta = f"tags={r['tags']}"
            if r["status"]:
                meta = f"status={r['status']}  " + meta
            print(f"    [{label:<8}] {r['title']:<42} {meta}")

    # ── Traversal ────────────────────────────────────────────────────────────
    # The links plain markdown can only render, Cypher can now *query*.
    banner("2. Relationship traversal — following the links")

    # 2a. Direct neighbours: which concepts does the vector-search ADR rely on?
    print("The 'Add optional vector search' ADR links directly to:")
    rows = db.execute(
        """
        MATCH (a {title: 'Add optional vector search'})-[:LINKS_TO]->(b)
        RETURN DISTINCT b.title AS title, b.description AS description
        ORDER BY title
        """
    )
    for row in rows:
        print(f"  -> {row['title']}: {row['description']}")
    print(
        "  (i.e. to understand this decision you'd read those documents next — the\n"
        "   graph encodes that dependency explicitly, instead of you grepping links.)"
    )

    # 2b. Multi-hop: everything reachable from the runbook within 2 hops.
    print("\nConcepts within 2 hops of the slow-query runbook:")
    rows = db.execute(
        """
        MATCH (r {title: 'Triaging a slow graph query'})-[:LINKS_TO*1..2]->(c)
        RETURN DISTINCT c.title AS title
        ORDER BY title
        """
    )
    for row in rows:
        print(f"  ~ {row['title']}")
    print(
        "  (transitive context: the runbook mentions Cypher, which in turn links to\n"
        "   its decision record — variable-length traversal surfaces both at once.)"
    )

    # ── Aggregation ──────────────────────────────────────────────────────────
    banner("3. Aggregation — which external sources does this KB lean on?")
    rows = db.execute(
        """
        MATCH (a)-[:CITES]->(r:Reference)
        RETURN r.url AS url, COUNT(*) AS citations
        ORDER BY citations DESC
        LIMIT 3
        """
    )
    for row in rows:
        print(f"  [{row['citations']}x] {row['url']}")
    print(
        "  (a COUNT(*) with implicit GROUP BY over the CITES edges — the most-cited\n"
        "   URLs are the load-bearing references behind the whole bundle.)"
    )

    # ── Semantic search ──────────────────────────────────────────────────────
    banner("4. Semantic search — retrieval by meaning, not keywords")
    question = "how do I make a query run faster"
    print(f"Question (no keyword overlap with the docs): {question!r}\n")
    hits = db.semantic_search(question, index="okf", k=3)
    for hit in hits:
        node = hit["entity"] if "entity" in hit else hit["node"]
        print(
            f"  {node.properties['title']:<42} "
            f"[{', '.join(node.labels)}]  score={hit['score']:.3f}"
        )

    # ── Exploiting a hit ─────────────────────────────────────────────────────
    # The key point: a search result is not a text snippet — it's a full graph
    # node. You hold all of its properties AND its position in the graph.
    banner("5. Exploiting the top hit — a result is a full node, not a snippet")
    top = (hits[0]["entity"] if "entity" in hits[0] else hits[0]["node"])
    props = top.properties

    print(f"Top hit: {props['title']!r}  (label {top.labels}, uri {top.uri})\n")

    # 5a. You have every property the document carried.
    print("  Available properties:", ", ".join(sorted(props)))
    print(f"    description : {props['description']}")
    print(f"    tags        : {props['tags']}")
    print(f"    last edited : {props['timestamp']}")

    # 5b. Including the full prose under `body` — ready to render or feed to an LLM.
    first_lines = "\n      ".join(props["body"].strip().splitlines()[:3])
    print(f"    body (head) :\n      {first_lines}")

    # 5c. And because it's a node, you can pivot from the hit back into the graph:
    #     "search found the right doc — now show me what it depends on."
    title = props["title"].replace("'", "\\'")
    rows = db.execute(
        f"""
        MATCH (hit {{title: '{title}'}})-[:LINKS_TO]->(c)
        RETURN DISTINCT c.title AS title
        ORDER BY title
        """
    )
    print("\n  Pivoting from the hit into the graph — concepts this doc relies on:")
    for row in rows:
        print(f"    -> {row['title']}")
    print(
        "\n  That is the OKF payoff: semantic search lands you on the right node, and\n"
        "  the graph lets you walk outward from it — meaning-based recall plus\n"
        "  structured context, over the same plain markdown you authored."
    )

    db.close()


if __name__ == "__main__":
    main()
