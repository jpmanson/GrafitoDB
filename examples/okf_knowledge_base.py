"""A *narrative* (non-tabular) Open Knowledge Format bundle in GrafitoDB.

Where OKF shines is curated, cross-linked, evolving knowledge — not tabular
data. This example imports a small engineering knowledge base (architecture
decision records, an on-call runbook, and glossary terms) and shows the two
things plain markdown can't do on its own:

1. **Relationship traversal** with Cypher over the `LINKS_TO` / `CITES` graph
   built from the markdown links.
2. **Semantic search** over the prose — retrieval by meaning, not keywords.

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


def main() -> None:
    db = GrafitoDatabase(":memory:")
    summary = db.import_okf_bundle(str(BUNDLE), embed=make_embedder())
    print(f"\nImported: {summary}\n")

    # 1. Relationship traversal: which concepts does the vector-search ADR rely on?
    print("Vector-search ADR links to:")
    rows = db.execute(
        """
        MATCH (a {title: 'Add optional vector search'})-[:LINKS_TO]->(b)
        RETURN DISTINCT b.title AS title, b.type AS type
        ORDER BY title
        """
    )
    for row in rows:
        print(f"  -> {row['title']}")

    # 2. Two-hop traversal: terms reachable from a runbook through its links.
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

    # 3. Citations: external sources cited across the bundle, counted with an
    #    implicit GROUP BY (grouped by the non-aggregated return item).
    print("\nMost-cited external sources:")
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

    # 4. Semantic search over prose: a paraphrased question, no keyword overlap.
    print("\nSemantic search — 'how do I make a query run faster?'")
    for hit in db.semantic_search("how do I make a query run faster", index="okf", k=3):
        node = hit["entity"] if "entity" in hit else hit["node"]
        print(f"  {node.properties.get('title')}  (score={hit['score']:.3f})")

    db.close()


if __name__ == "__main__":
    main()
