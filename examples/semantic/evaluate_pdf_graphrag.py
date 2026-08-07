"""Evaluate PDF GraphRAG retrieval strategies used in pdf_chunking_colab.ipynb.

This is deliberately small and reproducible: it builds the same in-memory Grafito
index over Anthropic's "Building Effective AI Agents" PDF, then compares basic
retrieval against graph-expanded variants.

The goal is not to prove universal superiority. It is to detect whether
SEMANTIC_NEAR and recent GraphRAG APIs add useful context or just noise.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import liteparse
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

from grafito import GrafitoDatabase
from grafito.document import DocumentIngestor, MarkdownChunker, RecursiveChunker, TitleContextEnricher
from grafito.embedding_functions.base import EmbeddingFunction

PDF_URL = (
    "https://resources.anthropic.com/hubfs/Building%20Effective%20AI%20Agents-"
    "%20Architecture%20Patterns%20and%20Implementation%20Frameworks.pdf"
)
DOC_KEY = "anthropic/building-effective-ai-agents"
SEMANTIC_REL = "SEMANTIC_NEAR"
OUT_DIR = Path("examples/semantic/eval_outputs")
CACHE_DIR = Path(tempfile.gettempdir()) / "grafito_pdf_graphrag_eval"


@dataclass(frozen=True)
class EvalQuery:
    key: str
    question: str
    expected_sections: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    notes: str = ""


QUERIES: list[EvalQuery] = [
    EvalQuery(
        key="automation_difference",
        question="How do AI agents differ from traditional automation?",
        expected_sections=("business case for ai agents", "executive summary"),
        evidence_terms=("independently", "tools", "complex", "recover", "rigid", "prewritten"),
        notes="Should recover the definition of agents as autonomous tool-using systems, not just generic AI business value.",
    ),
    EvalQuery(
        key="single_vs_multi",
        question="When should I use a single agent instead of multi-agent orchestration?",
        expected_sections=("single-agent systems", "multi-agent systems", "decision framework"),
        evidence_terms=("single-agent", "multi-agent", "complex", "specialized", "resource", "orchestration"),
        notes="Needs both sides of the tradeoff, not only a definition of one architecture.",
    ),
    EvalQuery(
        key="agent_skills",
        question="What are Agent Skills and when should an organization use them?",
        expected_sections=("agent design best practices", "single-agent systems"),
        evidence_terms=("skills", "domain-specific", "standardized workflows", "specialized", "expertise", "integration"),
        notes="Exact term and nearby explanation should matter; hybrid should help.",
    ),
    EvalQuery(
        key="customer_support",
        question="What customer support use cases are described for AI agents?",
        expected_sections=("customer support and operations", "common use cases"),
        evidence_terms=("customer", "support", "intercom", "fin", "resolution", "operations"),
        notes="A lexical-heavy query with named entities; pure vector should not be assumed best.",
    ),
    EvalQuery(
        key="evaluator_optimizer",
        question="What is the evaluator-optimizer workflow pattern useful for?",
        expected_sections=("evaluator-optimizer", "agentic workflows"),
        evidence_terms=("evaluator", "optimizer", "feedback", "iterative", "quality", "loop"),
        notes="Tests whether retrieval can reach a specific workflow pattern, not just broad agent architecture.",
    ),
    EvalQuery(
        key="observability",
        question="Why do agent systems need observability and audit trails?",
        expected_sections=("agent design best practices",),
        evidence_terms=("observable", "explain", "audit", "trace", "decision", "black"),
        notes="Tests whether expansion adds the neighboring best-practices rationale.",
    ),
]


class STEmbedder(EmbeddingFunction):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model
        self.model = SentenceTransformer(model)
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        dim = dim_fn()
        if dim is None:
            raise RuntimeError("SentenceTransformer did not report an embedding dimension")
        self._dim = int(dim)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()

    @staticmethod
    def name() -> str:
        return "st_pdf_eval"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    @staticmethod
    def build_from_config(config: dict) -> "STEmbedder":
        return STEmbedder(config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))

    def get_config(self) -> dict:
        return {"model_name": self.model_name, "dim": self._dim}

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    @property
    def dimension(self) -> int:
        return self._dim


def clean_markdown(md_text: str) -> str:
    out: list[str] = []
    prev: str | None = None
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if re.fullmatch(r"Chapter \d+", body, re.I):
                continue
            low = body.lower().rstrip(":").strip()
            if prev and (low == prev or low.startswith(prev) or prev.startswith(low)):
                continue
            out.append(line)
            prev = low
        else:
            if stripped:
                prev = None
            out.append(line)
    return "\n".join(out)


def load_pdf_markdown(cache_dir: Path) -> str:
    pdf_path = cache_dir / "building_effective_ai_agents.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(requests.get(PDF_URL, timeout=120).content)
    parser = liteparse.LiteParse(output_format="markdown", ocr_enabled=False)
    parsed = parser.parse(str(pdf_path))
    pages = [parsed.get_page(i) for i in range(1, parsed.num_pages + 1)]
    return clean_markdown("\n\n".join(page.markdown for page in pages if page))


def build_corpus() -> tuple[GrafitoDatabase, DocumentIngestor, dict[str, Any]]:
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_text = load_pdf_markdown(cache_dir)

    db = GrafitoDatabase(":memory:")
    db.create_vector_index(
        "chunks",
        backend="bruteforce",
        embedding_function=STEmbedder(),
        options={"metric": "cosine"},
    )
    ing = DocumentIngestor(
        db,
        chunker=MarkdownChunker(
            max_chars=1100,
            overlap=120,
            overflow_chunker=RecursiveChunker(max_size=1100, overlap=120),
        ),
        embed_index="chunks",
        configure_fts=db.has_fts5(),
        enricher=TitleContextEnricher(),
        hierarchy="auto",
        write_next_passage=True,
    )
    result = ing.ingest(
        doc_text,
        document_key=DOC_KEY,
        title="Building Effective AI Agents",
        source=PDF_URL,
        embed=True,
    )
    sem_report = db.create_semantic_graph(
        index="chunks",
        rel_type=SEMANTIC_REL,
        labels=["Chunk"],
        k=3,
        min_score=0.45,
        undirected=True,
    )
    return db, ing, {"ingest": result, "semantic_graph": sem_report}


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def node_text(node) -> str:
    text = node.properties.get("text") or ""
    if text.strip():
        return text
    # Empty body chunks occur at some section boundaries; use context as weak fallback.
    return node.properties.get("context") or ""


def unique_nodes(nodes: Iterable) -> list:
    seen = set()
    out = []
    for node in nodes:
        if node is None or node.id in seen:
            continue
        seen.add(node.id)
        out.append(node)
    return out


def search_hits_to_nodes(hits: Iterable) -> list:
    return unique_nodes(hit.node if hasattr(hit, "node") else hit["node"] for hit in hits)


def section_path(ing: DocumentIngestor, node) -> str:
    ex = ing.expand(node, window=0, include_ancestors=True)
    names = [a.properties.get("title") or "" for a in ex.ancestors]
    if ex.section:
        names.append(ex.section.properties.get("title") or "")
    return " / ".join(x for x in names if x)


def reading_expand(ing: DocumentIngestor, seeds: list, window: int = 1) -> list:
    nodes = []
    for node in seeds:
        expanded = ing.expand(node, window=window, include_ancestors=False)
        nodes.extend(expanded.passages)
    return unique_nodes(nodes)


def semantic_near_nodes(db: GrafitoDatabase, query: str, *, k: int, expand: int, max_nodes: int = 18) -> list:
    sub = db.semantic_subgraph(
        query,
        k=k,
        index="chunks",
        filter_labels=["Chunk"],
        labels=["Chunk"],
        rel_types=[SEMANTIC_REL],
        expand=expand,
        max_nodes=max_nodes,
    )
    return unique_nodes(sub.nodes)


def hybrid_semantic_nodes(db: GrafitoDatabase, ing: DocumentIngestor, query: str, *, k: int, expand: int, max_nodes: int = 18) -> list:
    # Use DocumentIngestor.hybrid_search here rather than db.hybrid_subgraph.
    # The document helper degrades cleanly when SQLite FTS rejects punctuation in
    # natural-language questions, matching the notebook path under evaluation.
    hits = [{"node": hit.node, "score": hit.score} for hit in ing.hybrid_search(query, k=k)]
    sub = db.subgraph(
        hits,
        labels=["Chunk"],
        rel_types=[SEMANTIC_REL],
        expand=expand,
        max_nodes=max_nodes,
    )
    return unique_nodes(sub.nodes)


def combined_reading_semantic(db: GrafitoDatabase, ing: DocumentIngestor, query: str, *, k: int = 1) -> list:
    seeds = search_hits_to_nodes(ing.search(query, k=k))
    semantic = semantic_near_nodes(db, query, k=k, expand=1, max_nodes=12)
    return unique_nodes([*reading_expand(ing, seeds, window=1), *semantic])


def reranked_nodes(ing: DocumentIngestor, reranker: CrossEncoder, query: str, *, candidates: int = 10, k: int = 3) -> list:
    hits = list(ing.search(query, k=candidates))
    pairs = [(query, node_text(hit.node)) for hit in hits]
    scores = [float(x) for x in reranker.predict(pairs)]
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return unique_nodes(hit.node for hit, _ in ranked[:k])


def term_present(term: str, context: str) -> bool:
    # Terms may be multi-word. Use substring after normalization: simple and auditable.
    return norm(term) in context


def evaluate_nodes(query: EvalQuery, ing: DocumentIngestor, nodes: list, reranker: CrossEncoder) -> dict:
    texts = [node_text(node) for node in nodes]
    context = norm("\n".join(texts))
    paths = [section_path(ing, node) for node in nodes]
    paths_norm = [norm(path) for path in paths]
    expected_hit = any(
        any(expected in path for expected in query.expected_sections)
        for path in paths_norm
    )
    section_precision = 0.0
    if paths_norm:
        section_precision = sum(
            1 for path in paths_norm if any(expected in path for expected in query.expected_sections)
        ) / len(paths_norm)
    terms_found = [term for term in query.evidence_terms if term_present(term, context)]
    term_recall = len(terms_found) / len(query.evidence_terms)
    non_empty = [text for text in texts if text.strip()]
    ce_scores = []
    if non_empty:
        ce_scores = [float(x) for x in reranker.predict([(query.question, text) for text in non_empty])]
    return {
        "chunks": len(nodes),
        "chars": sum(len(text) for text in texts),
        "empty_chunks": len(texts) - len(non_empty),
        "unique_sections": len(set(paths)),
        "expected_section_hit": expected_hit,
        "section_precision": section_precision,
        "term_recall": term_recall,
        "terms_found": ", ".join(terms_found),
        "ce_max": max(ce_scores) if ce_scores else None,
        "ce_mean": statistics.mean(ce_scores) if ce_scores else None,
        "node_ids": ", ".join(str(node.id) for node in nodes),
        "seqs": ", ".join(str(node.properties.get("global_seq")) for node in nodes),
        "sections": " | ".join(dict.fromkeys(paths)),
    }


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return ""
    return str(value)


def write_markdown(rows: list[dict], summary: dict, path: Path) -> None:
    strategies = sorted({row["strategy"] for row in rows})
    lines = [
        "# PDF GraphRAG retrieval evaluation",
        "",
        "Corpus: Anthropic, *Building Effective AI Agents*. Same pipeline as `pdf_chunking_colab.ipynb`.",
        "",
        "This is a retrieval/context evaluation, not a generated-answer benchmark. It asks: do newer graph features add useful evidence, or mostly noise?",
        "",
        "## Corpus build",
        "",
        f"- Sections: {summary['sections']}",
        f"- Passages/chunks: {summary['passages']}",
        f"- Semantic proximity edges: {summary['semantic_edges']} `{SEMANTIC_REL}` over {summary['semantic_nodes']} chunks",
        "- Semantic graph parameters: `k=3`, `min_score=0.45`, `undirected=True`, `labels=['Chunk']`",
        "",
        "## Metrics",
        "",
        "- `section_hit`: at least one retrieved chunk is in the expected section(s).",
        "- `term_recall`: fraction of manually specified evidence terms present in the retrieved context.",
        "- `ce_max` / `ce_mean`: cross-encoder relevance scores for query/chunk pairs (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Higher is better within a query.",
        "- `chunks` / `chars`: context budget proxy. More is not automatically better.",
        "",
        "## Aggregate by strategy",
        "",
        "| strategy | section_hit_rate | avg_term_recall | avg_ce_max | avg_ce_mean | avg_chunks | avg_chars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        srows = [row for row in rows if row["strategy"] == strategy]
        lines.append(
            "| {strategy} | {hit:.2f} | {term:.2f} | {cemax:.2f} | {cemean:.2f} | {chunks:.1f} | {chars:.0f} |".format(
                strategy=strategy,
                hit=sum(1 for r in srows if r["expected_section_hit"]) / len(srows),
                term=statistics.mean(float(r["term_recall"]) for r in srows),
                cemax=statistics.mean(float(r["ce_max"] or 0) for r in srows),
                cemean=statistics.mean(float(r["ce_mean"] or 0) for r in srows),
                chunks=statistics.mean(int(r["chunks"]) for r in srows),
                chars=statistics.mean(int(r["chars"]) for r in srows),
            )
        )
    lines.extend([
        "",
        "## Per-query results",
        "",
    ])
    for query_key in [q.key for q in QUERIES]:
        qrows = [row for row in rows if row["query_key"] == query_key]
        query_text = qrows[0]["question"]
        lines.extend([
            f"### {query_key}",
            "",
            f"Query: {query_text}",
            "",
            "| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for row in sorted(qrows, key=lambda r: r["strategy"]):
            lines.append(
                f"| {row['strategy']} | {row['expected_section_hit']} | {float(row['term_recall']):.2f} | "
                f"{float(row['ce_max'] or 0):.2f} | {float(row['ce_mean'] or 0):.2f} | "
                f"{row['chunks']} | {row['chars']} | {row['seqs']} | {row['terms_found']} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "Use this table defensively. A graph-expanded strategy wins only if it improves evidence coverage or relevance at an acceptable context cost. If it adds many chunks with lower `ce_mean`, it is exploration/visualization value, not proof of better answer quality.",
        "",
        "For article claims, the honest framing is: semantic proximity edges make vector similarity traversable and explainable as a graph. They can improve recall/context assembly, but they do not automatically improve top-k ranking; reranking or budgeting is still needed to control noise.",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db, ing, build = build_corpus()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    strategies: dict[str, Callable[[str], list]] = {
        "vector_top3": lambda q: search_hits_to_nodes(ing.search(q, k=3)),
        "vector_reading_top1": lambda q: reading_expand(ing, search_hits_to_nodes(ing.search(q, k=1)), window=1),
        "vector_reading_top3": lambda q: reading_expand(ing, search_hits_to_nodes(ing.search(q, k=3)), window=1),
        "semantic_near_top1": lambda q: semantic_near_nodes(db, q, k=1, expand=1, max_nodes=12),
        "semantic_near_top3": lambda q: semantic_near_nodes(db, q, k=3, expand=1, max_nodes=18),
        "hybrid_top3": lambda q: search_hits_to_nodes(ing.hybrid_search(q, k=3)),
        "hybrid_reading_top1": lambda q: reading_expand(ing, search_hits_to_nodes(ing.hybrid_search(q, k=1)), window=1),
        "hybrid_semantic_top1": lambda q: hybrid_semantic_nodes(db, ing, q, k=1, expand=1, max_nodes=12),
        "combined_reading_semantic_top1": lambda q: combined_reading_semantic(db, ing, q, k=1),
        "rerank_vector10_top3": lambda q: reranked_nodes(ing, reranker, q, candidates=10, k=3),
    }

    rows: list[dict] = []
    for query in QUERIES:
        for name, strategy in strategies.items():
            nodes = strategy(query.question)
            metrics = evaluate_nodes(query, ing, nodes, reranker)
            rows.append({
                "query_key": query.key,
                "question": query.question,
                "strategy": name,
                **metrics,
            })

    csv_path = OUT_DIR / "pdf_graphrag_eval.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sections": build["ingest"].n_sections,
        "passages": build["ingest"].n_passages,
        "semantic_edges": build["semantic_graph"].edges_created,
        "semantic_nodes": build["semantic_graph"].nodes_processed,
    }
    (OUT_DIR / "pdf_graphrag_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    write_markdown(rows, summary, OUT_DIR / "pdf_graphrag_eval.md")

    print(f"Wrote {csv_path}")
    print(f"Wrote {OUT_DIR / 'pdf_graphrag_eval.md'}")
    print("\nAggregate:")
    for line in (OUT_DIR / "pdf_graphrag_eval.md").read_text().splitlines()[22:35]:
        print(line)

    db.close()


if __name__ == "__main__":
    main()
