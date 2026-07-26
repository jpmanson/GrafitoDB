"""``DocumentTools`` — passage-level retrieval as an MCP/agent tool tier.

Hangs on a :class:`~grafito.document.DocumentIngestor`, so it knows about managed
passages, reading-order ``expand`` and budgeted ``pack`` — the retrieval semantics
``GraphTools.vector_search`` cannot express. OKF-free: the ingestor hangs on a
:class:`~grafito.GrafitoDatabase`, so this tier serves a plain graph unchanged.

Read-only: ``document_context``/``search``/``expand``/``toc``/``load_sections``.
Ingest/replace/delete (the write tier) are intentionally not here yet — this
toolset cannot mutate the graph. Errors come back as JSON ``{"error": ...}`` like
every other toolset, so a bad call does not kill the consumer's loop.

Altitude, one tier up from :class:`~grafito.GraphTools`: ``document_context`` is
the one-shot RAG tool (search → expand → pack, grounded + budgeted); the lower
tools let an agent drive the pipeline itself — vector path
(``document_search`` → ``document_expand``) or tree/agentic path
(``document_toc`` → ``document_load_sections``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..tools import _dispatch, _node_dict

if TYPE_CHECKING:
    from .ingest import DocumentIngestor
    from .types import ExpandResult, PackedContext, SearchHit


def _ref(document_ref: Any) -> int | str:
    """Accept an int id or a string ``document_key``; coerce numeric strings to int."""
    if isinstance(document_ref, str) and document_ref.lstrip("-").isdigit():
        return int(document_ref)
    return document_ref


def _hit_dict(hit: "SearchHit") -> dict:
    item = _node_dict(hit.node)
    item["score"] = round(hit.score, 4)
    item["owner_document_id"] = hit.owner_document_id
    item["global_seq"] = hit.global_seq
    if hit.view is not None:
        item["view"] = hit.view
    return item


def _packed_dict(packed: "PackedContext") -> dict:
    return {
        "text": packed.text,
        "order": packed.order,
        "truncated": packed.truncated,
        "citations": [
            {
                "node_id": s.node_id,
                "document_id": s.document_id,
                "section_path": s.section_path,
                "char_start": s.char_start,
                "char_end": s.char_end,
            }
            for s in packed.segments
        ],
    }


def _expand_dict(result: "ExpandResult") -> dict:
    return {
        "center_id": result.center.id,
        "passages": [_node_dict(n) for n in result.passages],
        "parent": _node_dict(result.parent) if result.parent else None,
        "section": _node_dict(result.section) if result.section else None,
        "ancestors": [_node_dict(n) for n in result.ancestors],
    }


class DocumentTools:
    """Passage-level RAG over a :class:`DocumentIngestor` (read-only).

    Read-only by construction: only ``document_context``/``search``/``expand``/
    ``toc``/``load_sections`` are exposed; nothing here mutates the graph.
    ``raise_errors`` matches the graph tiers — wrapped ``{"error": ...}`` by
    default, propagated when a framework drives its own retry.

    ``default_window`` / ``default_budget_tokens`` are deployment decisions (like
    ``budget_tokens`` in the OKF context tool), fixed here rather than per call,
    though ``document_context``/``document_expand`` still accept per-call overrides.
    """

    enabled = frozenset(
        {
            "document_context",
            "document_search",
            "document_expand",
            "document_toc",
            "document_load_sections",
        }
    )

    def __init__(
        self,
        ingestor: "DocumentIngestor",
        *,
        default_window: int = 1,
        default_budget_tokens: int = 1500,
        raise_errors: bool = False,
    ) -> None:
        self.ing = ingestor
        self.default_window = default_window
        self.default_budget_tokens = default_budget_tokens
        self.raise_errors = raise_errors
        self.schemas = [self._SCHEMAS[name] for name in self._SCHEMA_ORDER]

    # --- ToolSet contract -------------------------------------------------
    def call(self, name: str, args: dict) -> str:
        return _dispatch(self, name, args, self.raise_errors)

    def _call(self, name: str, args: dict) -> Any:
        return getattr(self, f"_{name}")(**args)

    # --- tools ------------------------------------------------------------
    def _document_context(
        self,
        query: str,
        k: int = 5,
        window: int | None = None,
        max_tokens: int | None = None,
        include_ancestors: bool = False,
    ) -> dict:
        """search → expand → pack, one shot: grounded, budgeted passage context."""
        window = self.default_window if window is None else window
        budget = self.default_budget_tokens if max_tokens is None else max_tokens
        hits = self.ing.search(query, k=k)
        if not hits:
            return {"query": query, "text": "", "citations": [], "documents": []}
        # Expand each hit within its reading window; dedup passages by node id so a
        # passage shared by two windows is packed once.
        by_id: dict[int, Any] = {}
        for hit in hits:
            result = self.ing.expand(
                hit.node,
                window=window,
                include_parent=False,
                include_ancestors=include_ancestors,
            )
            for node in result.passages:
                by_id.setdefault(node.id, node)
        scores = {hit.node.id: hit.score for hit in hits}
        packed = self.ing.pack(
            list(by_id.values()),
            max_tokens=budget,
            order="reading",
            deduplicate_overlap=True,
            include_citations=True,
            scores=scores,
        )
        out = _packed_dict(packed)
        out["query"] = query
        out["documents"] = sorted({h.owner_document_id for h in hits if h.owner_document_id})
        return out

    def _document_search(
        self, query: str, k: int = 5, owner_document_id: int | None = None
    ) -> list[dict]:
        hits = self.ing.search(query, k=k, owner_document_id=owner_document_id)
        return [_hit_dict(hit) for hit in hits]

    def _document_expand(
        self,
        node_id: int,
        window: int | None = None,
        include_parent: bool = True,
        include_ancestors: bool = False,
    ) -> dict:
        window = self.default_window if window is None else window
        return _expand_dict(
            self.ing.expand(
                node_id,
                window=window,
                include_parent=include_parent,
                include_ancestors=include_ancestors,
            )
        )

    def _document_toc(self, document_ref: Any) -> list[dict]:
        return self.ing.toc(_ref(document_ref), as_dict=True)

    def _document_load_sections(self, document_ref: Any, node_keys: list[str]) -> list[dict]:
        nodes = self.ing.load_sections(_ref(document_ref), node_keys)
        return [_node_dict(node) for node in nodes]

    # --- schemas ----------------------------------------------------------
    _SCHEMA_ORDER = (
        "document_context",
        "document_search",
        "document_expand",
        "document_toc",
        "document_load_sections",
    )

    _SCHEMAS: dict[str, dict] = {
        "document_context": {
            "type": "function",
            "function": {
                "name": "document_context",
                "description": "Retrieve grounded, budgeted context for a question over "
                "ingested documents: finds the best passages, expands each to its reading "
                "neighbours, and packs them under a token budget with citations. The "
                "one-shot RAG tool — prefer it over document_search unless you need to "
                "drive expand/pack yourself.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {
                            "type": "integer",
                            "default": 5,
                            "description": "How many passages to seed from.",
                        },
                        "window": {
                            "type": "integer",
                            "description": "Reading-order neighbours (±) per hit.",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Token budget for the packed context.",
                        },
                        "include_ancestors": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
        },
        "document_search": {
            "type": "function",
            "function": {
                "name": "document_search",
                "description": "Find passages by meaning over managed documents (active "
                "generation, passage role). Returns ranked passages with node id, score, "
                "owner_document_id and global_seq. Use the ids with document_expand.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                        "owner_document_id": {
                            "type": "integer",
                            "description": "Restrict to one document.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        "document_expand": {
            "type": "function",
            "function": {
                "name": "document_expand",
                "description": "Expand a passage to its reading-order neighbours (±window "
                "by global_seq), optionally with parent document and section ancestors. "
                "Use the node id from document_search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "integer"},
                        "window": {"type": "integer"},
                        "include_parent": {"type": "boolean", "default": True},
                        "include_ancestors": {"type": "boolean", "default": False},
                    },
                    "required": ["node_id"],
                },
            },
        },
        "document_toc": {
            "type": "function",
            "function": {
                "name": "document_toc",
                "description": "Table of contents for a document: the section tree "
                "(titles + summaries, no bodies) of its active version. Use it to decide "
                "which sections matter, then document_load_sections to load them.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_ref": {
                            "type": ["string", "integer"],
                            "description": "Document id or document_key.",
                        },
                    },
                    "required": ["document_ref"],
                },
            },
        },
        "document_load_sections": {
            "type": "function",
            "function": {
                "name": "document_load_sections",
                "description": "Load section nodes by their per-document node_key (from "
                "document_toc). Returns the section nodes with their properties.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_ref": {"type": ["string", "integer"]},
                        "node_keys": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["document_ref", "node_keys"],
                },
            },
        },
    }
