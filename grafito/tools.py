"""Tools: the framework-free contract for exposing capabilities as LLM tools.

This module hosts three things, in ascending specificity:

* :class:`ToolSet` — the contract every tool provider satisfies: OpenAI-style
  ``schemas`` plus a matching ``call(name, args) -> str``. It lives here, in the
  core, rather than in :mod:`grafito.okf`, because it is not OKF-specific:
  :class:`~grafito.okf.BundleTools` is one toolset, :class:`GraphTools` another.
* :class:`ToolRegistry` — several toolsets presented as one, the seam every tool
  *consumer* shares (an agent loop, an MCP server). Consumers depend on this and
  the :class:`ToolSet` protocol, never on a concrete toolset, which is what lets
  the same server front an OKF bundle or a plain graph unchanged.
* :class:`GraphTools` / :class:`CypherTools` — the graph tool tiers (escalón 2-3
  of the MCP proposal), over a plain :class:`~grafito.GrafitoDatabase`.

This module imports nothing from :mod:`grafito.okf` on purpose — a graph
deployment must not drag OKF in. Altitude climbs across the graph tiers:
``graph_schema``/``graph_neighbors``/``text_search`` are intent-shaped and safe;
``graph_query`` is raw Cypher, read-only and row-capped, the escape hatch for
exact structural questions the higher tools cannot express.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .database import GrafitoDatabase
    from .models import Node


@runtime_checkable
class ToolSet(Protocol):
    """Anything with OpenAI-style tool ``schemas`` plus a matching ``call``.

    :class:`~grafito.okf.BundleTools` and :class:`GraphTools` are two such
    toolsets; a :class:`ToolRegistry` is a third (it aggregates others). No base
    class needed — any object with this shape works, the same spirit as an
    injected model callable. Tool errors are conventionally returned as a JSON
    ``{"error": ...}`` string so a bad call does not kill the consumer's loop.
    """

    schemas: list[dict]

    def call(self, name: str, args: dict) -> str:
        ...


class ToolRegistry:
    """Several :class:`ToolSet`\\ s presented as one: merged ``schemas`` plus a
    single ``call`` that routes each tool to the toolset that owns it.

    This is the seam every tool *consumer* shares — an agent loop, an MCP
    server, any other loop — so the aggregation lives here once instead of
    being re-implemented against each. Consumers depend on this and on the
    :class:`ToolSet` protocol, never on a concrete toolset like
    :class:`~grafito.okf.BundleTools`: that is what lets the same server front an
    OKF bundle (``BundleTools(kb)``) and a plain graph (a ``ToolSet`` over
    ``GrafitoDatabase``) without change. A registry is itself a :class:`ToolSet`
    — it has ``schemas`` and ``call`` — so registries nest.

    Tool names must be unique across the toolsets; a collision raises
    ``ValueError`` at construction, before any tool can run. A call for a name
    no toolset owns comes back as ``{"error": ...}`` (JSON), the same shape an
    individual tool error takes, so the caller handles both the same way.
    """

    def __init__(self, toolsets: "list[ToolSet]") -> None:
        self.schemas: list[dict] = []
        self._dispatch: dict[str, ToolSet] = {}
        for toolset in toolsets:
            for schema in toolset.schemas:
                name = schema["function"]["name"]
                if name in self._dispatch:
                    raise ValueError(f"Duplicate tool name {name!r} across toolsets")
                self._dispatch[name] = toolset
                self.schemas.append(schema)

    @property
    def names(self) -> list[str]:
        """The tool names this registry can dispatch, in schema order."""
        return [schema["function"]["name"] for schema in self.schemas]

    def __contains__(self, name: str) -> bool:
        return name in self._dispatch

    def call(self, name: str, args: dict) -> str:
        """Route one tool call to its owning toolset; unknown names error.

        The routed toolset owns its own error policy (JSON ``{"error": ...}``
        by default, or raising when built with ``raise_errors=True``); this
        only adds the unknown-name case.
        """
        toolset = self._dispatch.get(name)
        if toolset is None:
            return json.dumps({"error": f"Unknown tool {name!r}"})
        return toolset.call(name, args)

# Cypher clauses that can mutate the graph. CypherTools rejects any query that
# contains one as a whole word — a deliberately conservative guard: it may
# refuse a read query that merely mentions one of these words in a string or
# property name, which is the safe direction to err for a read-only tool.
_MUTATING = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP)\b", re.IGNORECASE
)


def _node_dict(node: "Node") -> dict:
    return {"id": node.id, "labels": list(node.labels), "properties": dict(node.properties)}


def _dispatch(owner: Any, name: str, args: dict, raise_errors: bool) -> str:
    """Run ``owner._{name}(**args)`` and return its JSON result, or a JSON error.

    Shared by both toolsets: a name the toolset does not expose is rejected
    here (not merely absent from ``schemas``), so a stale or invented name
    cannot reach a private method.
    """
    try:
        if name not in owner.enabled:
            raise ValueError(f"Unknown tool {name!r}")
        result = owner._call(name, args)
    except Exception as exc:
        if raise_errors:
            raise
        result = {"error": str(exc)}
    return json.dumps(result, ensure_ascii=False, default=str)


class GraphTools:
    """Structured, read-only access to a graph (escalón 2).

    ``graph_schema`` orients the model — what labels, relationship types and
    indexes exist — so it can ask meaningful questions without guessing the
    shape. ``text_search`` finds nodes by keyword (needs FTS configured on the
    database; returns nothing when it is not). ``graph_neighbors`` traverses from
    a node id — the ids that ``text_search`` and ``graph_query`` return — so the
    three chain into exploration without any Cypher: orient, find, traverse.

    Read-only: nothing here mutates the graph. ``raise_errors`` matches
    :class:`~grafito.okf.BundleTools` — wrapped ``{"error": ...}`` by default,
    propagated when a framework drives its own retry.
    """

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "graph_schema",
                "description": "Orient in the graph: the labels, relationship types, "
                "node count and indexes that exist. Call this first to learn the shape "
                "before searching or querying.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "text_search",
                "description": "Find nodes by keyword (full-text). Returns ranked nodes "
                "with their id, labels and properties. Use the ids with graph_neighbors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vector_search",
                "description": "Find nodes by meaning (semantic/vector search) over one of "
                "the graph's vector indexes — see graph_schema for their names. Returns "
                "ranked nodes with id, labels, properties and score. Errors when the "
                "index has no embedding function configured.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                        "index": {"type": "string", "default": "default"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_neighbors",
                "description": "Traverse the graph from a node: its neighbours by "
                "relationship direction ('outgoing'/'incoming'/'both'), optionally "
                "restricted to one relationship type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "integer"},
                        "direction": {
                            "type": "string",
                            "enum": ["outgoing", "incoming", "both"],
                            "default": "both",
                        },
                        "rel_type": {"type": "string", "description": "e.g. 'KNOWS'"},
                    },
                    "required": ["node_id"],
                },
            },
        },
    ]
    enabled = frozenset({"graph_schema", "text_search", "vector_search", "graph_neighbors"})

    def __init__(self, db: "GrafitoDatabase", *, raise_errors: bool = False) -> None:
        self.db = db
        self.raise_errors = raise_errors

    def call(self, name: str, args: dict) -> str:
        return _dispatch(self, name, args, self.raise_errors)

    def _call(self, name: str, args: dict) -> Any:
        return getattr(self, f"_{name}")(**args)

    def _graph_schema(self) -> dict:
        labels = sorted(
            {label for row in self.db.execute("MATCH (n) RETURN DISTINCT labels(n) AS l")
             for label in (row["l"] or [])}
        )
        rel_types = sorted(
            row["t"]
            for row in self.db.execute("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t")
            if row["t"]
        )
        return {
            "labels": labels,
            "relationship_types": rel_types,
            "node_count": self.db.get_node_count(),
            "indexes": self.db.list_indexes(),
            # Name + dim only: enough for the model to pick one for vector_search,
            # without dumping each index's embedder config.
            "vector_indexes": [
                {"name": vi["name"], "dim": vi["dim"]}
                for vi in self.db.list_vector_indexes()
            ],
        }

    def _text_search(self, query: str, k: int = 5) -> list[dict]:
        hits = self.db.text_search(query, k=k)
        out = []
        for hit in hits:
            if hit["entity_type"] != "node":
                continue
            item = _node_dict(hit["entity"])
            item["score"] = round(hit["score"], 4)
            out.append(item)
        return out

    def _vector_search(self, query: str, k: int = 5, index: str = "default") -> list[dict]:
        hits = self.db.semantic_search(query, k=k, index=index)
        out = []
        for hit in hits:
            item = _node_dict(hit["node"])
            item["score"] = round(hit["score"], 4)
            out.append(item)
        return out

    def _graph_neighbors(
        self, node_id: int, direction: str = "both", rel_type: str | None = None
    ) -> list[dict]:
        neighbors = self.db.get_neighbors(node_id, direction=direction, rel_type=rel_type)
        return [_node_dict(node) for node in neighbors]


class CypherTools:
    """A read-only Cypher escape hatch (escalón 3).

    ``graph_query`` runs arbitrary Cypher for the exact, structural questions the
    higher tools cannot express — counts, shortest paths, aggregations. Two
    guards keep it an escape hatch rather than a console:

    * **read-only** — a query containing a mutating clause (``CREATE``,
      ``MERGE``, ``DELETE``, ``DETACH``, ``SET``, ``REMOVE``, ``DROP``) is
      refused. The check is conservative (whole-word, case-insensitive), so it
      errs toward refusing a legitimate read rather than allowing a write.
    * **row-capped** — at most ``max_rows`` rows come back, with a ``truncated``
      flag when more existed. Note the cap is applied after execution, so a
      query that would return an enormous result still materialises it; a
      ``LIMIT`` in the query itself is the real bound.

    Writes (escalón 4) are intentionally not here: this toolset cannot mutate the
    graph even when a caller wants it to.
    """

    def __init__(
        self, db: "GrafitoDatabase", *, max_rows: int = 100, raise_errors: bool = False
    ) -> None:
        self.db = db
        self.max_rows = max_rows
        self.raise_errors = raise_errors
        self.enabled = frozenset({"graph_query"})
        self.schemas = [
            {
                "type": "function",
                "function": {
                    "name": "graph_query",
                    "description": "Run a read-only Cypher query for exact structural "
                    "answers (counts, paths, aggregations) the other tools cannot "
                    "express. Reads only — mutating clauses are rejected. Add a LIMIT "
                    f"for large results; at most {max_rows} rows are returned.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

    def call(self, name: str, args: dict) -> str:
        return _dispatch(self, name, args, self.raise_errors)

    def _call(self, name: str, args: dict) -> Any:
        return getattr(self, f"_{name}")(**args)

    def _graph_query(self, query: str) -> dict:
        if _MUTATING.search(query):
            raise ValueError(
                "Read-only: this query contains a mutating clause "
                "(CREATE/MERGE/DELETE/DETACH/SET/REMOVE/DROP) and was refused."
            )
        rows = self.db.execute(query)
        truncated = len(rows) > self.max_rows
        return {"rows": rows[: self.max_rows], "count": len(rows), "truncated": truncated}
