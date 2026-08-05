"""Outcomes of bulk operations: document ingest and semantic graph builds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IndexReport:
    """What :meth:`~grafito.GrafitoDatabase.index_documents` actually did.

    ``unresolved`` is the part worth checking: relationship targets that named
    an id no document supplied. They are skipped rather than raising, because a
    dataset slice legitimately references rows outside the slice — but a long
    list here usually means ``id_key`` does not match the ids used in the
    relationship field.
    """

    nodes_created: int = 0
    nodes_updated: int = 0
    relationships_created: int = 0
    #: Number of nodes whose text was embedded.
    embedded: int = 0
    #: External id -> node id, for every row that carried an id.
    ids: dict[Any, int] = field(default_factory=dict)
    #: Relationship targets that could not be resolved to a node.
    unresolved: list[Any] = field(default_factory=list)

    @property
    def nodes(self) -> int:
        """Total nodes touched, created or updated."""
        return self.nodes_created + self.nodes_updated

    def __str__(self) -> str:
        parts = [
            f"{self.nodes_created} created",
            f"{self.nodes_updated} updated",
            f"{self.relationships_created} relationships",
            f"{self.embedded} embedded",
        ]
        if self.unresolved:
            parts.append(f"{len(self.unresolved)} unresolved")
        return f"IndexReport({', '.join(parts)})"


@dataclass
class SemanticGraphReport:
    """What :meth:`~grafito.GrafitoDatabase.create_semantic_graph` materialised.

    ``edges_created`` is the number worth watching: it grows as ``k`` times the
    node count, and those edges live in the same table as the domain's own. A
    build that produced far more edges than expected is usually a ``min_score``
    that is too permissive.
    """

    edges_created: int = 0
    edges_removed: int = 0
    #: Nodes whose neighbourhood was computed.
    nodes_processed: int = 0
    #: Nodes skipped because `approximate` found them already linked.
    nodes_skipped: int = 0
    #: True when `max_edges` stopped the build before every node was processed.
    truncated: bool = False

    def __str__(self) -> str:
        parts = [
            f"{self.edges_created} edges",
            f"{self.nodes_processed} nodes",
        ]
        if self.edges_removed:
            parts.append(f"{self.edges_removed} replaced")
        if self.nodes_skipped:
            parts.append(f"{self.nodes_skipped} skipped")
        if self.truncated:
            parts.append("truncated")
        return f"SemanticGraphReport({', '.join(parts)})"
