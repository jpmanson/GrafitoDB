"""Induced subgraphs: the result of a search, as a graph rather than a list.

A ranked list of hits throws away the thing a graph database knows that a vector
store does not — how the hits relate to each other. :class:`Subgraph` keeps
both: the ranked seeds *and* the edges among them (plus, optionally, their
neighbourhood), so a search result can be visualised, analysed, or packed into
an LLM prompt without a second round of queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import Node, Relationship


@dataclass
class Subgraph:
    """Nodes and relationships selected by a search, with their provenance.

    ``scores`` and ``hops`` are what make this explainable: every node either
    matched the query directly (``hops == 0``, with a score) or was reached by
    expansion from one that did (``hops >= 1``, no score). Without them a
    subgraph is an undifferentiated blob and there is no way to tell a strong
    match from something two hops away.
    """

    nodes: list[Node] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    #: Seed hits, best first: ``[{"node": Node, "score": float}, ...]``.
    seeds: list[dict[str, Any]] = field(default_factory=list)
    #: Seed node id -> retrieval score. Expanded nodes are absent.
    scores: dict[int, float] = field(default_factory=dict)
    #: Node id -> hop distance from the nearest seed (0 for seeds).
    hops: dict[int, int] = field(default_factory=dict)
    #: Ordered node ids of each route found, when the subgraph came from
    #: :meth:`~grafito.GrafitoDatabase.path_context`. Empty otherwise.
    paths: list[list[int]] = field(default_factory=list)

    def node_ids(self) -> list[int]:
        """Ids of every node in the subgraph, in insertion order."""
        return [node.id for node in self.nodes]

    def seed_ids(self) -> list[int]:
        """Ids of the seed nodes, best-scoring first."""
        return [hit["node"].id for hit in self.seeds]

    def is_empty(self) -> bool:
        return not self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def to_networkx(self, directed: bool = True):
        """Build a NetworkX graph of this subgraph.

        Nodes carry ``labels``, ``properties``, ``uri``, plus the ``score`` and
        ``hops`` provenance; edges carry ``type``, ``properties`` and ``uri``.
        Suitable for :func:`grafito.integrations.viz.export_graph` and for the
        ``graph=`` argument of :meth:`~grafito.GrafitoDatabase.centrality`.
        """
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - networkx is a core dependency
            from .exceptions import DatabaseError

            raise DatabaseError(
                "networkx is not installed. Install with `pip install networkx`."
            ) from exc

        graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
        for node in self.nodes:
            graph.add_node(
                node.id,
                labels=list(node.labels),
                properties=dict(node.properties),
                uri=getattr(node, "uri", None),
                score=self.scores.get(node.id),
                hops=self.hops.get(node.id),
            )
        for rel in self.relationships:
            graph.add_edge(
                rel.source_id,
                rel.target_id,
                key=rel.id,
                id=rel.id,
                type=rel.type,
                properties=dict(rel.properties),
                uri=getattr(rel, "uri", None),
            )
        return graph
