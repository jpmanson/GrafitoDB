"""Graph analysis over NetworkX: centrality and community detection.

These are thin, opinionated wrappers. They exist so that "which nodes matter?"
and "what clusters are there?" are one call against the database instead of an
export plus a NetworkX incantation, and so that the *choice* of algorithm is a
documented string rather than an import path.

Everything here operates on plain NetworkX graphs and node ids; the
``GrafitoDatabase`` methods (:meth:`~grafito.GrafitoDatabase.centrality`,
:meth:`~grafito.GrafitoDatabase.communities`) resolve ids back to nodes.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .exceptions import DatabaseError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import Node

#: Centrality measures, mapped to how they are computed.
CENTRALITY_KINDS = (
    "pagerank",
    "degree",
    "in_degree",
    "out_degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "harmonic",
)

#: Community detection algorithms. ``lpa`` is an alias for label propagation.
COMMUNITY_ALGORITHMS = ("louvain", "greedy", "lpa", "label_propagation")


@dataclass
class Community:
    """One detected community.

    ``id`` is the community's index in the returned list (communities are
    ordered largest first), not a stable identifier across runs — community
    detection is not deterministic across graph edits.
    """

    id: int
    nodes: list[Node]
    size: int
    #: Populated by topic labelling; empty for plain community detection.
    terms: list[str] = field(default_factory=list)
    label: str | None = None

    def ids(self) -> list[int]:
        """Node ids in this community."""
        return [node.id for node in self.nodes]

    def __len__(self) -> int:
        return self.size


def _require_networkx():
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - networkx is a core dependency
        raise DatabaseError(
            "networkx is not installed. Install with `pip install networkx`."
        ) from exc
    return nx


def compute_centrality(
    graph: Any,
    kind: str = "pagerank",
    *,
    weight: str | None = None,
    **kwargs: Any,
) -> dict[int, float]:
    """Score every node in ``graph`` by a centrality measure.

    Args:
        graph: A NetworkX graph.
        kind: One of :data:`CENTRALITY_KINDS`.
        weight: Edge attribute to use as weight, or ``None`` for unweighted.
        **kwargs: Passed through to the underlying NetworkX function.

    Returns:
        Mapping of node id to score.
    """
    nx = _require_networkx()
    if kind not in CENTRALITY_KINDS:
        raise DatabaseError(
            f"Unknown centrality kind '{kind}'. Expected one of: {', '.join(CENTRALITY_KINDS)}"
        )
    if graph.number_of_nodes() == 0:
        return {}

    directed = graph.is_directed()
    if kind in {"in_degree", "out_degree"} and not directed:
        raise DatabaseError(
            f"'{kind}' centrality requires a directed graph; pass directed=True"
        )
    if weight is not None and kind in {"degree", "in_degree", "out_degree"}:
        raise DatabaseError(
            f"'{kind}' centrality counts edges and ignores weights; drop `weight=`"
        )

    if weight is not None:
        # Normalise the weight into the "weight" attribute, reading through the
        # `properties` dict that to_networkx() nests relationship properties in.
        # Without this a weight= naming a relationship property would silently
        # fall back to NetworkX's default of 1 for every edge.
        graph = _as_simple_graph(graph, weight=weight)
        weight = "weight"

    if kind == "pagerank":
        try:
            return nx.pagerank(graph, weight=weight, **kwargs)
        except ImportError:
            # NetworkX routes pagerank through scipy, which GrafitoDB does not
            # depend on — adding it would more than double install size for one
            # measure. Fall back to the same power iteration it implements.
            return _pagerank(graph, weight=weight, **kwargs)
    if kind == "degree":
        return dict(nx.degree_centrality(graph, **kwargs))
    if kind == "in_degree":
        return dict(nx.in_degree_centrality(graph, **kwargs))
    if kind == "out_degree":
        return dict(nx.out_degree_centrality(graph, **kwargs))
    if kind == "betweenness":
        return dict(nx.betweenness_centrality(graph, weight=weight, **kwargs))
    if kind == "closeness":
        # closeness_centrality takes `distance`, not `weight`.
        if weight is not None:
            kwargs.setdefault("distance", weight)
        return dict(nx.closeness_centrality(graph, **kwargs))
    if kind == "harmonic":
        if weight is not None:
            kwargs.setdefault("distance", weight)
        return dict(nx.harmonic_centrality(graph, **kwargs))
    if kind == "eigenvector":
        simple = _as_simple_graph(graph, weight=weight)
        try:
            return dict(nx.eigenvector_centrality(simple, weight="weight", **kwargs))
        except nx.PowerIterationFailedConvergence as exc:
            raise DatabaseError(
                "eigenvector centrality did not converge; try kind='pagerank', "
                "or raise max_iter"
            ) from exc
    raise DatabaseError(f"Unhandled centrality kind '{kind}'")  # pragma: no cover


def detect_communities(
    graph: Any,
    algorithm: str = "louvain",
    *,
    weight: str | None = None,
    resolution: float = 1.0,
    seed: int | None = None,
    **kwargs: Any,
) -> list[set[int]]:
    """Partition ``graph`` into communities, largest first.

    Community detection is defined on undirected graphs, so a directed graph is
    treated as undirected here: edge direction is dropped, and parallel edges
    collapse into a single weighted edge. This is a real loss of information —
    it is inherent to modularity-based methods, not to this wrapper.

    Args:
        graph: A NetworkX graph.
        algorithm: One of :data:`COMMUNITY_ALGORITHMS`.
        weight: Edge attribute to use as weight, or ``None`` for unweighted.
        resolution: Higher values yield more, smaller communities
            (``louvain`` and ``greedy`` only).
        seed: Seed for the algorithms that are randomised (``louvain``, ``lpa``),
            for reproducible partitions.

    Returns:
        List of node-id sets, ordered by descending size.
    """
    nx = _require_networkx()
    if algorithm not in COMMUNITY_ALGORITHMS:
        raise DatabaseError(
            f"Unknown community algorithm '{algorithm}'. "
            f"Expected one of: {', '.join(COMMUNITY_ALGORITHMS)}"
        )
    if graph.number_of_nodes() == 0:
        return []

    # _as_simple_graph normalises whatever `weight` named into "weight".
    undirected = _as_simple_graph(graph, weight=weight, undirected=True)

    if algorithm == "louvain":
        groups = nx.community.louvain_communities(
            undirected, weight="weight", resolution=resolution, seed=seed, **kwargs
        )
    elif algorithm == "greedy":
        groups = nx.community.greedy_modularity_communities(
            undirected, weight="weight", resolution=resolution, **kwargs
        )
    else:  # lpa / label_propagation
        if seed is not None:
            kwargs.setdefault("seed", seed)
        groups = nx.community.asyn_lpa_communities(undirected, weight="weight", **kwargs)

    result = [set(group) for group in groups]
    result.sort(key=len, reverse=True)
    return result


#: How term scores are computed when labelling communities.
LABEL_SCORINGS = ("tfidf", "frequency")

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def label_communities(
    groups: list[list[str]],
    *,
    terms: int = 4,
    scoring: str = "tfidf",
    stopwords: set[str] | None = None,
    min_length: int = 3,
) -> list[tuple[list[str], str]]:
    """Pick the terms that distinguish each group's text from the others.

    Args:
        groups: One list of text fragments per community, in the same order as
            the communities they describe.
        terms: Terms to keep per community.
        scoring: ``"tfidf"`` weights a term by how concentrated it is in one
            community, so words common to all of them score near zero.
            ``"frequency"`` is a plain count, which mostly surfaces whatever the
            corpus says most often.
        stopwords: Words to drop outright. With ``tfidf`` this is rarely needed
            — ubiquitous words are already discounted — but it helps for terms
            frequent in *some* communities and meaningless in all.
        min_length: Shortest term to keep.

    Returns:
        ``(terms, label)`` per group, where ``label`` joins the terms with
        ``", "``. Empty for a community whose nodes carry no usable text.
    """
    if scoring not in LABEL_SCORINGS:
        raise DatabaseError(
            f"Unknown label scoring '{scoring}'. Expected one of: {', '.join(LABEL_SCORINGS)}"
        )
    if terms <= 0:
        raise DatabaseError("terms must be a positive integer")

    stop = {word.lower() for word in (stopwords or set())}
    counts: list[Counter[str]] = []
    for fragments in groups:
        counter: Counter[str] = Counter()
        for fragment in fragments:
            for token in _WORD.findall((fragment or "").lower()):
                if len(token) >= min_length and token not in stop:
                    counter[token] += 1
        counts.append(counter)

    # Document frequency across communities, not across nodes: the question is
    # which terms single a community out from its siblings.
    appearances: Counter[str] = Counter()
    for counter in counts:
        appearances.update(counter.keys())

    total = len(counts) or 1
    results: list[tuple[list[str], str]] = []
    for counter in counts:
        if not counter:
            results.append(([], ""))
            continue
        if scoring == "frequency":
            scored = {term: float(count) for term, count in counter.items()}
        else:
            occurrences = sum(counter.values()) or 1
            scored = {
                term: (count / occurrences) * math.log(total / appearances[term] + 1.0)
                for term, count in counter.items()
            }
        # Alphabetical tiebreak so equal scores do not depend on dict order.
        top = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:terms]
        chosen = [term for term, _ in top]
        results.append((chosen, ", ".join(chosen)))
    return results


def _pagerank(
    graph: Any,
    *,
    alpha: float = 0.85,
    weight: str | None = None,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    **_: Any,
) -> dict[int, float]:
    """PageRank by power iteration, without scipy.

    Mirrors ``nx.pagerank``'s contract (same defaults, same dangling-node
    handling, so results agree to within ``tol``). Used only when scipy is
    absent; NetworkX's scipy path is faster on large graphs.
    """
    nx = _require_networkx()
    collapsed = _as_simple_graph(graph, weight=weight)
    # stochastic_graph is directed-only; an undirected graph becomes a pair of
    # opposing edges, which is how nx.pagerank treats it too.
    directed = collapsed.to_directed() if not collapsed.is_directed() else collapsed.copy()
    simple = nx.stochastic_graph(directed, weight="weight")
    n = len(simple)
    if n == 0:
        return {}

    scores = dict.fromkeys(simple, 1.0 / n)
    teleport = 1.0 / n
    dangling = [node for node in simple if simple.out_degree(node, weight="weight") == 0.0]

    for _iteration in range(max_iter):
        previous = scores
        scores = dict.fromkeys(previous.keys(), 0.0)
        leaked = alpha * sum(previous[node] for node in dangling)
        for node in previous:
            for neighbour, edge in simple[node].items():
                scores[neighbour] += alpha * previous[node] * edge["weight"]
        for node in scores:
            scores[node] += leaked * teleport + (1.0 - alpha) * teleport
        if sum(abs(scores[node] - previous[node]) for node in scores) < n * tol:
            return scores
    raise DatabaseError(
        f"pagerank did not converge in {max_iter} iterations; raise max_iter"
    )


def _as_simple_graph(graph: Any, *, weight: str | None = None, undirected: bool = False) -> Any:
    """Collapse a multigraph into a simple graph, summing parallel edges.

    Modularity and eigenvector methods are defined on simple graphs. Collapsing
    explicitly — rather than letting NetworkX pick — keeps parallel edges
    meaningful: two ``KNOWS`` edges between the same pair become weight 2.

    The result always carries the summed value under the ``"weight"`` attribute,
    whatever the source attribute was named, so callers pass ``weight="weight"``
    to NetworkX regardless of the caller's chosen property.
    """
    nx = _require_networkx()
    directed = graph.is_directed() and not undirected

    if weight is None and not graph.is_multigraph() and directed == graph.is_directed():
        # Nothing to collapse and no attribute to rename; NetworkX defaults any
        # missing "weight" to 1, which is exactly the unweighted reading.
        return graph

    simple = nx.DiGraph() if directed else nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        if source == target:
            continue
        increment = 1.0
        if weight is not None:
            raw = data.get(weight)
            if raw is None and isinstance(data.get("properties"), dict):
                raw = data["properties"].get(weight)
            if raw is not None:
                increment = float(raw)
        if simple.has_edge(source, target):
            simple[source][target]["weight"] += increment
        else:
            simple.add_edge(source, target, weight=increment)
    return simple
