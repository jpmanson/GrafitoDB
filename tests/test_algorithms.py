"""Centrality and community detection over the graph."""

import pytest

from grafito import Community, GrafitoDatabase
from grafito.exceptions import DatabaseError


@pytest.fixture
def barbell() -> GrafitoDatabase:
    """Two triangles joined by a single bridge node.

    Structure is deliberate: `bridge` is weak by degree but maximal by
    betweenness, so the two measures must disagree. `SEMANTIC_SIMILAR` edges
    cross-link the clusters, standing in for derived edges that would wreck any
    analysis that fails to exclude them.
    """
    db = GrafitoDatabase(':memory:')
    ids = {}
    for name in ["a1", "a2", "a3", "bridge", "b1", "b2", "b3"]:
        ids[name] = db.create_node(labels=["N"], properties={"name": name}).id
    for source, target in [
        ("a1", "a2"), ("a2", "a3"), ("a3", "a1"),
        ("a1", "bridge"), ("bridge", "b1"),
        ("b1", "b2"), ("b2", "b3"), ("b3", "b1"),
    ]:
        db.create_relationship(ids[source], ids[target], "LINK")
    for source, target in [("a1", "b3"), ("a2", "b2"), ("a3", "b1")]:
        db.create_relationship(ids[source], ids[target], "SEMANTIC_SIMILAR", {"score": 0.9})
    db.node_ids = ids
    yield db
    db.close()


def _names(results: list[dict]) -> list[str]:
    return [row["node"].properties["name"] for row in results]


# --- centrality ------------------------------------------------------------


def test_betweenness_finds_the_bridge(barbell):
    results = barbell.centrality(
        "betweenness", exclude_rel_types=["SEMANTIC_SIMILAR"], directed=False
    )
    assert _names(results)[0] == "bridge"


def test_centrality_results_are_sorted_descending(barbell):
    results = barbell.centrality("pagerank")
    scores = [row["score"] for row in results]
    assert scores == sorted(scores, reverse=True)
    assert len(results) == 7


def test_limit_truncates_to_top_n(barbell):
    assert len(barbell.centrality("pagerank", limit=3)) == 3


def test_excluding_relationship_types_changes_the_ranking(barbell):
    with_noise = barbell.centrality("degree", directed=False)
    without = barbell.centrality(
        "degree", exclude_rel_types=["SEMANTIC_SIMILAR"], directed=False
    )
    assert with_noise != without


def test_rel_types_restricts_to_a_single_type(barbell):
    results = barbell.centrality(
        "degree", rel_types=["SEMANTIC_SIMILAR"], directed=False
    )
    scored = {row["node"].properties["name"]: row["score"] for row in results}
    # Only the six cross-link endpoints have any degree at all.
    assert scored["bridge"] == 0.0
    assert scored["a1"] > 0.0


@pytest.mark.parametrize(
    "kind", ["pagerank", "degree", "betweenness", "closeness", "harmonic", "eigenvector"]
)
def test_every_centrality_kind_runs(barbell, kind):
    results = barbell.centrality(kind, directed=False)
    assert len(results) == 7
    assert all(isinstance(row["score"], float) for row in results)


def test_directed_only_kinds(barbell):
    assert len(barbell.centrality("in_degree", directed=True)) == 7
    with pytest.raises(DatabaseError, match="requires a directed graph"):
        barbell.centrality("in_degree", directed=False)


def test_unknown_centrality_kind_is_rejected(barbell):
    with pytest.raises(DatabaseError, match="Unknown centrality kind"):
        barbell.centrality("nope")


def test_degree_centrality_rejects_a_weight(barbell):
    """Silently ignoring the weight would be worse than refusing it."""
    with pytest.raises(DatabaseError, match="ignores weights"):
        barbell.centrality("degree", weight_property="score")


def test_weight_property_reads_through_relationship_properties(barbell):
    """`score` lives in the properties dict, not as a top-level edge attribute."""
    weighted = barbell.centrality(
        "pagerank", rel_types=["SEMANTIC_SIMILAR"], weight_property="score"
    )
    unweighted = barbell.centrality("pagerank", rel_types=["SEMANTIC_SIMILAR"])
    assert len(weighted) == len(unweighted) == 7


def test_centrality_on_an_empty_graph():
    db = GrafitoDatabase(':memory:')
    assert db.centrality("pagerank") == []
    db.close()


def test_centrality_accepts_a_prebuilt_graph(barbell):
    graph = barbell.to_analysis_graph(exclude_rel_types=["SEMANTIC_SIMILAR"])
    assert _names(barbell.centrality("pagerank", graph=graph, limit=1))


# --- communities -----------------------------------------------------------


def test_louvain_separates_the_two_clusters(barbell):
    communities = barbell.communities(
        "louvain", exclude_rel_types=["SEMANTIC_SIMILAR"], seed=42
    )
    assert len(communities) == 2
    grouped = [sorted(node.properties["name"] for node in c.nodes) for c in communities]
    assert ["b1", "b2", "b3"] in grouped
    assert sorted(grouped[0]) == grouped[0]
    assert all(isinstance(c, Community) for c in communities)


def test_communities_are_ordered_largest_first(barbell):
    communities = barbell.communities(
        "louvain", exclude_rel_types=["SEMANTIC_SIMILAR"], seed=42
    )
    sizes = [c.size for c in communities]
    assert sizes == sorted(sizes, reverse=True)
    assert communities[0].id == 0


def test_community_ids_and_len(barbell):
    community = barbell.communities("louvain", seed=42)[0]
    assert community.ids() == sorted(node.id for node in community.nodes)
    assert len(community) == community.size


@pytest.mark.parametrize("algorithm", ["louvain", "greedy", "lpa", "label_propagation"])
def test_every_community_algorithm_runs(barbell, algorithm):
    communities = barbell.communities(
        algorithm, exclude_rel_types=["SEMANTIC_SIMILAR"], seed=7
    )
    assert communities
    covered = {node.id for c in communities for node in c.nodes}
    assert len(covered) == 7  # every node lands in exactly one community


def test_seed_makes_louvain_reproducible(barbell):
    first = barbell.communities("louvain", seed=99)
    second = barbell.communities("louvain", seed=99)
    assert [c.ids() for c in first] == [c.ids() for c in second]


def test_min_size_drops_small_communities(barbell):
    all_communities = barbell.communities("louvain", seed=42)
    filtered = barbell.communities("louvain", seed=42, min_size=4)
    assert len(filtered) <= len(all_communities)
    assert all(c.size >= 4 for c in filtered)


def test_unknown_community_algorithm_is_rejected(barbell):
    with pytest.raises(DatabaseError, match="Unknown community algorithm"):
        barbell.communities("spectral")


def test_communities_on_an_empty_graph():
    db = GrafitoDatabase(':memory:')
    assert db.communities() == []
    db.close()


# --- to_analysis_graph -----------------------------------------------------


def test_analysis_graph_filters_relationship_types(barbell):
    graph = barbell.to_analysis_graph(rel_types=["LINK"])
    assert graph.number_of_edges() == 8
    assert all(data["type"] == "LINK" for _, _, data in graph.edges(data=True))


def test_analysis_graph_filters_nodes_by_label(barbell):
    barbell.create_node(labels=["Other"], properties={"name": "outsider"})
    graph = barbell.to_analysis_graph(labels=["N"])
    assert graph.number_of_nodes() == 7


def test_analysis_graph_drops_edges_with_a_filtered_endpoint(barbell):
    outsider = barbell.create_node(labels=["Other"], properties={"name": "outsider"})
    barbell.create_relationship(barbell.node_ids["a1"], outsider.id, "LINK")
    graph = barbell.to_analysis_graph(labels=["N"], rel_types=["LINK"])
    assert graph.number_of_edges() == 8  # the cross-label edge is gone


def test_analysis_graph_promotes_weight_property(barbell):
    graph = barbell.to_analysis_graph(
        rel_types=["SEMANTIC_SIMILAR"], weight_property="score"
    )
    assert all(data["weight"] == 0.9 for _, _, data in graph.edges(data=True))


def test_analysis_graph_defaults_missing_weights_to_one(barbell):
    graph = barbell.to_analysis_graph(rel_types=["LINK"], weight_property="score")
    assert all(data["weight"] == 1.0 for _, _, data in graph.edges(data=True))


def test_analysis_graph_rejects_a_non_numeric_weight(barbell):
    barbell.create_relationship(
        barbell.node_ids["a1"], barbell.node_ids["a2"], "WEIRD", {"score": "high"}
    )
    with pytest.raises(DatabaseError, match="non-numeric"):
        barbell.to_analysis_graph(rel_types=["WEIRD"], weight_property="score")


def test_analysis_graph_rejects_contradictory_filters(barbell):
    with pytest.raises(DatabaseError, match="overlap"):
        barbell.to_analysis_graph(rel_types=["LINK"], exclude_rel_types=["LINK"])


def test_analysis_graph_undirected(barbell):
    assert not barbell.to_analysis_graph(directed=False).is_directed()


# --- pagerank without scipy ------------------------------------------------


def test_pagerank_falls_back_when_scipy_is_missing(barbell, monkeypatch):
    """NetworkX routes pagerank through scipy, which is not a dependency."""
    import networkx as nx

    def _no_scipy(*args, **kwargs):
        raise ImportError("No module named 'scipy'")

    monkeypatch.setattr(nx, "pagerank", _no_scipy)
    results = barbell.centrality("pagerank")
    assert len(results) == 7
    assert abs(sum(row["score"] for row in results) - 1.0) < 1e-6


def test_pagerank_fallback_matches_networkx(barbell):
    """The fallback is only useful if it agrees with the real implementation."""
    pytest.importorskip("scipy")
    import networkx as nx

    from grafito.algorithms import _pagerank

    graph = barbell.to_analysis_graph()
    reference = nx.pagerank(graph)
    fallback = _pagerank(graph)
    assert max(abs(reference[n] - fallback[n]) for n in reference) < 1e-9


def test_pagerank_fallback_handles_dangling_nodes():
    """A node with no outgoing edges leaks rank unless it is redistributed."""
    import networkx as nx

    from grafito.algorithms import _pagerank

    scores = _pagerank(nx.path_graph(5, create_using=nx.DiGraph))
    assert abs(sum(scores.values()) - 1.0) < 1e-9


def test_pagerank_fallback_on_weighted_graph(barbell):
    import networkx as nx

    def _no_scipy(*args, **kwargs):
        raise ImportError("No module named 'scipy'")

    original = nx.pagerank
    nx.pagerank = _no_scipy
    try:
        results = barbell.centrality(
            "pagerank", rel_types=["SEMANTIC_SIMILAR"], weight_property="score"
        )
    finally:
        nx.pagerank = original
    assert abs(sum(row["score"] for row in results) - 1.0) < 1e-6
