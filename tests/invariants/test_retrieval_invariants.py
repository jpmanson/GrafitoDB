"""Properties of subgraph retrieval and graph analysis.

Where the semantic-graph invariants check how edges are built, these check what
comes back out: that a subgraph is internally consistent, that its provenance
means what it says, and that analysis over planted clusters recovers them.
"""

from __future__ import annotations

import pytest

from grafito.exceptions import DatabaseError

from .corpus import make_corpus

CLUSTERS = 3
PER_CLUSTER = 6


@pytest.fixture
def corpus():
    built = make_corpus(clusters=CLUSTERS, per_cluster=PER_CLUSTER, seed=13, spread=0.05)
    # A chain through every node, so expansion has structure unrelated to
    # similarity — otherwise "expanded" and "similar" would be the same set.
    for left, right in zip(built.node_ids, built.node_ids[1:]):
        built.db.create_relationship(left, right, "NEXT")
    yield built
    built.close()


# --- subgraph consistency ---------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 8])
@pytest.mark.parametrize("expand", [0, 1, 2])
def test_subgraph_is_internally_consistent(corpus, k, expand):
    sub = corpus.db.semantic_subgraph(corpus.vectors[0], k=k, expand=expand)
    present = set(sub.node_ids())

    assert len(present) == len(sub.nodes), "duplicate nodes in subgraph"
    for rel in sub.relationships:
        assert rel.source_id in present
        assert rel.target_id in present
    assert set(sub.hops) == present
    assert set(sub.scores) <= present


@pytest.mark.parametrize("k", [1, 3, 8])
@pytest.mark.parametrize("expand", [0, 1, 2])
def test_seeds_are_hop_zero_and_carry_scores(corpus, k, expand):
    sub = corpus.db.semantic_subgraph(corpus.vectors[0], k=k, expand=expand)
    seed_ids = set(sub.seed_ids())

    assert seed_ids <= set(sub.node_ids())
    assert all(sub.hops[node_id] == 0 for node_id in seed_ids)
    assert set(sub.scores) == seed_ids
    # Everything else came from expansion, and cannot be at distance zero.
    assert all(sub.hops[nid] > 0 for nid in set(sub.node_ids()) - seed_ids)


@pytest.mark.parametrize("expand", [0, 1, 2, 3])
def test_hops_never_exceed_the_expansion_budget(corpus, expand):
    sub = corpus.db.semantic_subgraph(corpus.vectors[0], k=2, expand=expand)
    assert max(sub.hops.values()) <= expand


@pytest.mark.parametrize("expand", [0, 1, 2])
def test_expanding_further_never_shrinks_the_subgraph(corpus, expand):
    smaller = corpus.db.semantic_subgraph(corpus.vectors[0], k=2, expand=expand)
    larger = corpus.db.semantic_subgraph(corpus.vectors[0], k=2, expand=expand + 1)
    assert set(smaller.node_ids()) <= set(larger.node_ids())


@pytest.mark.parametrize("k", [1, 3, 8])
def test_more_seeds_never_shrink_the_subgraph(corpus, k):
    smaller = corpus.db.semantic_subgraph(corpus.vectors[0], k=k, expand=1)
    larger = corpus.db.semantic_subgraph(corpus.vectors[0], k=k + 2, expand=1)
    assert set(smaller.node_ids()) <= set(larger.node_ids())


@pytest.mark.parametrize("max_nodes", [1, 2, 5, 10])
def test_max_nodes_is_respected(corpus, max_nodes):
    sub = corpus.db.semantic_subgraph(
        corpus.vectors[0], k=3, expand=3, max_nodes=max_nodes
    )
    assert len(sub) <= max(max_nodes, len(sub.seeds))


@pytest.mark.parametrize("direction", ["both", "out", "in"])
def test_direction_restricts_expansion(corpus, direction):
    """`out` and `in` must each be a subset of what `both` reaches."""
    middle = corpus.node_ids[len(corpus.node_ids) // 2]
    both = corpus.db.subgraph([middle], expand=2, rel_types=["NEXT"])
    scoped = corpus.db.subgraph(
        [middle], expand=2, direction=direction, rel_types=["NEXT"]
    )
    assert set(scoped.node_ids()) <= set(both.node_ids())


def test_excluded_relationship_types_are_never_traversed(corpus):
    corpus.db.create_semantic_graph(k=3, min_score=0.0)
    without = corpus.db.subgraph(
        [corpus.node_ids[0]], expand=2, exclude_rel_types=["SEMANTIC_SIMILAR"]
    )
    assert not [r for r in without.relationships if r.type == "SEMANTIC_SIMILAR"]


def test_subgraph_to_networkx_matches_the_subgraph(corpus):
    sub = corpus.db.semantic_subgraph(corpus.vectors[0], k=4, expand=1)
    graph = sub.to_networkx()
    assert graph.number_of_nodes() == len(sub.nodes)
    assert graph.number_of_edges() == len(sub.relationships)
    for node in sub.nodes:
        assert graph.nodes[node.id]["hops"] == sub.hops[node.id]


# --- retrieval quality against planted clusters -----------------------------


@pytest.mark.parametrize("cluster", range(CLUSTERS))
def test_seeds_come_from_the_queried_cluster(corpus, cluster):
    """recall@k against ground truth: a query at a centroid retrieves its own."""
    member = corpus.members(cluster)[0]
    vector = corpus.db._get_vector_index(corpus.index).get_vector(member)

    sub = corpus.db.semantic_subgraph(vector, k=PER_CLUSTER, expand=0)
    retrieved = set(sub.seed_ids())
    planted = set(corpus.members(cluster))

    recall = len(retrieved & planted) / len(planted)
    assert recall == 1.0, f"recall@{PER_CLUSTER} was {recall:.2f}"


def test_seed_scores_are_ordered(corpus):
    sub = corpus.db.semantic_subgraph(corpus.vectors[0], k=8, expand=0)
    scores = [hit["score"] for hit in sub.seeds]
    assert scores == sorted(scores, reverse=True)


# --- analysis invariants ----------------------------------------------------


@pytest.mark.parametrize(
    "kind", ["pagerank", "degree", "betweenness", "closeness", "harmonic"]
)
def test_centrality_covers_every_node(corpus, kind):
    results = corpus.db.centrality(kind, directed=False)
    assert {row["node"].id for row in results} == set(corpus.node_ids)
    assert all(isinstance(row["score"], float) for row in results)


@pytest.mark.parametrize("kind", ["pagerank", "degree", "betweenness"])
def test_centrality_is_ordered_and_limitable(corpus, kind):
    full = corpus.db.centrality(kind, directed=False)
    scores = [row["score"] for row in full]
    assert scores == sorted(scores, reverse=True)
    assert [r["node"].id for r in corpus.db.centrality(kind, directed=False, limit=3)] == \
        [r["node"].id for r in full[:3]]


def test_pagerank_is_a_distribution(corpus):
    total = sum(row["score"] for row in corpus.db.centrality("pagerank"))
    assert abs(total - 1.0) < 1e-6


@pytest.mark.parametrize("algorithm", ["louvain", "greedy", "lpa"])
def test_communities_partition_every_node(corpus, algorithm):
    corpus.db.create_semantic_graph(k=3, min_score=0.0)
    communities = corpus.db.communities(
        algorithm, rel_types=["SEMANTIC_SIMILAR"], seed=5
    )
    seen: set[int] = set()
    for community in communities:
        ids = set(community.ids())
        assert not (ids & seen), "node appears in two communities"
        seen |= ids
    assert seen == set(corpus.node_ids)


@pytest.mark.parametrize("algorithm", ["louvain", "greedy"])
def test_communities_recover_planted_clusters(corpus, algorithm):
    """Purity against ground truth, not against a hand-written expectation."""
    corpus.db.create_semantic_graph(k=3, min_score=0.0)
    communities = corpus.db.communities(
        algorithm, rel_types=["SEMANTIC_SIMILAR"], weight_property="score", seed=5
    )
    assert len(communities) == CLUSTERS
    for community in communities:
        planted = {corpus.cluster_of(node.id) for node in community.nodes}
        assert len(planted) == 1


def test_analysis_graph_excludes_what_it_is_told_to(corpus):
    corpus.db.create_semantic_graph(k=3, min_score=0.0)
    everything = corpus.db.to_analysis_graph()
    chain_only = corpus.db.to_analysis_graph(rel_types=["NEXT"])
    assert chain_only.number_of_edges() == len(corpus.node_ids) - 1
    assert everything.number_of_edges() > chain_only.number_of_edges()


def test_centrality_rejects_impossible_combinations(corpus):
    with pytest.raises(DatabaseError):
        corpus.db.centrality("in_degree", directed=False)
    with pytest.raises(DatabaseError):
        corpus.db.centrality("degree", weight_property="score")
