"""Properties the semantic graph must hold across parameter combinations.

These differ from the unit tests in `tests/test_semantic_graph.py` in kind, not
just in coverage: each one states an invariant and checks it over a matrix of
`k`, `min_score` and `undirected`, on generated corpora. Every bug fixed in
0.7.1 and 0.7.2 was a violated invariant that no single-fixture test happened to
exercise — a truncated build stranding nodes, a rebuild losing the old graph, a
refresh accumulating duplicates.
"""

from __future__ import annotations

import sqlite3

import pytest

from grafito.exceptions import DatabaseError

from .corpus import (
    generated_edges,
    make_corpus,
    neighbours,
    undirected_pairs,
)

# Kept small: these run on every combination below, and the properties are
# structural — they do not get truer with more nodes.
CLUSTERS = 3
PER_CLUSTER = 5

K_VALUES = [1, 3, 5]
MIN_SCORES = [0.0, 0.5]
DIRECTIONS = [True, False]
SYMMETRIZE = ["union", "mutual", "directed"]


def _corpus(**kwargs):
    return make_corpus(clusters=CLUSTERS, per_cluster=PER_CLUSTER, seed=7, **kwargs)


@pytest.fixture
def corpus():
    built = _corpus()
    yield built
    built.close()


def _build_params():
    return [
        pytest.param(k, min_score, undirected, id=f"k{k}-s{min_score}-{'undir' if undirected else 'dir'}")
        for k in K_VALUES
        for min_score in MIN_SCORES
        for undirected in DIRECTIONS
    ]


# --- structural invariants of a single build -------------------------------


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_every_edge_meets_the_threshold(corpus, k, min_score, undirected):
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    for edge in generated_edges(corpus.db):
        assert edge["score"] >= min_score


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_no_self_edges(corpus, k, min_score, undirected):
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    assert not [e for e in generated_edges(corpus.db) if e["source"] == e["target"]]


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_no_duplicate_edges(corpus, k, min_score, undirected):
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    edges = generated_edges(corpus.db)
    directed_pairs = [(e["source"], e["target"]) for e in edges]
    assert len(directed_pairs) == len(set(directed_pairs))
    if undirected:
        assert len(edges) == len(undirected_pairs(edges))


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_out_degree_never_exceeds_k(corpus, k, min_score, undirected):
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    out_degree: dict[int, int] = {}
    for edge in generated_edges(corpus.db):
        out_degree[edge["source"]] = out_degree.get(edge["source"], 0) + 1
    assert all(count <= k for count in out_degree.values())


@pytest.mark.parametrize("k,undirected", [(k, u) for k in K_VALUES for u in DIRECTIONS])
def test_every_node_reaches_its_k_nearest(corpus, k, undirected):
    """Without truncation, nobody is left short of the neighbours it qualifies for.

    Deduplication removes an edge only when the pair already exists, so the
    neighbour is still reachable — reading the graph undirected, as intended.
    The bar is per node: how many neighbours actually clear `min_score`, which
    on well-separated clusters is fewer than `k` for most of them.
    """
    min_score = 0.0
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    adjacency = neighbours(generated_edges(corpus.db))

    index = corpus.db._get_vector_index(corpus.index)
    for node_id in corpus.node_ids:
        eligible = [
            other
            for other, score in index.search(index.get_vector(node_id), len(corpus.node_ids))
            if other != node_id and score >= min_score
        ]
        expected = min(k, len(eligible))
        assert len(adjacency.get(node_id, set())) >= expected


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_edges_carry_complete_provenance(corpus, k, min_score, undirected):
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    for edge in generated_edges(corpus.db):
        assert edge["index"] == corpus.index
        assert edge["generated_by"] == "create_semantic_graph"
        assert edge["generated_at"]


# --- invariants across repeated builds -------------------------------------


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_rebuild_is_idempotent(corpus, k, min_score, undirected):
    def build():
        corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
        return sorted((e["source"], e["target"]) for e in generated_edges(corpus.db))

    assert build() == build() == build()


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_refresh_converges(corpus, k, min_score, undirected):
    """Repeating an incremental refresh must be a no-op, not an accumulation."""
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    corpus.db.refresh_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    settled = sorted((e["source"], e["target"]) for e in generated_edges(corpus.db))

    for _ in range(3):
        corpus.db.refresh_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    assert sorted((e["source"], e["target"]) for e in generated_edges(corpus.db)) == settled


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_append_does_not_duplicate(corpus, k, min_score, undirected):
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    before = len(generated_edges(corpus.db))
    corpus.db.create_semantic_graph(
        k=k, min_score=min_score, undirected=undirected, replace=False
    )
    assert len(generated_edges(corpus.db)) == before


@pytest.mark.parametrize("undirected", DIRECTIONS)
def test_widening_k_only_adds(corpus, undirected):
    corpus.db.create_semantic_graph(k=1, min_score=0.0, undirected=undirected)
    narrow = undirected_pairs(generated_edges(corpus.db))
    corpus.db.create_semantic_graph(
        k=4, min_score=0.0, undirected=undirected, replace=False
    )
    wide = undirected_pairs(generated_edges(corpus.db))
    assert narrow <= wide
    assert len(wide) > len(narrow)


# --- truncation must not strand nodes --------------------------------------


@pytest.mark.parametrize("max_edges", [1, 3, 7, 12])
def test_refresh_recovers_from_a_truncated_build(corpus, max_edges):
    """The 0.7.2 bug: nodes that only *received* an edge were never searched."""
    truncated = corpus.db.create_semantic_graph(k=3, min_score=0.0, max_edges=max_edges)
    assert truncated.truncated is True

    corpus.db.refresh_semantic_graph(k=3, min_score=0.0)

    adjacency = neighbours(generated_edges(corpus.db))
    for node_id in corpus.node_ids:
        assert adjacency.get(node_id), f"node {node_id} left with no neighbours"


@pytest.mark.parametrize("max_edges", [1, 3, 7, 12])
def test_repeated_truncated_refreshes_reach_the_full_graph(corpus, max_edges):
    """Refreshing under a cap must make progress every time until it is done."""
    corpus.db.create_semantic_graph(k=3, min_score=0.0, max_edges=max_edges)
    for _ in range(20):
        corpus.db.refresh_semantic_graph(k=3, min_score=0.0, max_edges=max_edges)

    complete = make_corpus(clusters=CLUSTERS, per_cluster=PER_CLUSTER, seed=7)
    try:
        complete.db.create_semantic_graph(k=3, min_score=0.0)
        expected = undirected_pairs(generated_edges(complete.db))
    finally:
        complete.close()

    assert undirected_pairs(generated_edges(corpus.db)) == expected


# --- isolation from everything else in the graph ---------------------------


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_hand_made_edges_are_never_touched(corpus, k, min_score, undirected):
    a, b = corpus.node_ids[0], corpus.node_ids[-1]
    corpus.db.create_relationship(a, b, "SEMANTIC_SIMILAR", {"manual": True})

    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    corpus.db.drop_semantic_graph()

    survivors = corpus.db.conn.execute(
        "SELECT COUNT(*) AS c FROM relationships "
        "WHERE type = 'SEMANTIC_SIMILAR' "
        "AND json_extract(properties, '$.manual') = 1"
    ).fetchone()
    assert survivors["c"] == 1


@pytest.mark.parametrize("k", K_VALUES)
def test_indexes_do_not_interfere(corpus, k):
    corpus.db.create_vector_index(
        "other", dim=16, options={"store_embeddings": True, "metric": "cosine"}
    )
    corpus.db.upsert_embeddings_batch(corpus.node_ids, corpus.vectors, index="other")

    corpus.db.create_semantic_graph(index=corpus.index, k=k, min_score=0.0)
    first = undirected_pairs(generated_edges(corpus.db, index=corpus.index))

    corpus.db.create_semantic_graph(index="other", k=k, min_score=0.0)
    corpus.db.create_semantic_graph(index=corpus.index, k=k, min_score=0.0)

    assert undirected_pairs(generated_edges(corpus.db, index=corpus.index)) == first
    assert generated_edges(corpus.db, index="other")

    corpus.db.drop_semantic_graph(index="other")
    assert undirected_pairs(generated_edges(corpus.db, index=corpus.index)) == first
    assert not generated_edges(corpus.db, index="other")


@pytest.mark.parametrize("k,min_score,undirected", _build_params())
def test_a_failed_build_changes_nothing(corpus, k, min_score, undirected):
    """Atomicity, across the parameter matrix rather than one fixture."""
    corpus.db.create_semantic_graph(k=k, min_score=min_score, undirected=undirected)
    before = sorted((e["source"], e["target"]) for e in generated_edges(corpus.db))

    class _FailingInsert:
        def __init__(self, conn):
            self._conn = conn

        def executemany(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        def __getattr__(self, name):
            return getattr(self._conn, name)

    real = corpus.db.conn
    corpus.db.conn = _FailingInsert(real)
    try:
        with pytest.raises(DatabaseError):
            corpus.db.create_semantic_graph(
                k=k, min_score=min_score, undirected=undirected
            )
    finally:
        corpus.db.conn = real

    assert sorted((e["source"], e["target"]) for e in generated_edges(corpus.db)) == before


# --- the semantic graph reflects the embedding geometry --------------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_neighbours_come_from_the_same_cluster(k):
    """Well-separated clusters must not link across, or the graph is noise.

    `k` stays below the cluster size: asking for more neighbours than a cluster
    holds forces cross-cluster edges by construction.
    """
    built = _corpus(spread=0.05)
    try:
        built.db.create_semantic_graph(k=k, min_score=0.0)
        edges = generated_edges(built.db)
        assert edges
        for edge in edges:
            assert built.cluster_of(edge["source"]) == built.cluster_of(edge["target"])
    finally:
        built.close()


def test_communities_recover_the_planted_clusters():
    """End to end: embeddings → semantic graph → community detection."""
    built = _corpus(spread=0.05)
    try:
        built.db.create_semantic_graph(k=3, min_score=0.0)
        communities = built.db.communities(
            "louvain", rel_types=["SEMANTIC_SIMILAR"], weight_property="score", seed=11
        )
        assert len(communities) == CLUSTERS
        for community in communities:
            planted = {built.cluster_of(node.id) for node in community.nodes}
            assert len(planted) == 1, "community mixes planted clusters"
        covered = {node.id for c in communities for node in c.nodes}
        assert covered == set(built.node_ids)
    finally:
        built.close()


@pytest.mark.parametrize("max_edges", [2, 4, 9])
def test_capped_refresh_loop_terminates_at_the_full_graph(corpus, max_edges):
    """Looping on `edges_created` must halt, and halt at the complete graph.

    Not on `nodes_processed`: a node whose every edge was contributed by its
    neighbours has no outgoing edge of its own, so it is re-searched on every
    pass and produces nothing. That count never reaches zero.
    """
    corpus.db.create_semantic_graph(k=3, min_score=0.0, max_edges=max_edges)
    for passes in range(1, 60):
        if not corpus.db.refresh_semantic_graph(
            k=3, min_score=0.0, max_edges=max_edges
        ).edges_created:
            break
    else:
        pytest.fail("capped refresh loop did not terminate")

    complete = make_corpus(clusters=CLUSTERS, per_cluster=PER_CLUSTER, seed=7)
    try:
        complete.db.create_semantic_graph(k=3, min_score=0.0)
        expected = undirected_pairs(generated_edges(complete.db))
    finally:
        complete.close()
    assert undirected_pairs(generated_edges(corpus.db)) == expected


def test_a_settled_graph_creates_no_further_edges(corpus):
    """Re-searching deduplicated nodes is allowed; producing edges is not."""
    corpus.db.create_semantic_graph(k=3, min_score=0.0)
    for _ in range(3):
        assert corpus.db.refresh_semantic_graph(k=3, min_score=0.0).edges_created == 0


# --- symmetrize modes -------------------------------------------------------


@pytest.mark.parametrize("mode", SYMMETRIZE)
@pytest.mark.parametrize("k", K_VALUES)
def test_symmetrize_modes_hold_the_same_structural_invariants(corpus, mode, k):
    corpus.db.create_semantic_graph(k=k, min_score=0.0, symmetrize=mode)
    edges = generated_edges(corpus.db)

    assert not [e for e in edges if e["source"] == e["target"]]
    directed = [(e["source"], e["target"]) for e in edges]
    assert len(directed) == len(set(directed))
    if mode != "directed":
        assert len(edges) == len(undirected_pairs(edges))

    out_degree: dict[int, int] = {}
    for edge in edges:
        out_degree[edge["source"]] = out_degree.get(edge["source"], 0) + 1
    assert all(count <= k for count in out_degree.values())


@pytest.mark.parametrize("k", K_VALUES)
def test_mutual_is_a_subset_of_union(corpus, k):
    """Requiring reciprocity can only remove pairs, never invent them."""
    corpus.db.create_semantic_graph(k=k, min_score=0.0, symmetrize="mutual")
    mutual = undirected_pairs(generated_edges(corpus.db))
    corpus.db.create_semantic_graph(k=k, min_score=0.0, symmetrize="union")
    union = undirected_pairs(generated_edges(corpus.db))
    assert mutual <= union


@pytest.mark.parametrize("k", K_VALUES)
def test_union_is_a_subset_of_directed(corpus, k):
    corpus.db.create_semantic_graph(k=k, min_score=0.0, symmetrize="union")
    union = undirected_pairs(generated_edges(corpus.db))
    corpus.db.create_semantic_graph(k=k, min_score=0.0, symmetrize="directed")
    directed = undirected_pairs(generated_edges(corpus.db))
    assert union <= directed


@pytest.mark.parametrize("mode", SYMMETRIZE)
def test_every_mode_rebuilds_idempotently(corpus, mode):
    def build():
        corpus.db.create_semantic_graph(k=3, min_score=0.0, symmetrize=mode)
        return sorted((e["source"], e["target"]) for e in generated_edges(corpus.db))

    assert build() == build()


@pytest.mark.parametrize("mode", SYMMETRIZE)
def test_every_mode_converges_on_refresh(corpus, mode):
    corpus.db.create_semantic_graph(k=3, min_score=0.0, symmetrize=mode)
    corpus.db.refresh_semantic_graph(k=3, min_score=0.0, symmetrize=mode)
    settled = sorted((e["source"], e["target"]) for e in generated_edges(corpus.db))
    for _ in range(3):
        corpus.db.refresh_semantic_graph(k=3, min_score=0.0, symmetrize=mode)
    assert sorted((e["source"], e["target"]) for e in generated_edges(corpus.db)) == settled


def test_mutual_reduces_cross_cluster_noise():
    """The reason mutual exists, measured against ground truth.

    Overlapping clusters, where the modes actually differ: with tight clusters
    neither produces cross-cluster edges at all.
    """
    built = _corpus(spread=0.35)
    try:
        def cross_fraction(mode: str) -> float:
            built.db.create_semantic_graph(k=3, min_score=0.0, symmetrize=mode)
            edges = generated_edges(built.db)
            crossing = sum(
                1
                for edge in edges
                if built.cluster_of(edge["source"]) != built.cluster_of(edge["target"])
            )
            return crossing / len(edges)

        assert cross_fraction("mutual") < cross_fraction("union")
    finally:
        built.close()
