"""Context assembled from the routes between concepts, not from top-k."""

import hashlib
import re
from itertools import pairwise

import pytest

from grafito import GrafitoDatabase
from grafito.embedding_functions import EmbeddingFunction
from grafito.exceptions import DatabaseError


class _Embedder(EmbeddingFunction):
    """Deterministic offline embedder (md5-hashed bag of words)."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        out = []
        for text in input:
            vec = [0.0] * self._dim
            for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
                vec[int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little") % self._dim] += 1
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    @staticmethod
    def name() -> str:
        return "path_context_test_embedder"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "ip"]

    @staticmethod
    def build_from_config(config: dict) -> "_Embedder":
        return _Embedder()

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    @property
    def dimension(self) -> int:
        return self._dim


CHAIN = [
    ("roman", "the roman empire ruled the mediterranean"),
    ("saxons", "saxons settled in britain after rome withdrew"),
    ("vikings", "vikings raided saxon england repeatedly"),
    ("hastings", "the battle of hastings ended saxon rule"),
]


@pytest.fixture
def db() -> GrafitoDatabase:
    """A chain roman -> saxons -> vikings -> hastings, plus an unrelated node
    reachable only through a similarity edge."""
    database = GrafitoDatabase(':memory:')
    embedder = _Embedder()
    database.create_vector_index("default", dim=64, embedding_function=embedder)

    ids = {}
    for key, text in [*CHAIN, ("pasta", "italian pasta recipes with tomato")]:
        node = database.create_node(labels=["Doc"], properties={"id": key, "text": text})
        ids[key] = node.id
        database.upsert_embedding(node.id, embedder([text])[0])
    for left, right in pairwise(CHAIN):
        database.create_relationship(ids[left[0]], ids[right[0]], "LEADS_TO")
    database.create_relationship(
        ids["roman"], ids["pasta"], "SEMANTIC_SIMILAR", {"score": 0.4}
    )
    database.ids = ids
    yield database
    database.close()


def _names(subgraph) -> list[str]:
    return [node.properties["id"] for node in subgraph.nodes]


def _routes(db, subgraph) -> list[list[str]]:
    return [[db.get_node(nid).properties["id"] for nid in route] for route in subgraph.paths]


# --- finding routes ---------------------------------------------------------


def test_returns_the_nodes_between_two_concepts(db):
    """The point: the answer is the material along the way, not the endpoints."""
    sub = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"], k=1, max_hops=3
    )
    assert _names(sub) == ["roman", "saxons", "vikings", "hastings"]
    assert _routes(db, sub) == [["roman", "saxons", "vikings", "hastings"]]


def test_hops_measure_position_along_the_route(db):
    sub = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"], k=1, max_hops=3
    )
    positions = {node.properties["id"]: sub.hops[node.id] for node in sub.nodes}
    assert positions == {"roman": 0, "saxons": 1, "vikings": 2, "hastings": 3}


def test_only_waypoints_a_route_used_are_seeds(db):
    sub = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"], k=1, max_hops=3
    )
    assert sorted(hit["node"].properties["id"] for hit in sub.seeds) == [
        "hastings",
        "roman",
    ]
    assert set(sub.scores) == set(sub.seed_ids())


def test_unreachable_within_max_hops_returns_empty(db):
    """An empty answer is the honest one; top-k would have invented results."""
    sub = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"], k=1, max_hops=1
    )
    assert sub.is_empty()
    assert sub.paths == []


def test_raising_max_hops_finds_the_route(db):
    for hops, found in [(1, False), (2, False), (3, True)]:
        sub = db.path_context(
            ["roman empire mediterranean", "battle of hastings saxon"],
            k=1,
            max_hops=hops,
        )
        assert bool(sub.paths) is found


def test_more_than_two_waypoints(db):
    sub = db.path_context(
        ["roman empire", "vikings raided", "battle hastings"], k=1, max_hops=2
    )
    assert len(sub.paths) == 2
    assert set(_names(sub)) >= {"roman", "vikings", "hastings"}


def test_relationships_among_selected_nodes_are_returned(db):
    sub = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"], k=1, max_hops=3
    )
    assert {rel.type for rel in sub.relationships} == {"LEADS_TO"}
    present = set(sub.node_ids())
    assert all(
        rel.source_id in present and rel.target_id in present
        for rel in sub.relationships
    )


# --- traversal filters ------------------------------------------------------


def test_excluding_similarity_edges_removes_the_shortcut(db):
    """Similarity edges connect everything; a route through one means little."""
    through = db.path_context(
        ["roman empire mediterranean", "italian pasta tomato"], k=1, max_hops=2
    )
    assert "pasta" in _names(through)

    without = db.path_context(
        ["roman empire mediterranean", "italian pasta tomato"],
        k=1,
        max_hops=2,
        exclude_rel_types=["SEMANTIC_SIMILAR"],
    )
    assert without.is_empty()


def test_rel_types_restricts_travel(db):
    sub = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"],
        k=1,
        max_hops=3,
        rel_types=["LEADS_TO"],
    )
    assert _routes(db, sub) == [["roman", "saxons", "vikings", "hastings"]]


def test_direction_out_follows_the_chain_forward(db):
    forward = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"],
        k=1,
        max_hops=3,
        direction="out",
        rel_types=["LEADS_TO"],
    )
    backward = db.path_context(
        ["battle of hastings saxon", "roman empire mediterranean"],
        k=1,
        max_hops=3,
        direction="out",
        rel_types=["LEADS_TO"],
    )
    assert forward.paths
    assert backward.is_empty()


def test_max_paths_caps_the_search(db):
    sub = db.path_context(
        ["roman empire", "battle hastings"], k=3, max_hops=3, max_paths=1
    )
    assert len(sub.paths) <= 1


def test_expand_adds_a_neighbourhood(db):
    tight = db.path_context(
        ["roman empire mediterranean", "vikings raided"], k=1, max_hops=2
    )
    wide = db.path_context(
        ["roman empire mediterranean", "vikings raided"], k=1, max_hops=2, expand=1
    )
    assert set(_names(tight)) < set(_names(wide))


def test_include_edges_false_returns_nodes_only(db):
    sub = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"],
        k=1,
        max_hops=3,
        include_edges=False,
    )
    assert sub.nodes
    assert sub.relationships == []


# --- composition ------------------------------------------------------------


def test_result_is_a_subgraph(db):
    sub = db.path_context(
        ["roman empire mediterranean", "battle of hastings saxon"], k=1, max_hops=3
    )
    graph = sub.to_networkx()
    assert graph.number_of_nodes() == len(sub.nodes)
    assert db.centrality("pagerank", graph=graph, limit=1)


def test_accepts_vectors_as_waypoints(db):
    embedder = _Embedder()
    sub = db.path_context(
        [
            embedder(["the roman empire ruled the mediterranean"])[0],
            embedder(["the battle of hastings ended saxon rule"])[0],
        ],
        k=1,
        max_hops=3,
    )
    assert _names(sub) == ["roman", "saxons", "vikings", "hastings"]


# --- validation -------------------------------------------------------------


def test_fewer_than_two_waypoints_is_rejected(db):
    with pytest.raises(DatabaseError, match="at least two waypoints"):
        db.path_context(["only one"])
    with pytest.raises(DatabaseError, match="at least two waypoints"):
        db.path_context("not a list")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"k": 0}, "k must be"),
        ({"max_hops": 0}, "max_hops must be"),
        ({"max_paths": 0}, "max_paths must be"),
    ],
)
def test_invalid_arguments_are_rejected(db, kwargs, message):
    with pytest.raises(DatabaseError, match=message):
        db.path_context(["a", "b"], **kwargs)
