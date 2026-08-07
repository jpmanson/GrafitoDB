"""Induced subgraphs from semantic, lexical, and explicit seeds."""

import hashlib
import re

import pytest

from grafito import GrafitoDatabase, Subgraph
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
        return "subgraph_test_embedder"

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


@pytest.fixture
def chain() -> GrafitoDatabase:
    """A -> B -> C -> D, plus an isolated node and a derived cross-edge."""
    db = GrafitoDatabase(':memory:')
    embedder = _Embedder()
    db.create_vector_index("default", dim=64, embedding_function=embedder)
    db.create_text_index("node", "Doc", ["text"])

    texts = {
        "A": "alpha graph databases",
        "B": "beta vector search",
        "C": "gamma retrieval augmented",
        "D": "delta language models",
        "lonely": "epsilon unrelated cooking",
    }
    ids = {}
    for name, text in texts.items():
        node = db.create_node(labels=["Doc"], properties={"name": name, "text": text})
        ids[name] = node.id
        db.upsert_embedding(node.id, embedder([text])[0])
    for source, target in [("A", "B"), ("B", "C"), ("C", "D")]:
        db.create_relationship(ids[source], ids[target], "NEXT")
    db.create_relationship(ids["A"], ids["D"], "SEMANTIC_SIMILAR", {"score": 0.5})
    db.node_ids = ids
    yield db
    db.close()


def _names(subgraph: Subgraph) -> set[str]:
    return {node.properties["name"] for node in subgraph.nodes}


# --- subgraph() core -------------------------------------------------------


def test_expand_zero_keeps_only_seeds(chain):
    sub = chain.subgraph([chain.node_ids["A"], chain.node_ids["B"]], expand=0)
    assert _names(sub) == {"A", "B"}
    assert {rel.type for rel in sub.relationships} == {"NEXT"}


def test_edges_between_seeds_are_returned_even_without_expansion(chain):
    """The induced graph includes seed-to-seed edges, not just traversed ones."""
    sub = chain.subgraph([chain.node_ids["A"], chain.node_ids["D"]], expand=0)
    assert {rel.type for rel in sub.relationships} == {"SEMANTIC_SIMILAR"}


def test_expansion_records_hop_distance(chain):
    sub = chain.subgraph([chain.node_ids["A"]], expand=2, exclude_rel_types=["SEMANTIC_SIMILAR"])
    hops = {node.properties["name"]: sub.hops[node.id] for node in sub.nodes}
    assert hops == {"A": 0, "B": 1, "C": 2}


def test_direction_out_follows_only_outgoing_edges(chain):
    sub = chain.subgraph([chain.node_ids["C"]], expand=1, direction="out")
    assert _names(sub) == {"C", "D"}


def test_direction_in_follows_only_incoming_edges(chain):
    sub = chain.subgraph([chain.node_ids["C"]], expand=1, direction="in")
    assert _names(sub) == {"C", "B"}


def test_direction_both_is_the_default(chain):
    sub = chain.subgraph([chain.node_ids["C"]], expand=1, exclude_rel_types=["SEMANTIC_SIMILAR"])
    assert _names(sub) == {"B", "C", "D"}


def test_exclude_rel_types_blocks_expansion_through_them(chain):
    without = chain.subgraph(
        [chain.node_ids["A"]], expand=1, exclude_rel_types=["SEMANTIC_SIMILAR"]
    )
    with_all = chain.subgraph([chain.node_ids["A"]], expand=1)
    assert _names(without) == {"A", "B"}
    assert _names(with_all) == {"A", "B", "D"}


def test_rel_types_restricts_expansion(chain):
    sub = chain.subgraph([chain.node_ids["A"]], expand=3, rel_types=["SEMANTIC_SIMILAR"])
    assert _names(sub) == {"A", "D"}


def test_labels_restrict_what_expansion_reaches(chain):
    other = chain.create_node(labels=["Other"], properties={"name": "other"})
    chain.create_relationship(chain.node_ids["A"], other.id, "NEXT")
    sub = chain.subgraph([chain.node_ids["A"]], expand=1, labels=["Doc"])
    assert "other" not in _names(sub)


def test_max_nodes_caps_the_subgraph(chain):
    sub = chain.subgraph([chain.node_ids["A"]], expand=3, max_nodes=2)
    assert len(sub) == 2


def test_include_edges_false_returns_nodes_only(chain):
    sub = chain.subgraph([chain.node_ids["A"]], expand=1, include_edges=False)
    assert sub.relationships == []
    assert len(sub) > 1


def test_seeds_accept_ids_nodes_and_hits(chain):
    node = chain.get_node(chain.node_ids["A"])
    by_id = chain.subgraph([chain.node_ids["A"]], expand=0)
    by_node = chain.subgraph([node], expand=0)
    by_hit = chain.subgraph([{"node": node, "score": 0.9}], expand=0)
    assert _names(by_id) == _names(by_node) == _names(by_hit) == {"A"}
    assert by_hit.scores == {node.id: 0.9}
    assert by_id.scores == {}


def test_duplicate_seeds_are_collapsed(chain):
    sub = chain.subgraph([chain.node_ids["A"], chain.node_ids["A"]], expand=0)
    assert len(sub) == 1


def test_unknown_seed_ids_are_skipped(chain):
    sub = chain.subgraph([chain.node_ids["A"], 999999], expand=0)
    assert _names(sub) == {"A"}


def test_invalid_arguments_are_rejected(chain):
    with pytest.raises(DatabaseError, match="direction must be"):
        chain.subgraph([1], direction="sideways")
    with pytest.raises(DatabaseError, match="expand must be"):
        chain.subgraph([1], expand=-1)
    with pytest.raises(DatabaseError, match="max_nodes must be"):
        chain.subgraph([1], max_nodes=0)
    with pytest.raises(DatabaseError, match="overlap"):
        chain.subgraph([1], rel_types=["NEXT"], exclude_rel_types=["NEXT"])


# --- semantic_subgraph / text_subgraph -------------------------------------


def test_semantic_subgraph_seeds_from_the_vector_index(chain):
    sub = chain.semantic_subgraph("beta vector search", k=1, expand=0)
    assert _names(sub) == {"B"}
    assert sub.seed_ids() == [chain.node_ids["B"]]
    assert sub.scores[chain.node_ids["B"]] > 0.9


def test_semantic_subgraph_expands_and_keeps_provenance(chain):
    sub = chain.semantic_subgraph(
        "beta vector search", k=1, expand=1, exclude_rel_types=["SEMANTIC_SIMILAR"]
    )
    assert _names(sub) == {"A", "B", "C"}
    assert sub.hops[chain.node_ids["B"]] == 0
    assert sub.hops[chain.node_ids["A"]] == 1
    # Only the seed carries a retrieval score.
    assert set(sub.scores) == {chain.node_ids["B"]}


def test_semantic_subgraph_forwards_retrieval_filters(chain):
    sub = chain.semantic_subgraph(
        "beta vector search", k=5, expand=0, filter_props={"name": "A"}
    )
    assert _names(sub) == {"A"}


def test_text_subgraph_seeds_from_fts(chain):
    sub = chain.text_subgraph("gamma", k=2, expand=1, exclude_rel_types=["SEMANTIC_SIMILAR"])
    assert "C" in _names(sub)
    assert sub.hops[chain.node_ids["C"]] == 0


@pytest.mark.parametrize("query", ["gamma?", "gamma, retrieval", "gamma & retrieval"])
def test_text_subgraph_tolerates_natural_language_punctuation(chain, query):
    """FTS5 would reject these outright; seeding retries them as literal terms."""
    sub = chain.text_subgraph(query, k=2, expand=0)
    assert "C" in _names(sub)


def test_text_subgraph_still_raises_when_the_retry_cannot_help(chain):
    """Tolerating punctuation must not turn real failures into empty graphs."""
    # Nothing to quote as a term: the FTS syntax error stands.
    with pytest.raises(DatabaseError, match="Failed to search text index"):
        chain.text_subgraph("?!", k=2)
    # And validation errors survive the retry rather than being swallowed.
    with pytest.raises(DatabaseError, match="k must be"):
        chain.text_subgraph("gamma?", k=0)


def test_empty_search_yields_an_empty_subgraph(chain):
    sub = chain.subgraph([], expand=2)
    assert sub.is_empty()
    assert len(sub) == 0
    assert sub.node_ids() == []


# --- Subgraph.to_networkx --------------------------------------------------


def test_to_networkx_carries_provenance_attributes(chain):
    sub = chain.semantic_subgraph("beta vector search", k=1, expand=1)
    graph = sub.to_networkx()
    seed_attrs = graph.nodes[chain.node_ids["B"]]
    assert seed_attrs["hops"] == 0
    assert seed_attrs["score"] > 0.9
    assert seed_attrs["labels"] == ["Doc"]
    assert graph.nodes[chain.node_ids["A"]]["score"] is None


def test_to_networkx_edges_carry_type_and_properties(chain):
    sub = chain.subgraph([chain.node_ids["A"], chain.node_ids["D"]], expand=0)
    graph = sub.to_networkx()
    _, _, data = next(iter(graph.edges(data=True)))
    assert data["type"] == "SEMANTIC_SIMILAR"
    assert data["properties"]["score"] == 0.5


def test_to_networkx_undirected(chain):
    sub = chain.subgraph([chain.node_ids["A"]], expand=1)
    assert not sub.to_networkx(directed=False).is_directed()


def test_subgraph_feeds_centrality(chain):
    """The headline composition: retrieve a subgraph, then rank within it."""
    sub = chain.semantic_subgraph("alpha graph databases", k=5, expand=1)
    ranked = chain.centrality("pagerank", graph=sub.to_networkx(), limit=1)
    assert ranked and "node" in ranked[0]
