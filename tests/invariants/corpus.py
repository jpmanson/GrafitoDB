"""Synthetic corpora with known cluster structure.

The invariant tests need graphs that vary in shape without varying in what they
mean: a corpus where cluster membership is *known* lets a property be checked
across many parameter combinations, and later lets retrieval quality be scored
against ground truth rather than against a hand-written expectation.

Everything here is deterministic given ``seed``. A failing property must be
reproducible from its parameters alone.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from grafito import GrafitoDatabase


@dataclass
class Corpus:
    """A database of embedded nodes with known cluster membership."""

    db: GrafitoDatabase
    #: Node ids, in creation order.
    node_ids: list[int] = field(default_factory=list)
    #: cluster index of each node, parallel to node_ids — the ground truth.
    clusters: list[int] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    index: str = "default"

    def cluster_of(self, node_id: int) -> int:
        return self.clusters[self.node_ids.index(node_id)]

    def members(self, cluster: int) -> list[int]:
        return [
            node_id
            for node_id, group in zip(self.node_ids, self.clusters)
            if group == cluster
        ]

    def close(self) -> None:
        self.db.close()


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def make_corpus(
    *,
    clusters: int = 3,
    per_cluster: int = 6,
    dim: int = 16,
    spread: float = 0.15,
    seed: int = 0,
    metric: str = "cosine",
    label: str = "Doc",
    index: str = "default",
    store_embeddings: bool = True,
) -> Corpus:
    """Build a database of ``clusters * per_cluster`` embedded nodes.

    Each cluster gets a random unit centroid; members are the centroid plus
    gaussian noise scaled by ``spread``, renormalised. Small ``spread`` gives
    tight, well-separated clusters — the easy case, where a wrong result means a
    real defect rather than an ambiguous corpus.

    Vectors are written directly rather than through an embedding function, so
    the geometry under test is exactly the geometry described here.
    """
    rng = random.Random(seed)
    db = GrafitoDatabase(':memory:')
    db.create_vector_index(
        index,
        dim=dim,
        options={"store_embeddings": store_embeddings, "metric": metric},
    )

    centroids = [_unit([rng.gauss(0.0, 1.0) for _ in range(dim)]) for _ in range(clusters)]

    corpus = Corpus(db=db, index=index)
    for group, centroid in enumerate(centroids):
        for member in range(per_cluster):
            vector = _unit([
                value + rng.gauss(0.0, spread) for value in centroid
            ])
            node = db.create_node(
                labels=[label],
                properties={
                    "id": f"c{group}-n{member}",
                    "cluster": group,
                    "text": f"cluster {group} document {member}",
                },
            )
            corpus.node_ids.append(node.id)
            corpus.clusters.append(group)
            corpus.vectors.append(vector)

    db.upsert_embeddings_batch(corpus.node_ids, corpus.vectors, index=index)
    return corpus


def generated_edges(
    db: GrafitoDatabase,
    rel_type: str = "SEMANTIC_SIMILAR",
    index: str | None = None,
) -> list[dict[str, Any]]:
    """Every edge produced by create_semantic_graph, as plain dicts.

    Reads through SQL rather than Cypher so that a defect in pattern matching
    cannot mask a defect in graph construction.
    """
    sql = """
        SELECT source_node_id AS source, target_node_id AS target, properties
        FROM relationships
        WHERE type = ?
          AND json_extract(properties, '$.generated_by') = 'create_semantic_graph'
    """
    params: list[Any] = [rel_type]
    if index is not None:
        sql += " AND json_extract(properties, '$.index') = ?"
        params.append(index)
    sql += " ORDER BY id"

    import orjson

    return [
        {
            "source": int(row["source"]),
            "target": int(row["target"]),
            **orjson.loads(row["properties"]),
        }
        for row in db.conn.execute(sql, params).fetchall()
    ]


def undirected_pairs(edges: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """Edge endpoints as unordered pairs."""
    return {
        (min(edge["source"], edge["target"]), max(edge["source"], edge["target"]))
        for edge in edges
    }


def neighbours(edges: list[dict[str, Any]]) -> dict[int, set[int]]:
    """Adjacency ignoring direction — how the semantic graph is meant to be read."""
    adjacency: dict[int, set[int]] = {}
    for edge in edges:
        adjacency.setdefault(edge["source"], set()).add(edge["target"])
        adjacency.setdefault(edge["target"], set()).add(edge["source"])
    return adjacency
