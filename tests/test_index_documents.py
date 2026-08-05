"""Bulk ingest of row-shaped documents into nodes, edges, and embeddings."""

import hashlib
import re
from typing import ClassVar

import pytest

from grafito import GrafitoDatabase, IndexReport
from grafito.embedding_functions import EmbeddingFunction
from grafito.exceptions import DatabaseError


class _Embedder(EmbeddingFunction):
    """Counts its calls, so batching can be asserted rather than assumed."""

    calls: ClassVar[list[int]] = []

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        type(self).calls.append(len(input))
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
        return "index_documents_test_embedder"

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


ROWS = [
    {"id": "1", "text": "graph databases store nodes", "year": 2024, "links": ["2", "3"]},
    {
        "id": "2",
        "text": "vector search finds neighbors",
        "year": 2025,
        "links": [{"id": "3", "type": "CITES", "properties": {"weight": 2}}],
    },
    {"id": "3", "text": "retrieval augmented generation", "year": 2025, "links": []},
]


@pytest.fixture
def db() -> GrafitoDatabase:
    database = GrafitoDatabase(':memory:')
    _Embedder.calls = []
    database.create_vector_index("default", dim=32, embedding_function=_Embedder())
    yield database
    database.close()


def test_creates_a_node_per_row(db):
    report = db.index_documents(ROWS, label="Doc")
    assert isinstance(report, IndexReport)
    assert report.nodes_created == 3
    assert report.nodes_updated == 0
    assert report.nodes == 3
    assert len(db.match_nodes(labels=["Doc"])) == 3


def test_copies_every_other_key_as_a_property(db):
    db.index_documents(ROWS, label="Doc", relationships_key="links")
    node = db.match_nodes(labels=["Doc"], properties={"id": "1"})[0]
    assert node.properties == {
        "id": "1",
        "text": "graph databases store nodes",
        "year": 2024,
    }
    # The relationships key describes edges; storing it too would duplicate them.
    assert "links" not in node.properties


def test_copy_attributes_false_keeps_only_id_and_text(db):
    db.index_documents(ROWS, label="Doc", copy_attributes=False)
    node = db.match_nodes(labels=["Doc"], properties={"id": "1"})[0]
    assert set(node.properties) == {"id", "text"}


def test_explicit_property_allowlist_wins(db):
    db.index_documents(ROWS, label="Doc", properties=["id", "year"])
    node = db.match_nodes(labels=["Doc"], properties={"id": "1"})[0]
    assert set(node.properties) == {"id", "year"}


def test_bare_id_references_use_the_default_type(db):
    db.index_documents(ROWS, label="Doc", relationships_key="links")
    rows = db.execute(
        "MATCH (a:Doc {id: '1'})-[r]->(b:Doc) RETURN type(r) AS t, b.id AS b ORDER BY b"
    )
    assert rows == [{"t": "RELATED_TO", "b": "2"}, {"t": "RELATED_TO", "b": "3"}]


def test_mapping_references_carry_type_and_properties(db):
    db.index_documents(ROWS, label="Doc", relationships_key="links")
    rows = db.execute(
        "MATCH (a:Doc {id: '2'})-[r]->(b:Doc) RETURN type(r) AS t, r.weight AS w"
    )
    assert rows == [{"t": "CITES", "w": 2}]


def test_forward_references_resolve(db):
    """Row 1 links to row 3, which has not been created yet when row 1 is read."""
    report = db.index_documents(ROWS, label="Doc", relationships_key="links")
    assert report.relationships_created == 3
    assert report.unresolved == []


def test_custom_default_rel_type(db):
    db.index_documents(
        ROWS, label="Doc", relationships_key="links", default_rel_type="LINKS_TO"
    )
    rows = db.execute("MATCH (:Doc {id: '1'})-[r:LINKS_TO]->() RETURN count(r) AS c")
    assert rows == [{"c": 2}]


def test_unresolvable_targets_are_reported_not_raised(db):
    rows = [{"id": "1", "text": "solo", "links": ["missing"]}]
    report = db.index_documents(rows, label="Doc", relationships_key="links")
    assert report.unresolved == ["missing"]
    assert report.relationships_created == 0


def test_embeddings_are_written_and_searchable(db):
    report = db.index_documents(ROWS, label="Doc")
    assert report.embedded == 3
    hits = db.semantic_search("vector search finds neighbors", k=1)
    assert hits[0]["node"].properties["id"] == "2"


def test_embedding_happens_in_batches(db):
    db.index_documents(ROWS, label="Doc", batch_size=2)
    assert _Embedder.calls == [2, 1]


def test_index_none_skips_embedding(db):
    report = db.index_documents(ROWS, label="Doc", index=None)
    assert report.embedded == 0
    assert _Embedder.calls == []


def test_rows_without_text_are_stored_but_not_embedded(db):
    rows = [{"id": "1", "text": "has text"}, {"id": "2"}, {"id": "3", "text": "   "}]
    report = db.index_documents(rows, label="Doc")
    assert report.nodes_created == 3
    assert report.embedded == 1


def test_upsert_updates_instead_of_duplicating(db):
    db.index_documents(ROWS, label="Doc")
    report = db.index_documents(
        [{"id": "1", "text": "graph databases store nodes and edges", "year": 2026}],
        label="Doc",
    )
    assert report.nodes_created == 0
    assert report.nodes_updated == 1
    assert len(db.match_nodes(labels=["Doc"])) == 3
    node = db.match_nodes(labels=["Doc"], properties={"id": "1"})[0]
    assert node.properties["year"] == 2026


def test_upsert_false_creates_duplicates(db):
    db.index_documents(ROWS, label="Doc")
    report = db.index_documents(ROWS, label="Doc", upsert=False)
    assert report.nodes_created == 3
    assert len(db.match_nodes(labels=["Doc"])) == 6


def test_id_key_none_ignores_identity(db):
    rows = [{"text": "one"}, {"text": "two"}]
    report = db.index_documents(rows, label="Doc", id_key=None)
    assert report.nodes_created == 2
    assert report.ids == {}


def test_report_maps_external_ids_to_node_ids(db):
    report = db.index_documents(ROWS, label="Doc")
    assert set(report.ids) == {"1", "2", "3"}
    assert db.get_node(report.ids["1"]).properties["id"] == "1"


def test_report_str_is_readable(db):
    report = db.index_documents(ROWS, label="Doc", relationships_key="links")
    assert "3 created" in str(report)
    assert "3 relationships" in str(report)


def test_generators_are_accepted(db):
    report = db.index_documents((row for row in ROWS), label="Doc")
    assert report.nodes_created == 3


def test_non_dict_rows_are_rejected(db):
    with pytest.raises(DatabaseError, match="iterable of dicts"):
        db.index_documents(["not a dict"], label="Doc")


def test_invalid_batch_size_is_rejected(db):
    with pytest.raises(DatabaseError, match="batch_size"):
        db.index_documents(ROWS, label="Doc", batch_size=0)


def test_missing_embedding_function_is_reported_clearly():
    database = GrafitoDatabase(':memory:')
    database.create_vector_index("default", dim=4)
    with pytest.raises(DatabaseError, match="no embedding function"):
        database.index_documents(ROWS, label="Doc")
    database.close()


def test_ingested_graph_feeds_the_analysis_apis(db):
    """End to end: ingest rows, then retrieve and rank as a graph."""
    db.index_documents(ROWS, label="Doc", relationships_key="links")
    sub = db.semantic_subgraph("vector search finds neighbors", k=1, expand=1)
    assert len(sub) > 1
    assert db.centrality("pagerank", graph=sub.to_networkx(), limit=1)
