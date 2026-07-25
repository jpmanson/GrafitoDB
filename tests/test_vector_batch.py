"""Tests for batch vector upsert/remove and commit hygiene."""

import os
import tempfile

import pytest

from grafito import GrafitoDatabase
from grafito.exceptions import NodeNotFoundError


def test_upsert_embeddings_batch_single_persist(monkeypatch):
    db = GrafitoDatabase(":memory:")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".idx") as handle:
        path = handle.name
    try:
        db.create_vector_index(
            "batch_vec",
            dim=2,
            backend="bruteforce",
            options={"index_path": path, "metric": "cosine", "store_embeddings": True},
        )
        nodes = [
            db.create_node(labels=["Chunk"], properties={"i": i}) for i in range(5)
        ]
        persist_calls = {"n": 0}
        real = db._persist_vector_index

        def counting(name, vec_index):
            persist_calls["n"] += 1
            return real(name, vec_index)

        monkeypatch.setattr(db, "_persist_vector_index", counting)
        ids = [n.id for n in nodes]
        vectors = [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(5)]
        db.upsert_embeddings_batch(ids, vectors, index="batch_vec")
        assert persist_calls["n"] == 1

        results = db.semantic_search([1.0, 0.0], k=1, index="batch_vec")
        assert results[0]["node"].id == nodes[0].id
    finally:
        db.close()
        if os.path.exists(path):
            os.unlink(path)


def test_upsert_embeddings_documents_uses_batch(monkeypatch):
    from grafito.embedding_functions.base import EmbeddingFunction

    class Toy(EmbeddingFunction):
        def __call__(self, input):
            return [[float(len(t)), 0.0] for t in input]

        @staticmethod
        def name():
            return "toy_batch"

        def default_space(self):
            return "l2"

        def supported_spaces(self):
            return ["l2"]

        @staticmethod
        def build_from_config(config):
            return Toy()

        def get_config(self):
            return {}

        @staticmethod
        def validate_config(config):
            return None

        @property
        def dimension(self):
            return 2

    db = GrafitoDatabase(":memory:")
    emb = Toy()
    db.create_vector_index("docs", dim=2, embedding_function=emb)
    nodes = [db.create_node(labels=["X"], properties={"t": f"hi{i}"}) for i in range(3)]
    called = {"batch": 0}
    real = db.upsert_embeddings_batch

    def wrap(ids, vectors, index="default"):
        called["batch"] += 1
        return real(ids, vectors, index=index)

    monkeypatch.setattr(db, "upsert_embeddings_batch", wrap)
    db.upsert_embeddings([n.id for n in nodes], ["a", "bb", "ccc"], index="docs")
    assert called["batch"] == 1
    db.close()


def test_remove_embeddings_batch():
    db = GrafitoDatabase(":memory:")
    db.create_vector_index("rm", dim=2, options={"store_embeddings": True})
    n1 = db.create_node(labels=["X"], properties={})
    n2 = db.create_node(labels=["X"], properties={})
    db.upsert_embeddings_batch([n1.id, n2.id], [[1.0, 0.0], [0.0, 1.0]], index="rm")
    db.remove_embeddings_batch([n1.id, n2.id], index="rm")
    idx = db._get_vector_index("rm")
    assert idx.get_vector(n1.id) is None
    assert idx.get_vector(n2.id) is None
    db.close()


def test_upsert_embeddings_batch_empty():
    db = GrafitoDatabase(":memory:")
    db.create_vector_index("e", dim=2)
    db.upsert_embeddings_batch([], [], index="e")
    db.close()


def test_upsert_embeddings_batch_missing_node():
    db = GrafitoDatabase(":memory:")
    db.create_vector_index("e", dim=2)
    with pytest.raises(NodeNotFoundError):
        db.upsert_embeddings_batch([999], [[1.0, 0.0]], index="e")
    db.close()


def test_upsert_respects_transaction_no_premature_commit():
    db = GrafitoDatabase(":memory:")
    db.create_vector_index("tx", dim=2, options={"store_embeddings": True})
    n1 = db.create_node(labels=["X"], properties={"k": 1})
    db.begin_transaction()
    n2 = db.create_node(labels=["X"], properties={"k": 2})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="tx")
    db.rollback()
    # n2 should not survive if commit was properly deferred
    assert db.get_node(n2.id) is None
    db.close()


def test_delete_node_removes_bruteforce_vector():
    db = GrafitoDatabase(":memory:")
    db.create_vector_index("del", dim=2)
    n = db.create_node(labels=["X"], properties={})
    db.upsert_embedding(n.id, [1.0, 0.0], index="del")
    assert db._get_vector_index("del").get_vector(n.id) is not None
    assert db.delete_node(n.id) is True
    assert db._get_vector_index("del").get_vector(n.id) is None
    db.close()
