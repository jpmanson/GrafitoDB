"""Tests for vector index API."""

import os
import sys
import tempfile

import pytest

from grafito import GrafitoDatabase
from grafito.exceptions import DatabaseError

# Annoy has segfault issues with Python 3.13+
SKIP_ANNOY_PY313 = sys.version_info >= (3, 13)

from grafito.indexers import USearchIndexer, VoyagerIndexer


def test_create_list_drop_vector_index():
    db = GrafitoDatabase(':memory:')
    name = db.create_vector_index("people_vec", dim=2)
    indexes = db.list_vector_indexes()
    assert any(idx["name"] == name and idx["dim"] == 2 for idx in indexes)
    db.drop_vector_index(name)
    assert all(idx["name"] != name for idx in db.list_vector_indexes())
    db.close()


def test_create_vector_index_duplicate():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2)
    with pytest.raises(DatabaseError):
        db.create_vector_index("people_vec", dim=2)
    db.close()


def test_vector_search_basic():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2)
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="people_vec")

    results = db.semantic_search([1.0, 0.0], k=1, index="people_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_with_label_filter():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2)
    person = db.create_node(labels=["Person"], properties={"name": "Alice"})
    company = db.create_node(labels=["Company"], properties={"name": "Acme"})
    db.upsert_embedding(person.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(company.id, [0.9, 0.0], index="people_vec")

    results = db.semantic_search([1.0, 0.0], k=5, index="people_vec", filter_labels=["Person"])
    assert len(results) == 1
    assert results[0]["node"].id == person.id
    db.close()


def test_vector_embeddings_persisted():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle:
        db_path = handle.name

    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
        person = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(person.id, [1.0, 0.0], index="people_vec")
        db.close()

        db = GrafitoDatabase(db_path)
        results = db.semantic_search([1.0, 0.0], k=1, index="people_vec")
        assert len(results) == 1
        assert results[0]["node"].id == person.id
        db.close()
    finally:
        os.unlink(db_path)


def test_bruteforce_save_load_roundtrip():
    from grafito.vector_index.bruteforce import BruteForceIndex

    with tempfile.NamedTemporaryFile(delete=False, suffix=".idx") as handle:
        index_path = handle.name
    try:
        idx = BruteForceIndex(dim=2, method="flat", options={"metric": "cosine"})
        idx.add([1, 2], [[1.0, 0.0], [0.0, 1.0]])
        idx.save(index_path)

        loaded = BruteForceIndex(dim=2, method="flat", options={"metric": "cosine"})
        loaded.load(index_path)
        assert loaded.get_vector(1) == [1.0, 0.0]
        assert loaded.get_vector(2) == [0.0, 1.0]
        assert loaded.search([1.0, 0.0], k=1)[0][0] == 1
    finally:
        os.unlink(index_path)


def test_vector_search_bruteforce_persisted():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle_db:
        db_path = handle_db.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bruteforce") as handle_idx:
        index_path = handle_idx.name
    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index(
            "bruteforce_vec",
            dim=2,
            backend="bruteforce",
            options={"index_path": index_path},
        )
        node = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(node.id, [1.0, 0.0], index="bruteforce_vec")
        db.close()

        db = GrafitoDatabase(db_path)
        results = db.semantic_search([1.0, 0.0], k=1, index="bruteforce_vec")
        assert len(results) == 1
        assert results[0]["node"].id == node.id
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(index_path)


def test_vector_search_faiss_backend():
    faiss = pytest.importorskip("faiss")
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("faiss_vec", dim=2, backend="faiss", method="flat")
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="faiss_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="faiss_vec")

    results = db.semantic_search([1.0, 0.0], k=1, index="faiss_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_faiss_persisted():
    faiss = pytest.importorskip("faiss")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle_db:
        db_path = handle_db.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".idx") as handle_idx:
        index_path = handle_idx.name
    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index(
            "faiss_vec",
            dim=2,
            backend="faiss",
            method="flat",
            options={"index_path": index_path},
        )
        node = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(node.id, [1.0, 0.0], index="faiss_vec")
        db.close()

        db = GrafitoDatabase(db_path)
        results = db.semantic_search([1.0, 0.0], k=1, index="faiss_vec")
        assert len(results) == 1
        assert results[0]["node"].id == node.id
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(index_path)


@pytest.mark.skipif(SKIP_ANNOY_PY313, reason="Annoy has segfault issues with Python 3.13+")
def test_vector_search_annoy_backend():
    annoy = pytest.importorskip("annoy")
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("annoy_vec", dim=2, backend="annoy", method="annoy")
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="annoy_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="annoy_vec")

    results = db.semantic_search([1.0, 0.0], k=1, index="annoy_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


@pytest.mark.skipif(SKIP_ANNOY_PY313, reason="Annoy has segfault issues with Python 3.13+")
def test_vector_search_annoy_persisted():
    annoy = pytest.importorskip("annoy")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle_db:
        db_path = handle_db.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".annoy") as handle_idx:
        index_path = handle_idx.name
    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index(
            "annoy_vec",
            dim=2,
            backend="annoy",
            method="annoy",
            options={"index_path": index_path},
        )
        node = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(node.id, [1.0, 0.0], index="annoy_vec")
        db.close()

        db = GrafitoDatabase(db_path)
        results = db.semantic_search([1.0, 0.0], k=1, index="annoy_vec")
        assert len(results) == 1
        assert results[0]["node"].id == node.id
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(index_path)


@pytest.mark.skipif(SKIP_ANNOY_PY313, reason="Annoy has segfault issues with Python 3.13+")
def test_vector_search_annoy_invalid_options():
    annoy = pytest.importorskip("annoy")
    db = GrafitoDatabase(':memory:')
    with pytest.raises(ValueError, match="n_trees must be an integer"):
        db.create_vector_index(
            "annoy_bad",
            dim=2,
            backend="annoy",
            method="annoy",
            options={"n_trees": "bad"},
        )
    with pytest.raises(ValueError, match="search_k must be an integer"):
        db.create_vector_index(
            "annoy_bad2",
            dim=2,
            backend="annoy",
            method="annoy",
            options={"search_k": "bad"},
        )
    db.close()


def test_vector_search_hnswlib_backend():
    hnswlib = pytest.importorskip("hnswlib")
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("hnsw_vec", dim=2, backend="hnswlib", method="hnsw")
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="hnsw_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="hnsw_vec")

    results = db.semantic_search([1.0, 0.0], k=1, index="hnsw_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_hnswlib_persisted():
    hnswlib = pytest.importorskip("hnswlib")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle_db:
        db_path = handle_db.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hnswlib") as handle_idx:
        index_path = handle_idx.name
    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index(
            "hnsw_vec",
            dim=2,
            backend="hnswlib",
            method="hnsw",
            options={"index_path": index_path},
        )
        node = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(node.id, [1.0, 0.0], index="hnsw_vec")
        db.close()

        db = GrafitoDatabase(db_path)
        results = db.semantic_search([1.0, 0.0], k=1, index="hnsw_vec")
        assert len(results) == 1
        assert results[0]["node"].id == node.id
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(index_path)


def test_vector_search_leann_backend():
    leann = pytest.importorskip("leann")
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("leann_vec", dim=2, backend="leann", method="leann")
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="leann_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="leann_vec")

    results = db.semantic_search([1.0, 0.0], k=1, index="leann_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_leann_persisted():
    leann = pytest.importorskip("leann")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle_db:
        db_path = handle_db.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".leann") as handle_idx:
        index_path = handle_idx.name
    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index(
            "leann_vec",
            dim=2,
            backend="leann",
            method="leann",
            options={"index_path": index_path},
        )
        node = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(node.id, [1.0, 0.0], index="leann_vec")
        db.close()

        db = GrafitoDatabase(db_path)
        results = db.semantic_search([1.0, 0.0], k=1, index="leann_vec")
        assert len(results) == 1
        assert results[0]["node"].id == node.id
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(index_path)


def test_vector_search_leann_invalid_options():
    leann = pytest.importorskip("leann")
    db = GrafitoDatabase(':memory:')
    with pytest.raises(ValueError, match="n_trees must be an integer"):
        db.create_vector_index(
            "leann_bad",
            dim=2,
            backend="leann",
            method="leann",
            options={"n_trees": "bad"},
        )
    with pytest.raises(ValueError, match="search_k must be an integer"):
        db.create_vector_index(
            "leann_bad2",
            dim=2,
            backend="leann",
            method="leann",
            options={"search_k": "bad"},
        )
    db.close()


def test_vector_search_leann_auto_build_off_requires_rebuild():
    leann = pytest.importorskip("leann")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle_db:
        db_path = handle_db.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".leann") as handle_idx:
        index_path = handle_idx.name
    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index(
            "leann_vec",
            dim=2,
            backend="leann",
            method="leann",
            options={"index_path": index_path, "auto_build": False},
        )
        node = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(node.id, [1.0, 0.0], index="leann_vec")
        with pytest.raises(ValueError, match="LEANN index is dirty"):
            db.semantic_search([1.0, 0.0], k=1, index="leann_vec")
        db.rebuild_vector_index("leann_vec")
        results = db.semantic_search([1.0, 0.0], k=1, index="leann_vec")
        assert len(results) == 1
        assert results[0]["node"].id == node.id
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(index_path)


def test_vector_search_usearch_backend():
    usearch = pytest.importorskip("usearch")
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("usearch_vec", dim=2, backend="usearch", method="hnsw")
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="usearch_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="usearch_vec")

    results = db.semantic_search([1.0, 0.0], k=1, index="usearch_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_usearch_persisted():
    usearch = pytest.importorskip("usearch")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle_db:
        db_path = handle_db.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".usearch") as handle_idx:
        index_path = handle_idx.name
    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index(
            "usearch_vec",
            dim=2,
            backend="usearch",
            method="hnsw",
            options={"index_path": index_path},
        )
        node = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(node.id, [1.0, 0.0], index="usearch_vec")
        db.close()

        db = GrafitoDatabase(db_path)
        results = db.semantic_search([1.0, 0.0], k=1, index="usearch_vec")
        assert len(results) == 1
        assert results[0]["node"].id == node.id
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(index_path)


def test_vector_search_usearch_invalid_options():
    usearch = pytest.importorskip("usearch")
    db = GrafitoDatabase(':memory:')
    with pytest.raises(ValueError, match="connectivity must be a positive integer"):
        db.create_vector_index(
            "usearch_bad",
            dim=2,
            backend="usearch",
            method="hnsw",
            options={"connectivity": -1},
        )
    with pytest.raises(ValueError, match="expansion_add must be a positive integer"):
        db.create_vector_index(
            "usearch_bad2",
            dim=2,
            backend="usearch",
            method="hnsw",
            options={"expansion_add": 0},
        )
    db.close()


def test_vector_search_usearch_metrics():
    usearch = pytest.importorskip("usearch")
    db = GrafitoDatabase(':memory:')
    # Test cosine metric
    db.create_vector_index("usearch_cos", dim=2, backend="usearch", options={"metric": "cosine"})
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="usearch_cos")
    results = db.semantic_search([1.0, 0.0], k=1, index="usearch_cos")
    assert len(results) == 1
    
    # Test L2 metric
    db.create_vector_index("usearch_l2", dim=2, backend="usearch", options={"metric": "l2"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n2.id, [0.0, 1.0], index="usearch_l2")
    results = db.semantic_search([0.0, 1.0], k=1, index="usearch_l2")
    assert len(results) == 1
    
    # Test invalid metric
    with pytest.raises(ValueError, match="metric must be one of"):
        db.create_vector_index("usearch_bad", dim=2, backend="usearch", options={"metric": "invalid"})
    db.close()


def test_vector_search_voyager_backend():
    voyager = pytest.importorskip("voyager")
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("voyager_vec", dim=2, backend="voyager", method="hnsw")
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="voyager_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="voyager_vec")

    results = db.semantic_search([1.0, 0.0], k=1, index="voyager_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_voyager_persisted():
    voyager = pytest.importorskip("voyager")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle_db:
        db_path = handle_db.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".voyager") as handle_idx:
        index_path = handle_idx.name
    try:
        db = GrafitoDatabase(db_path)
        db.create_vector_index(
            "voyager_vec",
            dim=2,
            backend="voyager",
            method="hnsw",
            options={"index_path": index_path},
        )
        node = db.create_node(labels=["Person"], properties={"name": "Alice"})
        db.upsert_embedding(node.id, [1.0, 0.0], index="voyager_vec")
        db.close()

        db = GrafitoDatabase(db_path)
        results = db.semantic_search([1.0, 0.0], k=1, index="voyager_vec")
        assert len(results) == 1
        assert results[0]["node"].id == node.id
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(index_path)


def test_vector_search_voyager_invalid_options():
    voyager = pytest.importorskip("voyager")
    db = GrafitoDatabase(':memory:')
    with pytest.raises(ValueError, match="M must be a positive integer"):
        db.create_vector_index(
            "voyager_bad",
            dim=2,
            backend="voyager",
            method="hnsw",
            options={"M": -1},
        )
    with pytest.raises(ValueError, match="ef_construction must be a positive integer"):
        db.create_vector_index(
            "voyager_bad2",
            dim=2,
            backend="voyager",
            method="hnsw",
            options={"ef_construction": 0},
        )
    db.close()


def test_vector_search_voyager_metrics():
    voyager = pytest.importorskip("voyager")
    db = GrafitoDatabase(':memory:')
    # Test cosine metric
    db.create_vector_index("voyager_cos", dim=2, backend="voyager", options={"metric": "cosine"})
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="voyager_cos")
    results = db.semantic_search([1.0, 0.0], k=1, index="voyager_cos")
    assert len(results) == 1
    
    # Test euclidean metric
    db.create_vector_index("voyager_l2", dim=2, backend="voyager", options={"metric": "euclidean"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n2.id, [0.0, 1.0], index="voyager_l2")
    results = db.semantic_search([0.0, 1.0], k=1, index="voyager_l2")
    assert len(results) == 1
    
    # Test inner product metric
    db.create_vector_index("voyager_ip", dim=2, backend="voyager", options={"metric": "ip"})
    n3 = db.create_node(labels=["Person"], properties={"name": "Charlie"})
    db.upsert_embedding(n3.id, [1.0, 1.0], index="voyager_ip")
    results = db.semantic_search([1.0, 1.0], k=1, index="voyager_ip")
    assert len(results) == 1
    
    # Test invalid metric
    with pytest.raises(ValueError, match="metric must be one of"):
        db.create_vector_index("voyager_bad", dim=2, backend="voyager", options={"metric": "invalid"})
    db.close()


def test_vector_search_rerank_with_embeddings():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    a = db.create_node(labels=["Person"], properties={"name": "Alice"})
    b = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(a.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(b.id, [0.0, 1.0], index="people_vec")

    results = db.semantic_search(
        [1.0, 0.0],
        k=2,
        index="people_vec",
        rerank=True
    )
    assert [row["node"].id for row in results] == [a.id, b.id]
    db.close()


def test_vector_search_default_k_from_index():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"default_k": 1})
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="people_vec")

    results = db.semantic_search([1.0, 0.0], index="people_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_default_k_from_db():
    db = GrafitoDatabase(':memory:', default_top_k=1)
    db.create_vector_index("people_vec", dim=2)
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    n2 = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(n2.id, [0.0, 1.0], index="people_vec")

    results = db.semantic_search([1.0, 0.0], index="people_vec")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_custom_reranker_by_name():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(bob.id, [1.0, 0.0], index="people_vec")

    def reverse_reranker(query_vector, candidates):
        return [{"id": item["id"], "score": 1.0} for item in reversed(candidates)]

    db.register_reranker("reverse", reverse_reranker)
    results = db.semantic_search([1.0, 0.0], k=2, index="people_vec", reranker="reverse")
    assert [row["node"].id for row in results] == [bob.id, alice.id]
    db.close()


def test_vector_search_unknown_reranker_raises():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2)
    with pytest.raises(DatabaseError, match="Unknown reranker"):
        db.semantic_search([1.0, 0.0], k=1, index="people_vec", reranker="missing")
    db.close()


def test_vector_search_faiss_ivf_and_hnsw():
    faiss = pytest.importorskip("faiss")
    db = GrafitoDatabase(':memory:')
    db.create_vector_index(
        "faiss_ivf",
        dim=2,
        backend="faiss",
        method="ivf_flat",
        options={"nlist": 2, "nprobe": 1},
    )
    db.create_vector_index(
        "faiss_hnsw",
        dim=2,
        backend="faiss",
        method="hnsw",
        options={"hnsw_m": 8, "ef_search": 16},
    )
    nodes = [
        db.create_node(labels=["Person"], properties={"name": "A"}),
        db.create_node(labels=["Person"], properties={"name": "B"}),
        db.create_node(labels=["Person"], properties={"name": "C"}),
        db.create_node(labels=["Person"], properties={"name": "D"}),
    ]
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    for node, vector in zip(nodes, vectors):
        db.upsert_embedding(node.id, vector, index="faiss_ivf")
        db.upsert_embedding(node.id, vector, index="faiss_hnsw")

    results_ivf = db.semantic_search([1.0, 0.0], k=2, index="faiss_ivf")
    results_hnsw = db.semantic_search([1.0, 0.0], k=2, index="faiss_hnsw")
    assert results_ivf
    assert results_hnsw
    db.close()

def test_vector_search_usearch_indexer_class():
    usearch = pytest.importorskip("usearch")
    db = GrafitoDatabase(':memory:')
    indexer = USearchIndexer(dim=2, method="hnsw", options={"metric": "l2"})
    db.create_vector_index("usearch_idx", indexer=indexer)
    
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="usearch_idx")
    
    results = db.semantic_search([1.0, 0.0], k=1, index="usearch_idx")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()


def test_vector_search_voyager_indexer_class():
    voyager = pytest.importorskip("voyager")
    db = GrafitoDatabase(':memory:')
    indexer = VoyagerIndexer(dim=2, method="hnsw", options={"M": 12})
    db.create_vector_index("voyager_idx", indexer=indexer)
    
    n1 = db.create_node(labels=["Person"], properties={"name": "Alice"})
    db.upsert_embedding(n1.id, [1.0, 0.0], index="voyager_idx")
    
    results = db.semantic_search([1.0, 0.0], k=1, index="voyager_idx")
    assert len(results) == 1
    assert results[0]["node"].id == n1.id
    db.close()
