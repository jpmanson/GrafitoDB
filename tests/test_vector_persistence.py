"""Vectors of a bruteforce index survive reopening a file-backed database.

The bruteforce backend has no on-disk index of its own; without persistence its
vectors live only in memory and a semantic_search over a reopened .db silently
returns nothing. They are persisted in the SQLite vector_entries table by default
for a durable database.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from grafito import GrafitoDatabase


@pytest.fixture
def db_path():
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    handle.close()
    os.unlink(handle.name)  # want a fresh path, not the empty file
    yield handle.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(handle.name + suffix)
        except OSError:
            pass


def test_bruteforce_vectors_survive_reopen(db_path):
    db = GrafitoDatabase(db_path)
    db.create_vector_index("vec", dim=2, backend="bruteforce")
    a = db.create_node(labels=["Chunk"], properties={"t": "left"})
    b = db.create_node(labels=["Chunk"], properties={"t": "right"})
    db.upsert_embeddings_batch([a.id, b.id], [[1.0, 0.0], [0.0, 1.0]], index="vec")
    # in the same process it already works
    assert db.semantic_search([1.0, 0.0], k=1, index="vec")[0]["node"].id == a.id
    db.close()

    # fresh connection, same file: vectors must still be searchable
    reopened = GrafitoDatabase(db_path)
    hits = reopened.semantic_search([0.0, 1.0], k=1, index="vec")
    assert hits, "bruteforce vectors were lost on reopen"
    assert hits[0]["node"].id == b.id
    reopened.close()


def test_bruteforce_defaults_store_embeddings_only_for_file_db(db_path):
    # file-backed: store_embeddings defaulted on
    db = GrafitoDatabase(db_path)
    db.create_vector_index("vec", dim=2, backend="bruteforce")
    assert db.list_vector_indexes()[0]["options"].get("store_embeddings") is True
    db.close()

    # in-memory: no persistence possible, so no overhead defaulted on
    mem = GrafitoDatabase(":memory:")
    mem.create_vector_index("vec", dim=2, backend="bruteforce")
    assert "store_embeddings" not in mem.list_vector_indexes()[0]["options"]
    mem.close()


def test_explicit_store_embeddings_false_is_respected(db_path):
    db = GrafitoDatabase(db_path)
    db.create_vector_index(
        "vec", dim=2, backend="bruteforce", options={"store_embeddings": False}
    )
    assert db.list_vector_indexes()[0]["options"]["store_embeddings"] is False
    db.close()


# --- sidecar index_path is anchored to the db file, not the cwd ----------------


def test_sidecar_index_path_anchored_to_db_dir(tmp_path, monkeypatch):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    db_file = db_dir / "graph.db"
    db = GrafitoDatabase(str(db_file))
    # run from an unrelated directory: the path must NOT follow the cwd
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    opts = db._ensure_vector_index_path("myidx", {}, "faiss")
    path = opts["index_path"]
    assert path == os.path.join(str(db_dir), ".grafito", "indexes", "myidx.faiss.idx")
    assert str(elsewhere) not in path
    db.close()


def test_sidecar_index_path_memory_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = GrafitoDatabase(":memory:")
    opts = db._ensure_vector_index_path("myidx", {}, "annoy")
    assert opts["index_path"] == os.path.join(str(tmp_path), ".grafito", "indexes", "myidx.annoy")
    db.close()


def test_explicit_index_path_is_preserved(tmp_path):
    db = GrafitoDatabase(":memory:")
    custom = str(tmp_path / "custom" / "x.idx")
    opts = db._ensure_vector_index_path("myidx", {"index_path": custom}, "faiss")
    assert opts["index_path"] == custom
    db.close()
