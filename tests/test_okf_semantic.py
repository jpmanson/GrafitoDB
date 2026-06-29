"""Semantic-search integration for OKF imports.

Uses a tiny deterministic embedding function (hashed bag-of-words) so the test
needs no model downloads or optional dependencies.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.embedding_functions import EmbeddingFunction

pytest.importorskip("yaml")

BUNDLE = Path("examples") / "okf" / "okf_bundle"


class HashingEmbeddingFunction(EmbeddingFunction):
    """Deterministic bag-of-words embedder: tokens hashed into ``dim`` buckets."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []
        for text in input:
            vec = [0.0] * self._dim
            for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
                # Stable hash: Python's built-in hash() is randomized per process
                # (PYTHONHASHSEED), which would make the embedding non-deterministic.
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "little") % self._dim
                vec[bucket] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "hashing_test"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "ip"]

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(dim=config.get("dim", 64))

    def get_config(self) -> dict:
        return {"dim": self._dim}

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def db() -> GrafitoDatabase:
    database = GrafitoDatabase(":memory:")
    yield database
    database.close()


def test_import_without_embed_creates_no_vector_index(db):
    summary = db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    assert summary["embedded"] == 0
    assert db.list_vector_indexes() == []


def test_import_with_embed_populates_vector_index(db):
    summary = db.import_okf_bundle(
        str(BUNDLE), configure_fts=False, embed=HashingEmbeddingFunction()
    )
    assert summary["embedded"] == 3
    names = [idx["name"] for idx in db.list_vector_indexes()]
    assert "okf" in names


def test_semantic_search_over_imported_concepts(db):
    db.import_okf_bundle(
        str(BUNDLE), configure_fts=False, embed=HashingEmbeddingFunction()
    )
    # "email" and "signed up" only appear in the Customers concept body.
    results = db.semantic_search("customer email signed up", index="okf", k=3)
    assert results
    assert results[0]["node"].properties.get("title") == "Customers"


def test_custom_embed_index_name(db):
    db.import_okf_bundle(
        str(BUNDLE),
        configure_fts=False,
        embed=HashingEmbeddingFunction(),
        embed_index="knowledge_vec",
    )
    names = [idx["name"] for idx in db.list_vector_indexes()]
    assert "knowledge_vec" in names
    results = db.semantic_search("orders total usd", index="knowledge_vec", k=1)
    assert results[0]["node"].properties.get("title") == "Orders"
