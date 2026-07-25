"""OKF long-body opt-in: chunk long concept bodies, keep short concepts single-node."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.document.ingest import MANAGED_BY
from grafito.embedding_functions import EmbeddingFunction
from grafito.okf import OKFBundle

pytest.importorskip("yaml")


class _Embedder(EmbeddingFunction):
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
        return "long_body_embedder"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

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


LONG_BODY = (
    "# Overview\n\n"
    "This runbook explains how to triage slow database queries in production. "
    "Start by checking the connection pool for saturation and timeouts.\n\n"
    "## Connection pool\n\n"
    "When the pool is exhausted, requests queue and latency spikes. Increase the "
    "pool size or reduce long-held transactions. Watch for leaked connections.\n\n"
    "## Indexes\n\n"
    "Missing indexes cause full table scans. Use EXPLAIN to inspect the query plan "
    "and add covering indexes for the hot predicates in the WHERE clause.\n"
)


@pytest.fixture
def bundle(tmp_path) -> OKFBundle:
    empty = tmp_path / "kb"
    empty.mkdir()
    b = OKFBundle.load(str(empty), embed=_Embedder())
    yield b
    b.db.close()


def test_short_concept_not_chunked(bundle):
    bundle.enable_body_chunking(threshold=200, embed=_Embedder())
    bundle.add_concept("notes/short", type="Note", title="Short", body="Tiny body.")
    passages = bundle.db.match_nodes(properties={"managed_by": MANAGED_BY, "role": "passage"})
    assert passages == []


def test_long_concept_chunked_into_passages(bundle):
    bundle.enable_body_chunking(threshold=200, max_chars=120, embed=_Embedder())
    bundle.add_concept("runbooks/slow", type="Runbook", title="Slow queries", body=LONG_BODY)
    passages = bundle.db.match_nodes(
        properties={"managed_by": MANAGED_BY, "role": "passage", "corpus": "okf"}
    )
    assert len(passages) > 1
    hits = bundle.search_passages("connection pool timeouts", k=3)
    assert hits
    assert all(h.node.properties.get("managed_by") == MANAGED_BY for h in hits)


def test_concept_node_not_polluted_with_bookkeeping(bundle):
    bundle.enable_body_chunking(threshold=200, embed=_Embedder())
    bundle.add_concept("runbooks/slow", type="Runbook", title="Slow", body=LONG_BODY)
    node = bundle.concept("runbooks/slow").node
    for leaked in ("active_generation", "active_fingerprint", "active_version_id", "document_key"):
        assert leaked not in node.properties, f"{leaked} leaked onto the concept node"


def test_chunked_concept_excluded_from_concept_listing(bundle):
    bundle.enable_body_chunking(threshold=200, embed=_Embedder())
    bundle.add_concept("runbooks/slow", type="Runbook", title="Slow", body=LONG_BODY)
    ids = {c.id for c in bundle}
    assert ids == {"runbooks/slow"}  # managed Document/version/passages are not concepts


def test_search_passages_scoped_to_concept(bundle):
    bundle.enable_body_chunking(threshold=200, max_chars=120, embed=_Embedder())
    bundle.add_concept("runbooks/slow", type="Runbook", title="Slow", body=LONG_BODY)
    bundle.add_concept(
        "runbooks/deploy",
        type="Runbook",
        title="Deploy",
        body=LONG_BODY.replace("slow database queries", "deployment rollouts"),
    )
    scoped = bundle.search_passages("connection pool", k=10, concept_id="runbooks/slow")
    doc = bundle._chunk_doc_for("runbooks/slow")
    assert scoped
    assert all(h.owner_document_id == doc.id for h in scoped)


def test_update_to_short_body_removes_passages(bundle):
    bundle.enable_body_chunking(threshold=200, embed=_Embedder())
    bundle.add_concept("runbooks/slow", type="Runbook", title="Slow", body=LONG_BODY)
    assert bundle.db.match_nodes(properties={"managed_by": MANAGED_BY, "role": "passage"})
    bundle.update_concept("runbooks/slow", body="Now short.")
    assert bundle.db.match_nodes(properties={"managed_by": MANAGED_BY, "role": "passage"}) == []


def test_update_long_body_regenerates(bundle):
    bundle.enable_body_chunking(threshold=200, max_chars=120, embed=_Embedder())
    bundle.add_concept("runbooks/slow", type="Runbook", title="Slow", body=LONG_BODY)
    bundle.update_concept("runbooks/slow", body=LONG_BODY + "\n\n## Extra\n\nMore content here.\n")
    # Only one active generation survives (old GC'd).
    versions = bundle.db.match_nodes(
        properties={"managed_by": MANAGED_BY, "role": "version", "status": "ACTIVE"}
    )
    assert len(versions) == 1


def test_remove_concept_removes_passages(bundle):
    bundle.enable_body_chunking(threshold=200, embed=_Embedder())
    bundle.add_concept("runbooks/slow", type="Runbook", title="Slow", body=LONG_BODY)
    assert bundle.db.match_nodes(properties={"managed_by": MANAGED_BY})
    bundle.remove_concept("runbooks/slow")
    assert bundle.db.match_nodes(properties={"managed_by": MANAGED_BY}) == []


def test_disabled_by_default(bundle):
    # No enable_body_chunking → classic single-node behaviour.
    bundle.add_concept("runbooks/slow", type="Runbook", title="Slow", body=LONG_BODY)
    assert bundle.db.match_nodes(properties={"managed_by": MANAGED_BY}) == []
    with pytest.raises(ValueError, match="enable_body_chunking"):
        bundle.search_passages("x")


def test_chunked_bundle_round_trips(tmp_path):
    empty = tmp_path / "kb"
    empty.mkdir()
    b = OKFBundle.load(str(empty), embed=_Embedder())
    b.enable_body_chunking(threshold=200, embed=_Embedder())
    b.add_concept("runbooks/slow", type="Runbook", title="Slow", body=LONG_BODY)
    out = tmp_path / "out"
    b.save(str(out))
    # The concept round-trips as a single markdown file; managed nodes do not.
    md_files = {p.name for p in out.glob("**/*.md")}
    assert "slow.md" in {Path(p).name for p in out.glob("runbooks/*.md")} or "slow.md" in md_files
    b.db.close()
