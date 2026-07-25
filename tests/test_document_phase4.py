"""Deferred features: Chonkie adapter, dual multi-view indexing, OKF long-body opt-in."""

import re

import pytest

from grafito import GrafitoDatabase
from grafito.document import ChonkieChunker, DocumentIngestor, FixedChunker, MarkdownChunker
from grafito.document.ingest import MANAGED_BY
from grafito.embedding_functions.base import EmbeddingFunction


class ToyEmbedder(EmbeddingFunction):
    def __init__(self, dim: int = 24) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        out = []
        for text in input:
            vec = [0.0] * self._dim
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                vec[hash(tok) % self._dim] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out

    @staticmethod
    def name() -> str:
        return "toy_p4"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    @staticmethod
    def build_from_config(config):
        return ToyEmbedder()

    def get_config(self):
        return {"dim": self._dim}

    @staticmethod
    def validate_config(config):
        return None

    @property
    def dimension(self) -> int:
        return self._dim


# --------------------------------------------------------------------------
# Chonkie adapter (no real chonkie dependency; stub a Chonkie-like chunker)
# --------------------------------------------------------------------------


class _StubChunk:
    def __init__(self, text, start, end, tokens):
        self.text = text
        self.start_index = start
        self.end_index = end
        self.token_count = tokens


class _StubChonkieChunker:
    """Mimics a chonkie chunker: .chunk(text) -> list of chunk objects."""

    def chunk(self, text):
        # Split into ~20-char windows on word boundaries, chonkie-style objects.
        out = []
        i = 0
        while i < len(text):
            j = min(i + 20, len(text))
            out.append(_StubChunk(text[i:j], i, j, (j - i) // 4))
            i = j
        return out


def test_chonkie_adapter_maps_offsets_and_tokens():
    chunker = ChonkieChunker(_StubChonkieChunker(), name="chonkie:stub")
    specs = chunker.split("hello world this is a longer body of text to split")
    assert len(specs) > 1
    assert specs[0].char_start == 0
    assert all(s.strategy == "chonkie:stub" for s in specs)
    assert all(s.token_count is not None for s in specs)
    # Offsets contiguous and ord dense
    assert [s.ord for s in specs] == list(range(len(specs)))
    for a, b in zip(specs, specs[1:]):
        assert a.char_end == b.char_start


def test_chonkie_adapter_rejects_non_chunker():
    with pytest.raises(TypeError):
        ChonkieChunker(object())


def test_chonkie_adapter_empty_text():
    assert ChonkieChunker(_StubChonkieChunker()).split("") == []


def test_chonkie_from_recipe_without_dep_raises():
    pytest.importorskip  # noqa: B018 - marker only
    try:
        import chonkie  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="grafito\\[document-chonkie\\]"):
            ChonkieChunker.from_recipe("recursive", chunk_size=128)


def test_chonkie_chunker_ingests_as_passages():
    db = GrafitoDatabase(":memory:")
    db.create_vector_index("c", embedding_function=ToyEmbedder())
    ing = DocumentIngestor(
        db,
        chunker=ChonkieChunker(_StubChonkieChunker()),
        embed_index="c",
        hierarchy=False,
    )
    r = ing.ingest("some reasonably long text body for chonkie chunking here", document_key="ck")
    assert r.n_passages > 1
    hits = ing.search("chonkie", k=3)
    assert all(h.node.properties.get("managed_by") == MANAGED_BY for h in hits)


# --------------------------------------------------------------------------
# Dual multi-view indexing
# --------------------------------------------------------------------------

MULTIVIEW_DOC = """# Security

Authentication overview and password hashing rules for the service.

## OAuth

OAuth token rotation and refresh handling for connected clients.

## Sessions

Session cookies, timeouts, and revocation of active sessions.
"""


def _mv_ingestor(views=None):
    db = GrafitoDatabase(":memory:")
    db.create_vector_index("mv", embedding_function=ToyEmbedder())
    ing = DocumentIngestor(
        db,
        chunker=MarkdownChunker(max_chars=80, overlap=0),
        embed_index="mv",
        views=views,
    )
    return db, ing


def test_multiview_writes_both_segmentations():
    db, ing = _mv_ingestor()
    r = ing.ingest(MULTIVIEW_DOC, document_key="sec", views=["hierarchy", "fixed"])
    assert set(r.views) == {"hierarchy", "fixed"}
    passages = db.match_nodes(
        properties={"managed_by": MANAGED_BY, "role": "passage", "owner_document_id": r.owner_document_id}
    )
    views = {p.properties.get("view") for p in passages}
    assert views == {"hierarchy", "fixed"}
    # Each view numbers global_seq from 0 independently
    hier_seq = sorted(
        int(p.properties["global_seq"]) for p in passages if p.properties["view"] == "hierarchy"
    )
    fixed_seq = sorted(
        int(p.properties["global_seq"]) for p in passages if p.properties["view"] == "fixed"
    )
    assert hier_seq[0] == 0 and fixed_seq[0] == 0


def test_multiview_search_view_filter():
    db, ing = _mv_ingestor()
    ing.ingest(MULTIVIEW_DOC, document_key="sec", views=["hierarchy", "fixed"])
    only_fixed = ing.search("oauth token rotation", k=10, views=["fixed"])
    assert only_fixed
    assert all(h.view == "fixed" for h in only_fixed)
    only_hier = ing.search("oauth token rotation", k=10, views=["hierarchy"])
    assert all(h.view == "hierarchy" for h in only_hier)


def test_multiview_diversify_by_span_dedupes_overlap():
    db, ing = _mv_ingestor()
    ing.ingest(MULTIVIEW_DOC, document_key="sec", views=["hierarchy", "fixed"])
    # Without span diversify, the same region can appear from both views.
    fused = ing.search("session cookies revocation", k=6, diversify_by_span=True)
    # No two accepted hits from the same doc overlap in char span.
    spans = [
        (int(h.node.properties["char_start"]), int(h.node.properties["char_end"]))
        for h in fused
        if h.node.properties.get("char_start") is not None
    ]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            assert not (a[0] < b[1] and b[0] < a[1]), f"overlapping spans survived: {a} {b}"


def test_multiview_expand_stays_within_view():
    db, ing = _mv_ingestor()
    ing.ingest(MULTIVIEW_DOC, document_key="sec", views=["hierarchy", "fixed"])
    hits = ing.search("oauth", k=10, views=["fixed"])
    exp = ing.expand(hits[0].node, window=2)
    assert all(p.properties.get("view") == "fixed" for p in exp.passages)


def test_multiview_replace_removes_all_views():
    db, ing = _mv_ingestor()
    r1 = ing.ingest(MULTIVIEW_DOC, document_key="sec", views=["hierarchy", "fixed"])
    ing.replace("sec", "# New\n\nCompletely different body.", embed=True)
    old = db.match_nodes(
        properties={"managed_by": MANAGED_BY, "role": "passage", "generation": r1.generation}
    )
    assert old == []


def test_multiview_fingerprint_changes_with_views():
    db, ing = _mv_ingestor()
    fp_single = ing._fingerprint(MULTIVIEW_DOC, embed=True, views=["hierarchy"])
    fp_dual = ing._fingerprint(MULTIVIEW_DOC, embed=True, views=["hierarchy", "fixed"])
    assert fp_single != fp_dual


def test_unknown_view_rejected():
    db, ing = _mv_ingestor()
    with pytest.raises(Exception, match="unknown view"):
        ing.ingest(MULTIVIEW_DOC, document_key="sec", views=["banana"])
