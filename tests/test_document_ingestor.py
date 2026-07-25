"""Tests for grafito.document DocumentIngestor MVP."""

import re

import pytest

from grafito import GrafitoDatabase
from grafito.document import (
    DocumentIngestor,
    FixedChunker,
    MarkdownChunker,
    RecursiveChunker,
)
from grafito.document.ingest import MANAGED_BY
from grafito.embedding_functions.base import EmbeddingFunction


class ToyEmbedder(EmbeddingFunction):
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        out = []
        for text in input:
            vec = [0.0] * self._dim
            for i, tok in enumerate(re.findall(r"[a-z0-9]+", text.lower())):
                vec[hash(tok) % self._dim] += 1.0 + (i % 3) * 0.01
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out

    @staticmethod
    def name() -> str:
        return "toy_document"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    @staticmethod
    def build_from_config(config):
        return ToyEmbedder(dim=int(config.get("dim", 16)))

    def get_config(self):
        return {"dim": self._dim}

    @staticmethod
    def validate_config(config):
        return None

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def db_ing():
    db = GrafitoDatabase(":memory:")
    emb = ToyEmbedder()
    db.create_vector_index("docs_chunks", dim=emb.dimension, embedding_function=emb)
    ing = DocumentIngestor(
        db,
        chunker=MarkdownChunker(max_chars=500, overlap=50),
        embed_index="docs_chunks",
    )
    yield db, ing
    db.close()


def test_fixed_chunker_chars_and_overlap():
    # no whitespace -> word boundary can't help, falls back to hard char windows
    c = FixedChunker(max_size=10, overlap=4)
    specs = c.split("abcdefghijklmnopqrstuvwxyz")
    assert len(specs) >= 3
    assert specs[0].char_start == 0
    assert specs[0].text == "abcdefghij"
    # second window starts at step=6
    assert specs[1].char_start == 6


def test_fixed_chunker_word_boundary():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    c = FixedChunker(max_size=20, overlap=5, boundary="word")
    specs = c.split(text)
    assert len(specs) > 1
    for s in specs:
        assert s.text == text[s.char_start : s.char_end]  # offsets stay exact
        # no chunk starts or ends in the middle of a word
        if s.char_start > 0:
            assert text[s.char_start - 1].isspace() or text[s.char_start].isspace()
        if s.char_end < len(text):
            assert text[s.char_end - 1].isspace() or text[s.char_end].isspace()
    # a single token longer than max_size still terminates (hard cut fallback)
    hard = FixedChunker(max_size=6, overlap=0, boundary="word").split("supercalifragilistic end")
    assert hard and "".join(x.text for x in hard) == "supercalifragilistic end"


def test_fixed_chunker_boundary_none_cuts_words():
    c = FixedChunker(max_size=12, overlap=0, boundary="none")
    specs = c.split("alpha beta gamma delta epsilon")
    assert specs[0].text == "alpha beta g"  # cuts mid-word by design


def test_recursive_chunker_empty_and_short():
    c = RecursiveChunker(max_size=50, overlap=0)
    assert c.split("") == []
    specs = c.split("hello world")
    assert len(specs) == 1
    assert specs[0].text == "hello world"
    assert specs[0].char_start == 0
    assert specs[0].char_end == len("hello world")
    assert specs[0].strategy == "recursive"


def test_recursive_chunker_prefers_paragraphs():
    text = (
        "First paragraph about authentication and sessions.\n\n"
        "Second paragraph about authorization and roles.\n\n"
        "Third paragraph about auditing and compliance."
    )
    c = RecursiveChunker(max_size=60, overlap=0)
    specs = c.split(text)
    assert len(specs) >= 2
    for s in specs:
        assert s.text == text[s.char_start : s.char_end]
        assert len(s.text) <= 60 or "\n\n" not in s.text  # oversized only if no better split
    # Should not hard-cut mid-word if paragraphs fit
    joined_starts = [s.char_start for s in specs]
    assert joined_starts == sorted(joined_starts)


def test_recursive_chunker_hard_cut_long_token():
    text = "x" * 100
    c = RecursiveChunker(max_size=30, overlap=5)
    specs = c.split(text)
    assert len(specs) >= 3
    for s in specs:
        assert s.text == text[s.char_start : s.char_end]
        assert len(s.text) <= 30


def test_recursive_chunker_overlap_and_offsets():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    c = RecursiveChunker(max_size=20, overlap=5)
    specs = c.split(text)
    assert len(specs) > 1
    for s in specs:
        assert s.text == text[s.char_start : s.char_end]
    # With overlap, consecutive windows may overlap in char ranges
    if len(specs) >= 2:
        assert specs[1].char_start < specs[0].char_end or specs[1].char_start == specs[0].char_end


def test_recursive_chunker_custom_separators():
    text = "one|two|three|four|five|six"
    # keep=False: split on "|", then re-insert it when packing under max_size (LC style).
    c = RecursiveChunker(max_size=10, overlap=0, separators=["|", ""], keep_separator=False)
    specs = c.split(text)
    assert len(specs) >= 2
    for s in specs:
        assert s.text == text[s.char_start : s.char_end]
        assert len(s.text) <= 10
    # all original tokens still present across chunks
    joined = "|".join(s.text for s in specs)  # may double seps at boundaries — check tokens
    for tok in ("one", "two", "three", "four", "five", "six"):
        assert any(tok in s.text for s in specs)


def test_recursive_chunker_from_language_python():
    code = (
        "class Foo:\n"
        "    def bar(self):\n"
        "        return 1\n"
        "\n"
        "def baz():\n"
        "    return 2\n"
    )
    c = RecursiveChunker.from_language("python", max_size=40, overlap=0)
    specs = c.split(code)
    assert specs
    assert c.name == "recursive:python"
    for s in specs:
        assert s.text == code[s.char_start : s.char_end]


def test_recursive_chunker_unknown_language():
    with pytest.raises(ValueError, match="unknown language"):
        RecursiveChunker.from_language("cobol")


def test_recursive_chunker_from_language_regex_override():
    # markdown/latex presets default to regex; an explicit override must not
    # raise "multiple values for keyword argument 'is_separator_regex'".
    c = RecursiveChunker.from_language("markdown", is_separator_regex=False)
    assert c.is_separator_regex is False
    assert c.name == "recursive:markdown"


def test_recursive_chunker_degenerate_overlap_no_dup_or_collapse():
    # High overlap vs. tiny pieces used to emit duplicate windows that offset
    # recovery collapsed onto one span (dropping the tail). Now: monotonic
    # starts, no duplicate ranges, exact slices.
    text = "ab ab ab ab ab ab ab ab ab ab"
    specs = RecursiveChunker(max_size=8, overlap=4).split(text)
    ranges = [(s.char_start, s.char_end) for s in specs]
    assert len(ranges) == len(set(ranges))  # no duplicate spans
    assert [s.char_start for s in specs] == sorted(s.char_start for s in specs)
    for s in specs:
        assert s.text == text[s.char_start : s.char_end]
    # most of the document is covered (only a tiny tail may miss in this
    # pathological max_size)
    covered = set()
    for s in specs:
        covered.update(range(s.char_start, s.char_end))
    assert len(covered) >= len(text) - 4


def test_recursive_chunker_ingests_as_passages(db_ing):
    db, _ = db_ing
    body = "Para one about cats.\n\nPara two about dogs.\n\nPara three about birds and fish."
    ing = DocumentIngestor(
        db,
        chunker=RecursiveChunker(max_size=30, overlap=0),
        hierarchy=False,
        embed_index="docs_chunks",
    )
    result = ing.ingest(body, document_key="rec-demo", title="Rec", embed=True)
    assert result.n_passages >= 2
    hits = ing.search("dogs", k=3)
    assert hits


def test_markdown_pluggable_overflow_chunker():
    # A large section split by a pluggable overflow chunker (RecursiveChunker),
    # while the heading hierarchy is preserved and offsets stay exact.
    big = " ".join(f"word{i}" for i in range(300))
    md = f"# Guide\n\nIntro.\n\n## Big\n\n{big}\n"
    mc = MarkdownChunker(
        max_chars=400,
        overlap=40,
        overflow_chunker=RecursiveChunker(max_size=400, overlap=40),
    )
    specs = mc.split(md)
    assert len(specs) > 2  # the big section overflowed into several passages
    assert any(s.strategy == "recursive" for s in specs)  # overflow chunker used
    for s in specs:
        if s.char_start is not None:
            assert s.text == md[s.char_start : s.char_end]  # exact offsets


def test_markdown_preamble_passages():
    md = """# Security

Intro about authentication and session cookies.

## OAuth

OAuth body.

## MFA

MFA body.
"""
    specs = MarkdownChunker(max_chars=1000).split(md)
    texts = [s.text for s in specs]
    assert any("session cookies" in t for t in texts)
    assert any("OAuth body" in t for t in texts)


def test_ingest_search_expand_pack(db_ing):
    db, ing = db_ing
    md = """# Security

Intro about authentication and session cookies.

## OAuth

OAuth details and tokens.

## MFA

Multi factor devices.
"""
    result = ing.ingest(md, document_key="sec-1", title="Security")
    assert result.skipped is False
    assert result.n_passages >= 3
    assert result.generation == 1

    hits = ing.search("session cookies", k=5)
    assert hits
    cookie_hits = [h for h in hits if "session cookies" in h.node.properties.get("text", "")]
    assert cookie_hits, f"expected preamble in hits, got {[h.node.properties.get('text') for h in hits]}"

    expanded = ing.expand(cookie_hits[0].node, window=1)
    assert len(expanded.passages) >= 1
    seqs = [p.properties["global_seq"] for p in expanded.passages]
    assert seqs == sorted(seqs)

    packed = ing.pack(expanded, max_chars=2000, deduplicate_overlap=True)
    assert "session cookies" in packed.text
    assert packed.segments


def test_ingest_idempotent_skip(db_ing):
    db, ing = db_ing
    text = "Hello world document for fingerprint."
    r1 = ing.ingest(text, document_key="idem", embed=True)
    r2 = ing.ingest(text, document_key="idem", embed=True)
    assert r1.skipped is False
    assert r2.skipped is True
    assert r2.generation == r1.generation


def test_replace_generational_gc(db_ing):
    db, ing = db_ing
    r1 = ing.ingest("version one content alpha", document_key="gen", embed=True)
    r2 = ing.replace("gen", "version two content beta different", embed=True)
    assert r2.generation == r1.generation + 1
    old = db.match_nodes(
        properties={
            "managed_by": MANAGED_BY,
            "generation": r1.generation,
            "role": "passage",
        }
    )
    assert old == []
    new = db.match_nodes(
        properties={
            "managed_by": MANAGED_BY,
            "generation": r2.generation,
            "role": "passage",
        }
    )
    assert new


def test_replace_failure_keeps_previous_active(db_ing, monkeypatch):
    db, ing = db_ing
    r1 = ing.ingest("stable content that stays", document_key="fail", embed=True)

    def boom(*args, **kwargs):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(db, "upsert_embeddings", boom)
    with pytest.raises(RuntimeError):
        ing.replace("fail", "new content that should not activate", embed=True)

    parent = db.match_nodes(properties={"document_key": "fail"}, limit=1)[0]
    assert parent.properties["active_generation"] == r1.generation
    # Previous passages still present
    still = db.match_nodes(
        properties={
            "managed_by": MANAGED_BY,
            "generation": r1.generation,
            "role": "passage",
        }
    )
    assert still


def test_external_parent_not_deleted(db_ing):
    db, ing = db_ing
    external = db.create_node(labels=["Concept"], properties={"concept_id": "x"})
    r = ing.ingest("body text", parent_id=external.id, document_key="ext", embed=True)
    assert r.owner_document_id == external.id
    ing.delete(external.id, delete_owned_parent=True)
    # External parent remains
    assert db.get_node(external.id) is not None
    # Managed passages gone
    leftovers = db.match_nodes(
        properties={"managed_by": MANAGED_BY, "owner_document_id": external.id}
    )
    assert leftovers == []


def test_search_ignores_foreign_and_stale(db_ing):
    db, ing = db_ing
    ing.ingest("alpha beta gamma document", document_key="mine", embed=True)
    # Foreign chunk with similar embedding-ish text on same index
    foreign = db.create_node(
        labels=["Chunk"],
        properties={
            "text": "alpha beta gamma document",
            "managed_by": "someone.else",
            "embed_role": "passage",
            "generation": 99,
            "owner_document_id": 999,
        },
    )
    db.upsert_embeddings([foreign.id], [foreign.properties["text"]], index="docs_chunks")
    hits = ing.search("alpha beta gamma", k=5)
    assert all(h.node.properties.get("managed_by") == MANAGED_BY for h in hits)
    assert all(h.node.id != foreign.id for h in hits)


def test_pack_overlap_dedupe():
    db = GrafitoDatabase(":memory:")
    full = "AAA " + ("shared middle zone. " * 8) + "ZZZ"
    # Manual two overlapping nodes
    n1 = db.create_node(
        labels=["Chunk"],
        properties={
            "text": full[0:40],
            "char_start": 0,
            "char_end": 40,
            "global_seq": 0,
            "owner_document_id": 1,
            "managed_by": MANAGED_BY,
        },
    )
    n2 = db.create_node(
        labels=["Chunk"],
        properties={
            "text": full[20:60],
            "char_start": 20,
            "char_end": 60,
            "global_seq": 1,
            "owner_document_id": 1,
            "managed_by": MANAGED_BY,
        },
    )
    parent = db.create_node(
        labels=["Document"],
        properties={"text": full, "managed_by": MANAGED_BY, "role": "document"},
    )
    # Fix owner ids to parent
    for n in (n1, n2):
        p = dict(n.properties)
        p["owner_document_id"] = parent.id
        db.replace_node_properties(n.id, p)
        n = db.get_node(n.id)
    n1 = db.get_node(n1.id)
    n2 = db.get_node(n2.id)
    ing = DocumentIngestor(db, embed_index=None)
    naive = n1.properties["text"] + n2.properties["text"]
    packed = ing.pack([n1, n2], deduplicate_overlap=True, include_citations=False)
    assert len(packed.text) < len(naive)
    db.close()


def test_delete_owned_document(db_ing):
    db, ing = db_ing
    r = ing.ingest("bye", document_key="bye", embed=True)
    ing.delete("bye", delete_owned_parent=True)
    assert db.get_node(r.owner_document_id) is None


def test_pack_max_tokens_without_counter_uses_estimate():
    db = GrafitoDatabase(":memory:")
    parent = db.create_node(
        labels=["Document"],
        properties={"text": "x" * 200, "managed_by": MANAGED_BY, "role": "document"},
    )
    n = db.create_node(
        labels=["Chunk"],
        properties={
            "text": "x" * 200,
            "char_start": 0,
            "char_end": 200,
            "global_seq": 0,
            "owner_document_id": parent.id,
            "managed_by": MANAGED_BY,
        },
    )
    ing = DocumentIngestor(db, embed_index=None)
    # 200 chars ≈ 50 tokens (ceil(len/4)); budget 40 tokens must truncate
    packed = ing.pack([n], max_tokens=40, include_citations=False)
    assert packed.truncated is True
    assert len(packed.text) <= 40 * 4
    db.close()


def test_hierarchy_toc_and_ancestors(db_ing):
    db, ing = db_ing
    md = """# Security

Intro about authentication.

## OAuth

OAuth details tokens.

## MFA

MFA body text.
"""
    r = ing.ingest(md, document_key="hier", title="Security")
    assert r.hierarchy is True
    assert r.n_sections >= 3
    assert r.n_passages >= 3

    toc = ing.toc("hier")
    assert len(toc) == 1
    assert toc[0].title == "Security"
    child_titles = {c.title for c in toc[0].children}
    assert "OAuth" in child_titles and "MFA" in child_titles

    toc_d = ing.toc("hier", as_dict=True)
    assert toc_d[0]["node_key"]
    assert toc_d[0]["n_chunks"] >= 1

    hits = ing.search("OAuth details tokens", k=5)
    oauth_hit = next(
        h for h in hits if "OAuth" in h.node.properties.get("text", "") or "tokens" in h.node.properties.get("text", "")
    )
    expanded = ing.expand(oauth_hit.node, window=1, include_ancestors=True)
    assert expanded.section is not None
    assert expanded.section.properties.get("title") == "OAuth"
    assert any(a.properties.get("title") == "Security" for a in expanded.ancestors)

    loaded = ing.load_sections("hier", [toc[0].node_key])
    assert len(loaded) == 1
    assert loaded[0].properties.get("title") == "Security"


def test_hierarchy_false_skips_sections(db_ing):
    db, ing = db_ing
    ing.hierarchy = False
    ing.chunker = FixedChunker(max_size=80)
    r = ing.ingest("# Title\n\nJust flat body text for this test.", document_key="nohier")
    assert r.hierarchy is False
    assert r.n_sections == 0
    assert r.n_passages >= 1
    assert ing.toc("nohier") == []


def test_build_markdown_tree_unit():
    from grafito.document.tree import build_markdown_tree, flatten_chunks

    md = "# A\n\nintro\n\n## B\n\nbody b\n"
    forest = build_markdown_tree(md, max_chars=500)
    assert forest[0].title == "A"
    assert any(c.title == "B" for c in forest[0].children)
    chunks = flatten_chunks(forest)
    assert any("intro" in c.text for c in chunks)
    assert any("body b" in c.text for c in chunks)


def test_headings_inside_fenced_code_blocks_ignored():
    """Runbooks/OKF often have shell snippets with # comments — not sections."""
    from grafito.document.tree import build_markdown_tree, flatten_chunks

    md = """# Runbook

Setup steps:

```bash
# apt update
## not a heading either
echo hello
```

## Real section

Body of real section.

~~~
# still not a section
~~~
"""
    forest = build_markdown_tree(md, max_chars=2000)
    titles = []

    def walk(secs):
        for s in secs:
            titles.append(s.title)
            walk(s.children)

    walk(forest)
    assert "Runbook" in titles
    assert "Real section" in titles
    assert "apt update" not in titles
    assert "not a heading either" not in titles
    assert "still not a section" not in titles

    # Flat markdown chunker must agree
    specs = MarkdownChunker(max_chars=2000).split(md)
    headings = {s.heading for s in specs if s.heading}
    assert "apt update" not in headings
    assert any("echo hello" in s.text for s in specs)
    assert any("Body of real section" in s.text for s in flatten_chunks(forest))


def test_pack_overlap_without_full_text_stitches():
    db = GrafitoDatabase(":memory:")
    full = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Parent without full text stored
    parent = db.create_node(
        labels=["Document"],
        properties={"managed_by": MANAGED_BY, "role": "document"},
    )
    n1 = db.create_node(
        labels=["Chunk"],
        properties={
            "text": full[0:12],
            "char_start": 0,
            "char_end": 12,
            "global_seq": 0,
            "owner_document_id": parent.id,
            "managed_by": MANAGED_BY,
        },
    )
    n2 = db.create_node(
        labels=["Chunk"],
        properties={
            "text": full[8:20],
            "char_start": 8,
            "char_end": 20,
            "global_seq": 1,
            "owner_document_id": parent.id,
            "managed_by": MANAGED_BY,
        },
    )
    ing = DocumentIngestor(db, embed_index=None, store_full_text=False)
    packed = ing.pack([n1, n2], deduplicate_overlap=True, include_citations=False)
    assert packed.text == full[0:20]
    assert "ABCDEFGH" in packed.text and packed.text.count("IJKL") == 1
    db.close()
