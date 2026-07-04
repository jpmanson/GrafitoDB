from __future__ import annotations

from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.importers.okf import (
    extract_citations,
    extract_links,
    parse_frontmatter,
    parse_log_entries,
    split_citations,
)

pytest.importorskip("yaml")

BUNDLE = Path("examples") / "okf" / "okf_bundle"
KB_BUNDLE = Path("examples") / "okf" / "okf_knowledge_base"


@pytest.fixture
def db() -> GrafitoDatabase:
    database = GrafitoDatabase(":memory:")
    yield database
    database.close()


def _node_by_uri(db: GrafitoDatabase, uri: str):
    matches = [n for n in db.match_nodes() if n.uri == uri]
    assert len(matches) == 1, f"expected exactly one node with uri {uri!r}"
    return matches[0]


# --- Unit: frontmatter / link parsing ---------------------------------------


def test_parse_frontmatter_splits_block_and_body():
    text = "---\ntype: Metric\ntitle: Revenue\n---\n# Body\n\ntext\n"
    fm, body = parse_frontmatter(text)
    assert fm == {"type": "Metric", "title": "Revenue"}
    assert body.startswith("# Body")


def test_parse_frontmatter_no_block():
    fm, body = parse_frontmatter("# Just a heading\n")
    assert fm == {}
    assert body == "# Just a heading\n"


def test_extract_links_resolves_absolute_and_relative():
    body = (
        "See [customers](/tables/customers.md) and "
        "[sibling](./other.md) and [external](https://example.com) "
        "and [anchor](#section)."
    )
    links = extract_links(body, source_id="tables/orders")
    targets = {target for _, target in links}
    assert targets == {"tables/customers", "tables/other"}


# --- Integration: bundle import ---------------------------------------------


def test_import_counts(db):
    summary = db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    assert summary["nodes"] == 3  # sales, orders, customers (index.md skipped)
    assert summary["skipped"] == 1  # index.md
    assert summary["stubs"] == 0
    assert summary["relationships"] > 0
    assert db.get_node_count() == 3


def test_type_becomes_label_and_frontmatter_becomes_properties(db):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    orders = _node_by_uri(db, "okf:tables/orders")
    assert "BigQuery Table" in orders.labels
    assert orders.properties["title"] == "Orders"
    assert orders.properties["tags"] == ["sales", "orders"]
    assert "type" not in orders.properties  # promoted to a label
    assert "# Schema" in orders.properties["body"]
    assert orders.properties["concept_id"] == "tables/orders"


def test_links_become_relationships(db):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    rows = db.execute(
        """
        MATCH (a {title: 'Orders'})-[:LINKS_TO]->(b)
        RETURN b.title AS target ORDER BY target
        """
    )
    targets = {row["target"] for row in rows}
    assert "Customers" in targets
    assert "Sales" in targets


def test_index_and_log_are_skipped(db):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    # index.md must not have produced a node.
    assert not [n for n in db.match_nodes() if n.uri == "okf:index"]


def test_unknown_frontmatter_keys_preserved(db, tmp_path):
    doc = tmp_path / "thing.md"
    doc.write_text(
        "---\ntype: Widget\ntitle: Thing\nokf_version: '0.1'\ncustom_key: 42\n---\nbody\n",
        encoding="utf-8",
    )
    db.import_okf_bundle(str(tmp_path), configure_fts=False)
    node = _node_by_uri(db, "okf:thing")
    assert node.properties["okf_version"] == "0.1"
    assert node.properties["custom_key"] == 42


def test_missing_type_falls_back_to_concept_label(db, tmp_path):
    (tmp_path / "loose.md").write_text(
        "---\ntitle: Loose\n---\nbody\n", encoding="utf-8"
    )
    db.import_okf_bundle(str(tmp_path), configure_fts=False)
    node = _node_by_uri(db, "okf:loose")
    assert node.labels == ["Concept"]


def test_broken_link_creates_stub(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nLink to [missing](/b.md).\n",
        encoding="utf-8",
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False)
    assert summary["stubs"] == 1
    stub = _node_by_uri(db, "okf:b")
    assert stub.labels == ["Concept"]
    assert stub.properties.get("stub") is True


def test_configure_fts_enables_text_search(db):
    if not db.has_fts5():
        pytest.skip("SQLite build lacks FTS5")
    db.import_okf_bundle(str(BUNDLE), configure_fts=True)
    hits = db.text_search("customer", k=5)
    assert hits
    titles = {h["entity"].properties.get("title") for h in hits}
    assert "Customers" in titles


# --- Citations --------------------------------------------------------------


def test_split_citations_extracts_section():
    body = "# Schema\n\ntext\n\n# Citations\n\n[1] [src](https://x.com)\n"
    main, cites = split_citations(body)
    assert "# Schema" in main
    assert "# Citations" not in main
    assert "https://x.com" in cites


def test_parse_log_entries():
    log = (
        "# Directory Update Log\n"
        "## 2026-05-22\n"
        "* **Update**: Added [Metrics](/tables/metrics.md).\n"
        "* **Creation**: Established the playbook.\n"
        "## 2026-05-15\n"
        "* Plain entry without a kind.\n"
    )
    entries = parse_log_entries(log)
    assert len(entries) == 3
    assert entries[0] == ("2026-05-22", "Update", "**Update**: Added [Metrics](/tables/metrics.md).")
    assert entries[1][1] == "Creation"
    assert entries[2] == ("2026-05-15", None, "Plain entry without a kind.")


def test_split_citations_absent():
    main, cites = split_citations("# Schema\n\ntext\n")
    assert cites == ""
    assert main == "# Schema\n\ntext\n"


def test_extract_citations_markdown_and_bare_urls():
    block = (
        "# Citations\n"
        "[1] [Announcement](https://cloud.google.com/blog)\n"
        "- https://support.google.com/answer/123\n"
        "[2] [Runbook](/references/runbook.md)\n"
    )
    cites = extract_citations(block, source_id="tables/orders")
    kinds = {(kind, value) for _, kind, value in cites}
    assert ("external", "https://cloud.google.com/blog") in kinds
    assert ("external", "https://support.google.com/answer/123") in kinds
    assert ("concept", "references/runbook") in kinds


def test_external_citation_creates_reference_node(db, tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\n---\n"
        "Body text.\n\n# Citations\n- https://example.com/spec\n",
        encoding="utf-8",
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False)
    assert summary["citations"] == 1
    assert summary["references"] == 1
    rows = db.execute(
        "MATCH (a {title: 'Doc'})-[:CITES]->(r:Reference) RETURN r.url AS url"
    )
    assert rows[0]["url"] == "https://example.com/spec"


def test_duplicate_external_citation_deduplicated(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nx\n\n# Citations\n- https://example.com/s\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "---\ntype: Doc\ntitle: B\n---\ny\n\n# Citations\n- https://example.com/s\n",
        encoding="utf-8",
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False)
    assert summary["citations"] == 2
    assert summary["references"] == 1  # one shared Reference node
    refs = [n for n in db.match_nodes() if "Reference" in n.labels]
    assert len(refs) == 1


def test_intra_bundle_citation_is_cites_not_links_to(db, tmp_path):
    (tmp_path / "src.md").write_text(
        "---\ntype: Doc\ntitle: Src\n---\n"
        "See [other](/other.md).\n\n# Citations\n[1] [Ref](/ref.md)\n",
        encoding="utf-8",
    )
    db.import_okf_bundle(str(tmp_path), configure_fts=False)
    # The body link is LINKS_TO; the citation link is CITES.
    links = db.execute("MATCH (a {title: 'Src'})-[:LINKS_TO]->(b) RETURN b.concept_id AS c")
    cites = db.execute("MATCH (a {title: 'Src'})-[:CITES]->(b) RETURN b.concept_id AS c")
    assert {r["c"] for r in links} == {"other"}
    assert {r["c"] for r in cites} == {"ref"}


def test_citations_can_be_disabled(db, tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\n---\nx\n\n# Citations\n- https://example.com/s\n",
        encoding="utf-8",
    )
    summary = db.import_okf_bundle(str(tmp_path), citations=False, configure_fts=False)
    assert summary["citations"] == 0
    assert summary["references"] == 0


# --- Narrative (non-tabular) knowledge-base example bundle -------------------


def test_knowledge_base_bundle_imports(db):
    summary = db.import_okf_bundle(str(KB_BUNDLE), configure_fts=False)
    # 3 ADRs + 1 runbook + 3 glossary terms = 7 concepts; 4 index.md skipped.
    assert summary["nodes"] == 7
    assert summary["skipped"] == 4
    assert summary["stubs"] == 0  # all cross-links resolve within the bundle
    assert summary["citations"] > 0
    labels = set(db.get_all_labels())
    assert {"ADR", "Playbook", "Term", "Reference"} <= labels


def test_knowledge_base_cross_links_resolve(db):
    db.import_okf_bundle(str(KB_BUNDLE), configure_fts=False)
    rows = db.execute(
        """
        MATCH (a {title: 'Add optional vector search'})-[:LINKS_TO]->(b)
        RETURN DISTINCT b.title AS title ORDER BY title
        """
    )
    titles = {row["title"] for row in rows}
    assert "Semantic search" in titles
    assert "Use SQLite as the storage engine" in titles


# --- Permissive consumption: malformed frontmatter ---------------------------


def test_malformed_frontmatter_does_not_abort_import(db, tmp_path):
    (tmp_path / "good.md").write_text(
        "---\ntype: Doc\ntitle: Good\n---\nBody.\n", encoding="utf-8"
    )
    (tmp_path / "bad.md").write_text(
        "---\ntype: Doc\ntitle: [unclosed\n---\nStill useful text.\n", encoding="utf-8"
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False)
    assert summary["nodes"] == 2
    assert summary["malformed"] == ["bad"]
    # The bad file falls back to the generic label with its full text as body.
    bad = _node_by_uri(db, "okf:bad")
    assert bad.labels == ["Concept"]
    assert "Still useful text." in bad.properties["body"]


# --- Conformance validation (SPEC sec. 9) ------------------------------------


def test_validate_conformant_bundle():
    from grafito.okf import validate_okf_bundle

    report = validate_okf_bundle(str(KB_BUNDLE))
    assert report["conformant"] is True
    assert report["files"] == 7
    assert report["errors"] == []


def test_validate_reports_errors_and_warnings(tmp_path):
    from grafito.okf import validate_okf_bundle

    (tmp_path / "ok.md").write_text(
        "---\ntype: Doc\n---\nSee [missing](/nowhere.md).\n", encoding="utf-8"
    )
    (tmp_path / "no-type.md").write_text("---\ntitle: X\n---\nBody.\n", encoding="utf-8")
    (tmp_path / "bad-yaml.md").write_text("---\ntitle: [oops\n---\nBody.\n", encoding="utf-8")
    (tmp_path / "no-frontmatter.md").write_text("# Just markdown\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "index.md").write_text("---\nokf_version: '0.1'\n---\n# Index\n", encoding="utf-8")

    report = validate_okf_bundle(str(tmp_path))
    assert report["conformant"] is False
    assert report["files"] == 4
    errors = {e["path"]: e["error"] for e in report["errors"]}
    assert "missing or empty required field: type" in errors["no-type.md"]
    assert "not valid YAML" in errors["bad-yaml.md"]
    assert errors["no-frontmatter.md"] == "missing frontmatter block"
    warnings = {w["path"]: w["warning"] for w in report["warnings"]}
    assert "broken link to unknown concept: nowhere" in warnings["ok.md"]
    assert "root index.md" in warnings["sub/index.md"]


def test_validate_missing_bundle_raises(tmp_path):
    from grafito.okf import validate_okf_bundle

    with pytest.raises(NotADirectoryError):
        validate_okf_bundle(str(tmp_path / "nope"))


# --- Typed links from headings --------------------------------------------------


def _typed_bundle(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Table\ntitle: A\n---\n"
        "Intro link to [b](/b.md).\n\n"
        "# Joins with\n\nJoined with [b](/b.md) on id.\n\n"
        "# Links\n\nAlso see [c](/c.md).\n\n"
        "# Depends on!\n\n[c](/c.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text("---\ntype: Table\ntitle: B\n---\nx\n", encoding="utf-8")
    (tmp_path / "c.md").write_text("---\ntype: Table\ntitle: C\n---\nx\n", encoding="utf-8")


def test_typed_links_from_headings(db, tmp_path):
    _typed_bundle(tmp_path)
    db.import_okf_bundle(str(tmp_path), typed_links=True, configure_fts=False)
    rows = db.execute(
        "MATCH (a {title: 'A'})-[r]->(b) RETURN type(r) AS t, b.title AS target ORDER BY t"
    )
    pairs = {(row["t"], row["target"]) for row in rows}
    # Before any heading and under `# Links` -> the default type; headings
    # normalize (`Joins with` -> JOINS_WITH, `Depends on!` -> DEPENDS_ON).
    assert pairs == {
        ("LINKS_TO", "B"),
        ("JOINS_WITH", "B"),
        ("LINKS_TO", "C"),
        ("DEPENDS_ON", "C"),
    }


def test_typed_links_off_by_default(db, tmp_path):
    _typed_bundle(tmp_path)
    db.import_okf_bundle(str(tmp_path), configure_fts=False)
    rows = db.execute("MATCH (a {title: 'A'})-[r]->(b) RETURN type(r) AS t")
    assert {row["t"] for row in rows} == {"LINKS_TO"}


def test_rel_type_from_heading_normalization():
    from grafito.importers.okf import rel_type_from_heading

    assert rel_type_from_heading("Joins with") == "JOINS_WITH"
    assert rel_type_from_heading("Depends on!") == "DEPENDS_ON"
    assert rel_type_from_heading("Links") is None  # conventional default section
    assert rel_type_from_heading(None) is None
    assert rel_type_from_heading("123") is None  # not a valid type identifier


# --- Progress reporting -------------------------------------------------------


def test_progress_every_prints(db, capsys):
    db.import_okf_bundle(str(KB_BUNDLE), configure_fts=False, progress_every=2)
    out = capsys.readouterr().out
    assert "Importing concepts: 2" in out
    assert "Imported 7 concepts." in out
    assert "links." in out
    assert "citations." in out


def test_progress_callback_receives_phases(db):
    events: list[tuple[str, int]] = []
    db.import_okf_bundle(
        str(KB_BUNDLE), configure_fts=False, progress=lambda phase, n: events.append((phase, n))
    )
    phases = [phase for phase, _ in events]
    assert phases.count("concepts") == 8  # 7 per-file ticks + the phase-end report
    assert ("concepts", 7) in events
    assert phases[-1] == "done"
    assert "links" in phases and "citations" in phases


def test_progress_callback_silent_on_stdout(db, capsys):
    db.import_okf_bundle(str(KB_BUNDLE), configure_fts=False, progress=lambda *_: None)
    assert capsys.readouterr().out == ""


# --- concept_id expression index ----------------------------------------------


def test_import_creates_concept_id_index(db):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    plan = db.conn.execute(
        "EXPLAIN QUERY PLAN SELECT n.id FROM nodes n "
        "WHERE json_extract(n.properties, '$.concept_id') = ?",
        ("tables/orders",),
    ).fetchall()
    assert any("INDEX" in row["detail"].upper() for row in plan)
