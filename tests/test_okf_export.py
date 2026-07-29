from __future__ import annotations

from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.integrations import export_okf_bundle
from grafito.importers.okf import parse_frontmatter

pytest.importorskip("yaml")

BUNDLE = Path("examples") / "okf" / "okf_bundle"


@pytest.fixture
def db() -> GrafitoDatabase:
    database = GrafitoDatabase(":memory:")
    yield database
    database.close()


def _read(path: Path) -> tuple[dict, str]:
    return parse_frontmatter(path.read_text(encoding="utf-8"))


def test_export_writes_concept_files(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    summary = db.export_okf_bundle(str(tmp_path))
    assert summary["concepts"] == 3
    assert (tmp_path / "tables" / "orders.md").exists()
    assert (tmp_path / "datasets" / "sales.md").exists()
    assert (tmp_path / "index.md").exists()


def test_exported_frontmatter_has_type_from_label(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    db.export_okf_bundle(str(tmp_path))
    fm, body = _read(tmp_path / "tables" / "orders.md")
    assert fm["type"] == "BigQuery Table"
    assert fm["title"] == "Orders"
    assert "type" not in {k for k in fm if k != "type"} or fm["type"]  # type present once
    assert "# Schema" in body
    # Internal bookkeeping keys must not leak into frontmatter.
    assert "body" not in fm
    assert "concept_id" not in fm


def test_per_directory_indexes(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    db.export_okf_bundle(str(tmp_path))

    # Root index links to child directory indexes, not directly to concepts.
    root_index = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert "# Subdirectories" in root_index
    assert "[tables](tables/index.md)" in root_index
    assert "[datasets](datasets/index.md)" in root_index

    # Each subdirectory has its own index grouping concepts by type, with
    # relative links and quoted descriptions.
    tables_index = (tmp_path / "tables" / "index.md").read_text(encoding="utf-8")
    assert "# BigQuery Table" in tables_index
    assert "[Orders](orders.md)" in tables_index
    assert '"One row per completed customer order."' in tables_index


def test_write_viz_emits_html(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    summary = db.export_okf_bundle(str(tmp_path), write_viz=True)
    assert summary["viz"] is True
    viz = tmp_path / "viz.html"
    assert viz.exists()
    assert "<html" in viz.read_text(encoding="utf-8").lower()


def test_roundtrip_preserves_graph(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    db.export_okf_bundle(str(tmp_path))

    db2 = GrafitoDatabase(":memory:")
    db2.import_okf_bundle(str(tmp_path), configure_fts=False)
    try:
        assert db2.get_node_count() == db.get_node_count()
        assert db2.get_relationship_count() == db.get_relationship_count()
        orders = [n for n in db2.match_nodes() if n.uri == "okf:tables/orders"]
        assert len(orders) == 1
        assert "BigQuery Table" in orders[0].labels
        assert orders[0].properties["tags"] == ["sales", "orders"]
    finally:
        db2.close()


def test_stub_nodes_are_not_exported(db, tmp_path):
    db.create_node(labels=["Doc"], properties={"title": "A"}, uri="okf:a")
    db.create_node(labels=["Concept"], properties={"stub": True}, uri="okf:ghost")
    summary = db.export_okf_bundle(str(tmp_path))
    assert summary["concepts"] == 1
    assert summary["skipped"] == 1
    assert not (tmp_path / "ghost.md").exists()


def test_auto_reference_nodes_not_exported(db, tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\n---\nx\n\n# Citations\n- https://example.com/s\n",
        encoding="utf-8",
    )
    db.import_okf_bundle(str(tmp_path), configure_fts=False)
    out = tmp_path / "out"
    summary = db.export_okf_bundle(str(out))
    # The Doc concept is written; the auto-created Reference node is not.
    assert summary["concepts"] == 1
    assert summary["skipped"] == 1
    md_files = [p.name for p in out.rglob("*.md") if p.name != "index.md"]
    assert md_files == ["doc.md"]


def test_programmatic_node_synthesizes_links_section(db, tmp_path):
    a = db.create_node(labels=["Doc"], properties={"title": "A"}, uri="okf:a")
    b = db.create_node(labels=["Doc"], properties={"title": "B"}, uri="okf:b")
    db.create_relationship(a.id, b.id, "LINKS_TO", properties={"anchor": "see B"})
    db.export_okf_bundle(str(tmp_path))
    _, body = _read(tmp_path / "a.md")
    assert "# Links" in body
    assert "[see B](/b.md)" in body


# --- Provenance: the `sources` frontmatter (OKF v0.2 sec. 5.1) ----------------


def test_sources_frontmatter_round_trips_verbatim(db, tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    (source / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\nsources:\n"
        "  - id: policy\n    resource: https://wiki.acme/policy\n    title: Policy\n"
        "    author: team:finance\n    usage_count: 5000\n"
        "usage_window: {from: 2026-06-01, to: 2026-06-30}\n---\n\nBody.\n",
        encoding="utf-8",
    )
    db.import_okf_bundle(str(source), configure_fts=False)
    out = tmp_path / "out"
    db.export_okf_bundle(str(out))

    fm, body = _read(out / "doc.md")
    assert fm["sources"] == [
        {
            "id": "policy",
            "resource": "https://wiki.acme/policy",
            "title": "Policy",
            "author": "team:finance",
            "usage_count": 5000,
        }
    ]
    # The shared window stays a sibling of `sources`, not repeated per entry.
    assert fm["usage_window"] == {"from": "2026-06-01", "to": "2026-06-30"}
    assert "# Citations" not in body
    # Provenance is emitted last, after the producer-defined keys.
    assert list(fm)[-2:] == ["sources", "usage_window"]


def test_citation_edge_added_to_the_graph_reaches_frontmatter(db, tmp_path):
    doc = db.create_node(labels=["Doc"], properties={"title": "Doc"}, uri="okf:doc")
    ref = db.create_node(
        labels=["Reference"],
        properties={"title": "Spec", "url": "https://example.com/spec", "okf_auto": True},
        uri="https://example.com/spec",
    )
    db.create_relationship(
        doc.id, ref.id, "CITES", properties={"anchor": "Spec", "source_id": "spec"}
    )
    db.export_okf_bundle(str(tmp_path))

    fm, body = _read(tmp_path / "doc.md")
    assert fm["sources"] == [
        {"id": "spec", "resource": "https://example.com/spec", "title": "Spec"}
    ]
    assert "# Citations" not in body


def test_legacy_citations_stay_in_the_body(db, tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    (source / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\n---\nx\n\n# Citations\n- https://example.com/s\n",
        encoding="utf-8",
    )
    db.import_okf_bundle(str(source), configure_fts=False)
    out = tmp_path / "out"
    db.export_okf_bundle(str(out))

    fm, body = _read(out / "doc.md")
    # A v0.1 citation list is not lifted into frontmatter: the body already
    # carries it, and doing both would double the edges on re-import.
    assert "sources" not in fm
    assert "# Citations" in body

    db2 = GrafitoDatabase(":memory:")
    try:
        summary = db2.import_okf_bundle(str(out), configure_fts=False)
        assert summary["citations"] == 1
    finally:
        db2.close()


def test_source_naming_a_concept_uses_the_bundle_relative_form(db, tmp_path):
    a = db.create_node(labels=["Doc"], properties={"title": "A"}, uri="okf:notes/a")
    b = db.create_node(labels=["Doc"], properties={"title": "B"}, uri="okf:notes/b")
    db.create_relationship(a.id, b.id, "CITES", properties={"anchor": "B"})
    db.export_okf_bundle(str(tmp_path))
    fm, _ = _read(tmp_path / "notes" / "a.md")
    assert fm["sources"] == [{"resource": "/notes/b.md", "title": "B"}]


def test_authored_entry_wins_over_an_equivalent_edge(db, tmp_path):
    source = tmp_path / "in"
    (source / "notes").mkdir(parents=True)
    (source / "notes" / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\nsources:\n  - resource: ../notes/b.md\n    title: Authored\n"
        "---\n\nx\n",
        encoding="utf-8",
    )
    (source / "notes" / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\ny\n", encoding="utf-8")
    db.import_okf_bundle(str(source), configure_fts=False)
    out = tmp_path / "out"
    db.export_okf_bundle(str(out))

    fm, _ = _read(out / "notes" / "a.md")
    # The edge resolves to the same concept as the authored entry, written a
    # different way — it must not be appended as a second source.
    assert fm["sources"] == [{"resource": "../notes/b.md", "title": "Authored"}]


def test_trust_and_lifecycle_lead_the_producer_defined_keys(db, tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    (source / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\nowner: data-team\nstatus: stable\n"
        "generated: { by: agent/x, at: 2026-06-28T14:00:00Z }\n"
        "verified: { by: human:jp, at: 2026-06-25T09:00:00Z }\n"
        "stale_after: 2026-12-31\n---\n\nBody.\n",
        encoding="utf-8",
    )
    db.import_okf_bundle(str(source), configure_fts=False)
    out = tmp_path / "out"
    db.export_okf_bundle(str(out))

    fm, _ = _read(out / "doc.md")
    # The trust and lifecycle families sit where the SPEC's examples put them:
    # after the recommended keys, before anything producer-defined.
    assert list(fm) == ["type", "title", "status", "generated", "verified", "stale_after", "owner"]
    # A single verifier stays the bare mapping it was authored as (sec. 5.2).
    assert fm["verified"] == {"by": "human:jp", "at": "2026-06-25T09:00:00Z"}


def test_okf_version_is_declared_in_the_root_index_only(db, tmp_path):
    db.create_node(labels=["Doc"], properties={"title": "A"}, uri="okf:notes/a")
    db.export_okf_bundle(str(tmp_path), okf_version="0.2")

    root = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert root.startswith('---\nokf_version: \'0.2\'\n---\n')
    # Only the root index may carry frontmatter (SPEC sec. 12).
    assert not (tmp_path / "notes" / "index.md").read_text(encoding="utf-8").startswith("---")


def test_no_version_declared_writes_no_frontmatter(db, tmp_path):
    db.create_node(labels=["Doc"], properties={"title": "A"}, uri="okf:a")
    db.export_okf_bundle(str(tmp_path))
    assert not (tmp_path / "index.md").read_text(encoding="utf-8").startswith("---")


# --- Pruning orphaned concept files ------------------------------------------


def test_prune_deletes_orphaned_concept_files(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    db.export_okf_bundle(str(tmp_path))
    assert (tmp_path / "tables" / "orders.md").exists()

    orders = next(n for n in db.match_nodes() if n.uri == "okf:tables/orders")
    db.delete_node(orders.id)
    summary = db.export_okf_bundle(str(tmp_path), prune=True)
    assert summary["pruned"] == 1
    assert not (tmp_path / "tables" / "orders.md").exists()
    assert (tmp_path / "tables" / "customers.md").exists()


def test_prune_preserves_log_and_non_markdown_files(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    db.export_okf_bundle(str(tmp_path))
    (tmp_path / "log.md").write_text("# Log\n\n## 2026-01-01\n* **Update**: x\n", encoding="utf-8")
    (tmp_path / "diagram.png").write_bytes(b"\x89PNG")

    summary = db.export_okf_bundle(str(tmp_path), prune=True)
    assert summary["pruned"] == 0
    assert (tmp_path / "log.md").exists()
    assert (tmp_path / "diagram.png").exists()


def test_prune_removes_emptied_directories(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    db.export_okf_bundle(str(tmp_path))

    for node in list(db.match_nodes()):
        if node.uri and node.uri.startswith("okf:datasets/"):
            db.delete_node(node.id)
    db.export_okf_bundle(str(tmp_path), prune=True)
    assert not (tmp_path / "datasets").exists()
    assert (tmp_path / "tables").exists()


def test_synthesized_body_groups_links_by_type(db, tmp_path):
    a = db.create_node(labels=["Table"], properties={"title": "A"}, uri="okf:a")
    b = db.create_node(labels=["Table"], properties={"title": "B"}, uri="okf:b")
    c = db.create_node(labels=["Table"], properties={"title": "C"}, uri="okf:c")
    db.create_relationship(a.id, b.id, "JOINS_WITH", properties={"anchor": "joined with B"})
    db.create_relationship(a.id, c.id, "LINKS_TO", properties={"anchor": "see C"})
    db.export_okf_bundle(str(tmp_path))
    _, body = _read(tmp_path / "a.md")
    assert body.index("# Links") < body.index("# Joins with")  # default section first
    assert "[joined with B](/b.md)" in body.split("# Joins with")[1]

    # Round-trip: typed re-import recovers the same relationship types.
    db2 = GrafitoDatabase(":memory:")
    db2.import_okf_bundle(str(tmp_path), typed_links=True, configure_fts=False)
    rows = db2.execute("MATCH (x {title: 'A'})-[r]->(y) RETURN type(r) AS t, y.title AS n")
    assert {(row["t"], row["n"]) for row in rows} == {("JOINS_WITH", "B"), ("LINKS_TO", "C")}
    db2.close()


# --- log.md synthesis -----------------------------------------------------------


def test_write_log_groups_by_scope_and_date(db, tmp_path):
    db.create_node(labels=["Note"], properties={"title": "A"}, uri="okf:decisions/a")
    for date, kind, text, scope in [
        ("2026-07-01", "Creation", "**Creation**: Set up.", ""),
        ("2026-07-03", "Update", "**Update**: Tweaked [A](/decisions/a.md).", "decisions"),
        ("2026-07-02", "Update", "**Update**: Older tweak.", "decisions"),
    ]:
        db.create_node(
            labels=["LogEntry"],
            properties={"date": date, "kind": kind, "text": text, "scope": scope, "log": True},
        )
    summary = db.export_okf_bundle(str(tmp_path))
    assert summary["logs"] == 2
    root_log = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "## 2026-07-01" in root_log and "Set up." in root_log
    scoped = (tmp_path / "decisions" / "log.md").read_text(encoding="utf-8")
    assert scoped.index("## 2026-07-03") < scoped.index("## 2026-07-02")  # newest first


def test_write_log_never_blanks_existing_log(db, tmp_path):
    (tmp_path / "log.md").write_text("# Log\n## 2026-01-01\n* **Update**: history\n")
    db.create_node(labels=["Note"], properties={"title": "A"}, uri="okf:a")
    summary = db.export_okf_bundle(str(tmp_path))  # no LogEntry nodes in the graph
    assert summary["logs"] == 0
    assert "history" in (tmp_path / "log.md").read_text(encoding="utf-8")


def test_prune_off_by_default(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    db.export_okf_bundle(str(tmp_path))
    orders = next(n for n in db.match_nodes() if n.uri == "okf:tables/orders")
    db.delete_node(orders.id)
    summary = db.export_okf_bundle(str(tmp_path))
    assert summary["pruned"] == 0
    assert (tmp_path / "tables" / "orders.md").exists()
