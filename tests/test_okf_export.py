from __future__ import annotations

from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.integrations import export_okf_bundle
from grafito.importers.okf import parse_frontmatter

pytest.importorskip("yaml")

BUNDLE = Path("examples") / "okf_bundle"


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
