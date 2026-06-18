from __future__ import annotations

from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.importers.okf import extract_links, parse_frontmatter

pytest.importorskip("yaml")

BUNDLE = Path("examples") / "okf_bundle"


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
