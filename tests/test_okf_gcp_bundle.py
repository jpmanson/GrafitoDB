"""Verify GrafitoDB consumes a real OKF bundle from GoogleCloudPlatform.

The fixture under ``tests/res/okf_gcp_ga4`` is a downloaded subset of the
``ga4`` bundle from
https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf
(tables/datasets/references with their index.md files). It exercises real-world
OKF conventions: free-form ``type`` values with spaces, deeply nested markdown
bodies, a ``references/`` tree, and links to concepts outside the subset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grafito import GrafitoDatabase

pytest.importorskip("yaml")

BUNDLE = Path("tests") / "res" / "okf_gcp_ga4"


@pytest.fixture
def db() -> GrafitoDatabase:
    database = GrafitoDatabase(":memory:")
    yield database
    database.close()


def test_fixture_present():
    assert (BUNDLE / "tables" / "events_.md").exists(), "ga4 fixture missing"


def test_imports_real_bundle(db):
    summary = db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    # 8 concept files in the subset; 6 index.md files skipped.
    assert summary["nodes"] == 8
    assert summary["skipped"] == 6
    assert summary["relationships"] > 0


def test_real_types_become_labels(db):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    labels = set(db.get_all_labels())
    assert {"BigQuery Table", "BigQuery Dataset", "Reference"} <= labels


def test_links_outside_subset_become_stubs(db):
    summary = db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    # The events_ table and references link to concepts not in the subset;
    # those resolve to stub nodes rather than failing (SPEC sec. 5.3).
    assert summary["stubs"] > 0


def test_events_table_links_resolve(db):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    rows = db.execute(
        """
        MATCH (a)-[:LINKS_TO]->(b)
        WHERE a.concept_id = 'tables/events_'
        RETURN count(*) AS c
        """
    )
    assert rows[0]["c"] > 0


def test_fts_over_real_bundle(db):
    if not db.has_fts5():
        pytest.skip("SQLite build lacks FTS5")
    db.import_okf_bundle(str(BUNDLE), configure_fts=True)
    hits = db.text_search("pageviews", k=5)
    assert hits


def test_real_bundle_roundtrip(db, tmp_path):
    db.import_okf_bundle(str(BUNDLE), configure_fts=False)
    db.export_okf_bundle(str(tmp_path))

    db2 = GrafitoDatabase(":memory:")
    try:
        db2.import_okf_bundle(str(tmp_path), configure_fts=False)
        # Non-stub concepts survive the round-trip (stubs are not re-exported).
        original = sum(
            1 for n in db.match_nodes() if not n.properties.get("stub")
        )
        assert db2.get_node_count() <= db.get_node_count()
        exported_concepts = [
            n for n in db2.match_nodes() if not n.properties.get("stub")
        ]
        assert len(exported_concepts) >= original
    finally:
        db2.close()
