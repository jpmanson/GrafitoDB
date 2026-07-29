from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from grafito import GrafitoDatabase
from grafito.embedding_functions import EmbeddingFunction
from grafito.importers.okf import (
    classify_source,
    extract_citations,
    extract_links,
    generated_at,
    is_stale,
    normalize_sources,
    normalize_verified,
    parse_frontmatter,
    parse_log_entries,
    split_citations,
    trust_tier,
    verified_at,
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


# --- Provenance: the `sources` frontmatter (OKF v0.2 sec. 5.1) ----------------


SOURCES_DOC = """---
type: Doc
title: Doc
sources:
  - id: policy
    resource: https://wiki.acme/policy
    title: Revenue policy
    author: team:finance
    last_modified: 2026-04-02
  - id: dash
    resource: dashboards/exec-revenue
    title: Executive dashboard
    usage_count: 5000
  - id: sibling
    resource: /other.md
    title: The other concept
  - id: everything
    resource: all queries in BigQuery project acme
usage_window: { from: 2026-06-01, to: 2026-06-30 }
---

Body text.
"""


@pytest.fixture
def sources_bundle(tmp_path) -> Path:
    (tmp_path / "doc.md").write_text(SOURCES_DOC, encoding="utf-8")
    (tmp_path / "other.md").write_text("---\ntype: Doc\ntitle: Other\n---\ny\n", encoding="utf-8")
    return tmp_path


def test_normalize_sources_applies_shared_usage_window():
    entries = normalize_sources(
        {
            "sources": [
                {"resource": "https://a", "usage_count": 1},
                {"resource": "https://b", "usage_window": {"from": "2026-01-01"}},
            ],
            "usage_window": {"from": "2026-06-01", "to": "2026-06-30"},
        }
    )
    assert entries[0]["usage_window"] == {"from": "2026-06-01", "to": "2026-06-30"}
    # An entry's own window overrides the shared sibling.
    assert entries[1]["usage_window"] == {"from": "2026-01-01"}


def test_normalize_sources_is_permissive():
    # A lone entry may be written as a bare mapping; malformed entries (not a
    # mapping, or no `resource`) are dropped without costing the good ones.
    assert normalize_sources({"sources": {"resource": "https://a"}}) == [{"resource": "https://a"}]
    assert normalize_sources({"sources": [{"title": "no resource"}, "junk"]}) == []
    assert normalize_sources({}) == []
    assert normalize_sources({"sources": "not a list"}) == []


def test_classify_source_tells_scope_descriptors_from_paths():
    assert classify_source("https://a/b", "doc") == ("external", "https://a/b")
    assert classify_source("/tables/orders.md", "doc") == ("concept", "tables/orders")
    assert classify_source("../other.md", "a/doc") == ("concept", "other")
    # A population descriptor is not followable: whitespace is the only marker.
    assert classify_source("all queries in project X", "doc") == (
        "scope",
        "all queries in project X",
    )


def test_sources_create_citation_edges(db, sources_bundle):
    summary = db.import_okf_bundle(str(sources_bundle), configure_fts=False)
    assert summary["sources"] == 4
    assert summary["citations"] == 4  # every citation edge, whatever its form
    # An intra-bundle source resolves to the concept itself, not a Reference.
    concepts = db.execute(
        "MATCH (a {title: 'Doc'})-[:CITES]->(t) WHERE t.concept_id = 'other' RETURN t"
    )
    assert len(concepts) == 1
    # A URL, a followable artifact, and a scope descriptor all become References.
    refs = db.execute("MATCH (a {title: 'Doc'})-[:CITES]->(r:Reference) RETURN r.resource AS res")
    assert {row["res"] for row in refs} == {
        "https://wiki.acme/policy",
        "dashboards/exec-revenue",
        "all queries in BigQuery project acme",
    }


def test_source_paths_do_not_create_stub_concepts(db, tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\nsources:\n"
        "  - resource: references/attesters/revenue.py\n"
        "  - resource: /not-written-yet.md\n---\nx\n",
        encoding="utf-8",
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False)
    # A `.md` path is a concept that does not exist yet, so it gets a stub; a
    # path to something that is not a concept becomes a Reference instead.
    assert summary["stubs"] == 1
    assert summary["references"] == 1
    stub = db.execute("MATCH (n) WHERE n.stub = true RETURN n.concept_id AS cid")
    assert stub[0]["cid"] == "not-written-yet"


def test_source_signals_land_on_the_edge(db, sources_bundle):
    db.import_okf_bundle(str(sources_bundle), configure_fts=False)
    rows = db.execute(
        "MATCH (a {title: 'Doc'})-[r:CITES]->(t) WHERE r.source_id = 'dash' RETURN r"
    )
    edge = rows[0]["r"]["properties"]
    assert edge["anchor"] == "Executive dashboard"
    assert edge["usage_count"] == 5000
    # The shared sibling window frames every entry's count (sec. 5.1).
    assert edge["usage_window"] == {"from": "2026-06-01", "to": "2026-06-30"}
    assert edge["via"] == "sources"


def test_sources_respect_the_citations_switch(db, sources_bundle):
    summary = db.import_okf_bundle(str(sources_bundle), citations=False, configure_fts=False)
    assert summary["sources"] == 0
    assert summary["citations"] == 0
    # The frontmatter itself is still kept as a property.
    doc = _node_by_uri(db, "okf:doc")
    assert len(doc.properties["sources"]) == 4


def test_legacy_citations_and_sources_coexist(db, tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\nsources:\n  - resource: https://a/new\n---\n"
        "x\n\n# Citations\n- https://a/old\n",
        encoding="utf-8",
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False)
    assert summary["citations"] == 2 and summary["sources"] == 1
    rows = db.execute("MATCH (a {title: 'Doc'})-[r:CITES]->(t) RETURN r.via AS via, t.url AS url")
    assert {(row["via"], row["url"]) for row in rows} == {
        ("sources", "https://a/new"),
        ("citations", "https://a/old"),
    }


# --- Trust and lifecycle (OKF v0.2 sec. 5.2-5.5) ------------------------------


def test_verified_bare_mapping_reads_as_one_element_list():
    # A consumer MUST treat a bare mapping as a one-element list (sec. 11).
    bare = normalize_verified({"verified": {"by": "human:jp", "at": "2026-06-25T09:00:00Z"}})
    assert bare == [{"by": "human:jp", "at": "2026-06-25T09:00:00Z"}]
    listed = normalize_verified({"verified": [{"by": "human:jp"}, {"by": "process:nightly"}]})
    assert listed == [{"by": "human:jp"}, {"by": "process:nightly"}]
    # `by` is required within an event; `at` is not.
    assert normalize_verified({"verified": [{"at": "2026-06-25"}, "junk"]}) == []
    assert normalize_verified({}) == []


def test_trust_tier_keys_off_the_human_actor_prefix():
    assert trust_tier({}) == "unverified"
    assert trust_tier({"verified": [{"by": "process:finance-nightly"}]}) == "machine-confirmed"
    assert trust_tier({"verified": [{"by": "agent/gemini-2.5-pro"}]}) == "machine-confirmed"
    # One human among several verifiers lifts the whole concept (sec. 5.3).
    assert (
        trust_tier({"verified": [{"by": "process:nightly"}, {"by": "human:ahormati"}]})
        == "human-reviewed"
    )


def test_verified_at_is_the_most_recent_check():
    frontmatter = {
        "verified": [
            {"by": "human:jp", "at": "2026-06-25T09:00:00Z"},
            {"by": "process:nightly", "at": "2026-06-26T02:00:00Z"},
        ]
    }
    assert verified_at(frontmatter) == "2026-06-26T02:00:00Z"
    assert verified_at({"verified": [{"by": "human:jp"}]}) is None


def test_generated_at_falls_back_to_legacy_timestamp():
    assert generated_at({"generated": {"by": "agent/x", "at": "2026-06-28T14:00:00Z"}}) == (
        "2026-06-28T14:00:00Z"
    )
    # v0.1 documents (sec. 13.1): `timestamp` still answers "when did this change".
    assert generated_at({"timestamp": "2026-05-28T00:00:00Z"}) == "2026-05-28T00:00:00Z"
    # `generated` without an `at` is not a reason to ignore a usable timestamp.
    assert generated_at({"generated": {"by": "agent/x"}, "timestamp": "2026-05-28"}) == "2026-05-28"
    assert generated_at({}) is None


def test_is_stale_compares_against_an_absolute_date():
    today = datetime.date(2026, 9, 23)
    assert is_stale({"stale_after": "2026-09-23"}, today=today) is True  # stale *on* the day
    assert is_stale({"stale_after": "2026-09-24"}, today=today) is False
    assert is_stale({"stale_after": datetime.date(2026, 1, 1)}, today=today) is True
    # Staleness is asserted by the producer, never inferred from silence.
    assert is_stale({}, today=today) is False
    assert is_stale({"stale_after": "not a date"}, today=today) is False


def test_trust_fields_survive_the_graph_round_trip(db, tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\ntype: Doc\ntitle: Doc\nstatus: stable\n"
        "generated: { by: agent/x, at: 2026-06-28T14:00:00Z }\n"
        "verified: { by: human:jp, at: 2026-06-25T09:00:00Z }\n"
        "stale_after: 2026-12-31\n---\n\nBody.\n",
        encoding="utf-8",
    )
    db.import_okf_bundle(str(tmp_path), configure_fts=False)
    props = _node_by_uri(db, "okf:doc").properties
    # YAML parses these to date/datetime objects; grafito normalizes them to ISO
    # text on the way into the node, so the helpers must read either form.
    assert trust_tier(props) == "human-reviewed"
    assert generated_at(props) == "2026-06-28T14:00:00Z"
    assert is_stale(props, today=datetime.date(2027, 1, 1)) is True


def test_validate_warns_about_unreadable_trust_fields(tmp_path):
    from grafito.okf import validate_okf_bundle

    (tmp_path / "doc.md").write_text(
        "---\ntype: Doc\nstatus: approved\ngenerated: { at: 2026-06-28T14:00:00Z }\n"
        "verified:\n  - { at: 2026-06-25T09:00:00Z }\nstale_after: soon\n---\nx\n",
        encoding="utf-8",
    )
    report = validate_okf_bundle(str(tmp_path))
    assert report["conformant"] is True  # optional families never block (sec. 11)
    warnings = [w["warning"] for w in report["warnings"]]
    assert "'status' value 'approved' is not one of ['draft', 'stable', 'deprecated']" in warnings
    assert "'generated' is missing the required field: by" in warnings
    assert "verified[0] is missing the required field: by" in warnings
    assert "'stale_after' value 'soon' is not a YYYY-MM-DD date" in warnings


# --- Attested computations (OKF v0.2 sec. 10) ---------------------------------


COMPUTATION_DOC = """---
type: Attested Computation
title: Revenue for fiscal year
runtime: bigquery
parameters:
  - { name: year, type: integer, required: true }
executor:
  resource: references/skills/run-on-bq.md
  receipt: [job_id, executed_sql, result]
attester:
  resource: references/attesters/revenue.py
---

# Computation

    SELECT SUM(amount) AS revenue WHERE fiscal_year = @year
"""


@pytest.fixture
def computation_bundle(tmp_path) -> Path:
    (tmp_path / "computations").mkdir()
    (tmp_path / "computations" / "revenue.md").write_text(COMPUTATION_DOC, encoding="utf-8")
    (tmp_path / "references" / "skills").mkdir(parents=True)
    (tmp_path / "references" / "skills" / "run-on-bq.md").write_text(
        "---\ntype: Skill\ntitle: Run on BigQuery\n---\nRun it.\n", encoding="utf-8"
    )
    return tmp_path


def test_computation_paths_only_returns_real_paths():
    from grafito.importers.okf import computation_paths

    assert computation_paths(
        {
            "computation": "references/computations/revenue.sql",
            "executor": {"resource": "references/skills/run.md", "receipt": ["job_id"]},
            "attester": {"resource": "references/attesters/revenue.py"},
        }
    ) == [
        ("computation", "references/computations/revenue.sql"),
        ("executor", "references/skills/run.md"),
        ("attester", "references/attesters/revenue.py"),
    ]
    # An executor with no resource, and a type with none of the fields.
    assert computation_paths({"executor": {"receipt": ["job_id"]}}) == []
    assert computation_paths({"type": "Metric"}) == []


def test_computation_fields_become_edges(db, computation_bundle):
    summary = db.import_okf_bundle(str(computation_bundle), configure_fts=False)
    assert summary["computations"] == 2  # executor + attester
    rows = db.execute(
        "MATCH (c {title: 'Revenue for fiscal year'})-[r]->(t) "
        "WHERE type(r) <> 'CITES' RETURN type(r) AS rel, coalesce(t.title, t.resource) AS dst"
    )
    assert {(row["rel"], row["dst"]) for row in rows} == {
        ("EXECUTED_BY", "Run on BigQuery"),
        ("ATTESTED_BY", "references/attesters/revenue.py"),
    }


def test_executor_path_resolves_from_the_bundle_root(db, computation_bundle):
    # `references/skills/run-on-bq.md` is written from `computations/revenue.md`
    # but lives at the bundle root — the convention the SPEC's own example uses
    # (sec. 6.3). Reading it only relative to the citing concept would strand it
    # on a stub at `computations/references/skills/run-on-bq`.
    summary = db.import_okf_bundle(str(computation_bundle), configure_fts=False)
    assert summary["stubs"] == 0
    rows = db.execute(
        "MATCH (c {title: 'Revenue for fiscal year'})-[:EXECUTED_BY]->(t) "
        "RETURN t.concept_id AS cid, t.stub AS stub"
    )
    assert rows == [{"cid": "references/skills/run-on-bq", "stub": None}]


def test_non_concept_computation_paths_become_references(db, computation_bundle):
    db.import_okf_bundle(str(computation_bundle), configure_fts=False)
    # A `.py` attester is followable but is not a concept: Reference, not stub.
    refs = db.execute("MATCH (r:Reference) RETURN r.resource AS res")
    assert [row["res"] for row in refs] == ["references/attesters/revenue.py"]


def test_inline_computation_creates_no_edge(db, tmp_path):
    (tmp_path / "doc.md").write_text(
        "---\ntype: Attested Computation\ntitle: Inline\nruntime: python\n"
        "computation: sum the amounts booked to the year\n---\n\n# Computation\n\n    x = 1\n",
        encoding="utf-8",
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False)
    # `computation` here is prose, not a path (sec. 10.3): it stays a property.
    assert summary["computations"] == 0
    assert summary["references"] == 0
    assert _node_by_uri(db, "okf:doc").properties["computation"].startswith("sum the")


def test_validate_warns_about_an_unusable_computation_contract(tmp_path):
    from grafito.okf import validate_okf_bundle

    (tmp_path / "bad.md").write_text(
        "---\ntype: Attested Computation\ntitle: Bad\n"
        "parameters:\n  - { type: integer }\nexecutor: references/skills/run.md\n---\n\nNo fence.\n",
        encoding="utf-8",
    )
    report = validate_okf_bundle(str(tmp_path))
    assert report["conformant"] is True  # sec. 10 is a SHOULD, not a hard rule
    warnings = [w["warning"] for w in report["warnings"]]
    assert (
        "missing the required field: runtime (required for type 'Attested Computation')" in warnings
    )
    assert (
        "no computation: expected a 'computation' path or a '# Computation' body section"
        in warnings
    )
    assert "parameters[0] is missing the required field: name" in warnings
    assert "'executor' is not a mapping" in warnings


def test_computation_edges_do_not_duplicate_on_round_trip(db, computation_bundle, tmp_path):
    db.import_okf_bundle(str(computation_bundle), configure_fts=False)
    out = tmp_path / "out"
    db.export_okf_bundle(str(out))

    db2 = GrafitoDatabase(":memory:")
    try:
        summary = db2.import_okf_bundle(str(out), configure_fts=False)
        # The fields are re-read from frontmatter, never also written into the
        # body — so exactly the same two edges come back.
        assert summary["computations"] == 2
        assert db2.get_relationship_count() == db.get_relationship_count()
    finally:
        db2.close()


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


def test_validate_warns_about_malformed_sources(tmp_path):
    from grafito.okf import validate_okf_bundle

    (tmp_path / "doc.md").write_text(
        "---\ntype: Doc\nsources:\n  - title: no resource here\n  - resource: https://ok\n---\nx\n",
        encoding="utf-8",
    )
    report = validate_okf_bundle(str(tmp_path))
    # Optional families never make a bundle non-conformant (sec. 11), but an
    # entry naming no resource is dropped on import, so it is worth reporting.
    assert report["conformant"] is True
    warnings = [w["warning"] for w in report["warnings"] if w["path"] == "doc.md"]
    assert warnings == ["sources[0] is missing the required field: resource"]


def test_validate_missing_bundle_raises(tmp_path):
    from grafito.okf import validate_okf_bundle

    with pytest.raises(NotADirectoryError):
        validate_okf_bundle(str(tmp_path / "nope"))


# --- Layered linting: Core / Profile / Hygiene --------------------------------


def test_lint_core_layer_matches_validate_bundle():
    from grafito.okf import lint_okf_bundle, validate_okf_bundle

    validated = validate_okf_bundle(str(KB_BUNDLE))
    report = lint_okf_bundle(str(KB_BUNDLE))
    assert report["core"]["errors"] == validated["errors"]
    assert report["core"]["warnings"] == validated["warnings"]
    assert report["files"] == validated["files"]
    assert report["conformant"] is True
    assert report["profile"] == []


def test_lint_missing_bundle_raises(tmp_path):
    from grafito.okf import lint_okf_bundle

    with pytest.raises(NotADirectoryError):
        lint_okf_bundle(str(tmp_path / "nope"))


def test_lint_unknown_mode_raises(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lint_okf_bundle(str(tmp_path), mode="bogus")


def test_lint_hygiene_flags_missing_title_description_and_short_body(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "thin.md").write_text("---\ntype: Doc\n---\nToo short.\n", encoding="utf-8")
    report = lint_okf_bundle(str(tmp_path))
    rules = {(f["path"], f["rule"]) for f in report["hygiene"]}
    assert ("thin.md", "missing-title") in rules
    assert ("thin.md", "missing-description") in rules
    assert ("thin.md", "short-body") in rules


def test_lint_hygiene_flags_orphan_concept(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\ndescription: d\n---\n"
        "See [b](/b.md) for a much longer body than the short-body threshold requires here.\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "---\ntype: Doc\ntitle: B\ndescription: d\n---\n"
        "Linked from A, with a body long enough to dodge the short-body hygiene check too.\n",
        encoding="utf-8",
    )
    (tmp_path / "lonely.md").write_text(
        "---\ntype: Doc\ntitle: Lonely\ndescription: d\n---\n"
        "Nobody links here and this links nowhere, body padded past the threshold anyway.\n",
        encoding="utf-8",
    )
    report = lint_okf_bundle(str(tmp_path))
    orphans = {f["path"] for f in report["hygiene"] if f["rule"] == "orphan-concept"}
    assert orphans == {"lonely.md"}


def test_lint_hygiene_flags_duplicate_title(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: Same Title\ndescription: d\n---\n"
        "A body long enough to not trip the short-body hygiene rule on its own.\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "---\ntype: Doc\ntitle: Same Title\ndescription: d\n---\n"
        "Another body long enough to not trip the short-body hygiene rule either.\n",
        encoding="utf-8",
    )
    report = lint_okf_bundle(str(tmp_path))
    dupes = {f["path"] for f in report["hygiene"] if f["rule"] == "duplicate-title"}
    assert dupes == {"a.md", "b.md"}


def test_lint_validate_mode_omits_hygiene(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "thin.md").write_text("---\ntype: Doc\n---\nToo short.\n", encoding="utf-8")
    report = lint_okf_bundle(str(tmp_path), mode="validate")
    assert report["hygiene"] == []


def test_lint_profile_require_field_error_blocks_conformant(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "adr.md").write_text(
        "---\ntype: ADR\ntitle: A decision\n---\nBody.\n", encoding="utf-8"
    )
    profile = {
        "rules": [
            {"id": "adr-requires-status", "applies_to": "ADR", "require_field": "status", "severity": "error"}
        ]
    }
    report = lint_okf_bundle(str(tmp_path), profile=profile)
    assert report["conformant"] is False
    findings = [f for f in report["profile"] if f["rule"] == "adr-requires-status"]
    assert len(findings) == 1
    assert findings[0]["severity"] == "error"
    assert "missing required field: status" in findings[0]["message"]


def test_lint_profile_applies_to_filters_by_type(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "adr.md").write_text("---\ntype: ADR\ntitle: A\n---\nBody.\n", encoding="utf-8")
    (tmp_path / "note.md").write_text("---\ntype: Note\ntitle: N\n---\nBody.\n", encoding="utf-8")
    profile = {
        "rules": [{"id": "needs-status", "applies_to": "ADR", "require_field": "status"}]
    }
    report = lint_okf_bundle(str(tmp_path), profile=profile)
    flagged = {f["path"] for f in report["profile"]}
    assert flagged == {"adr.md"}


def test_lint_profile_severity_warning_does_not_block_conformant(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody.\n", encoding="utf-8")
    profile = {"rules": [{"id": "wants-tags", "require_field": "tags", "severity": "warning"}]}
    report = lint_okf_bundle(str(tmp_path), profile=profile)
    assert report["conformant"] is True
    assert len(report["profile"]) == 1


def test_lint_profile_from_yaml_file(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody.\n", encoding="utf-8")
    manifest = tmp_path / "profile.yaml"
    manifest.write_text(
        "rules:\n  - id: wants-tags\n    require_field: tags\n    severity: warning\n",
        encoding="utf-8",
    )
    report = lint_okf_bundle(str(tmp_path), profile=str(manifest))
    assert len(report["profile"]) == 1
    assert report["profile"][0]["rule"] == "wants-tags"


def test_lint_profile_missing_id_raises(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lint_okf_bundle(str(tmp_path), profile={"rules": [{"require_field": "tags"}]})


def test_lint_profile_max_length_allowed_values_and_pattern(tmp_path):
    from grafito.okf import lint_okf_bundle

    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A very very long title that exceeds the limit\n"
        "status: draft\n---\nBody.\n",
        encoding="utf-8",
    )
    profile = {
        "rules": [
            {"id": "title-len", "field": "title", "max_length": 10, "severity": "warning"},
            {
                "id": "status-enum",
                "field": "status",
                "allowed_values": ["accepted", "rejected"],
                "severity": "warning",
            },
            {"id": "title-pattern", "field": "title", "pattern": r"^\d+", "severity": "warning"},
        ]
    }
    report = lint_okf_bundle(str(tmp_path), profile=profile)
    rules = {f["rule"] for f in report["profile"]}
    assert rules == {"title-len", "status-enum", "title-pattern"}


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


# --- Obsidian wikilinks -------------------------------------------------------


def test_parse_wikilink_variants():
    from grafito.importers.okf import parse_wikilink

    assert parse_wikilink("Target") == ("Target", None)
    assert parse_wikilink("Target|Alias") == ("Target", "Alias")
    assert parse_wikilink("Target#Heading") == ("Target", None)
    assert parse_wikilink("Target#Heading|Alias") == ("Target", "Alias")


def test_resolve_wikilink_exact_path_and_basename_and_ambiguous():
    from grafito.importers.okf import resolve_wikilink

    concept_ids = {"decisions/0001-use-sqlite", "glossary/cypher", "notes/cypher"}
    basename_index = {"0001-use-sqlite": ["decisions/0001-use-sqlite"], "cypher": ["glossary/cypher", "notes/cypher"]}

    assert resolve_wikilink("decisions/0001-use-sqlite", concept_ids, basename_index) == "decisions/0001-use-sqlite"
    assert resolve_wikilink("0001-use-sqlite", concept_ids, basename_index) == "decisions/0001-use-sqlite"
    assert resolve_wikilink("cypher", concept_ids, basename_index) is None  # ambiguous
    assert resolve_wikilink("nope", concept_ids, basename_index) is None  # not found


def test_wikilinks_off_by_default(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [[b]] for details.\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nx\n", encoding="utf-8")
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False)
    assert summary["relationships"] == 0
    assert summary["stubs"] == 0


def test_wikilink_resolves_by_basename(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [[b]] for details.\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nx\n", encoding="utf-8")
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, wikilinks=True)
    assert summary["stubs"] == 0
    rows = db.execute("MATCH (a {title: 'A'})-[:LINKS_TO]->(b) RETURN b.title AS t")
    assert rows[0]["t"] == "B"


def test_wikilink_resolves_by_exact_concept_id(db, tmp_path):
    sub = tmp_path / "notes"
    sub.mkdir()
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [[notes/b]].\n", encoding="utf-8"
    )
    (sub / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nx\n", encoding="utf-8")
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, wikilinks=True)
    assert summary["stubs"] == 0
    rows = db.execute("MATCH (a {title: 'A'})-[:LINKS_TO]->(b) RETURN b.title AS t")
    assert rows[0]["t"] == "B"


def test_wikilink_alias_becomes_anchor(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [[b|Beta]].\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nx\n", encoding="utf-8")
    db.import_okf_bundle(str(tmp_path), configure_fts=False, wikilinks=True)
    rows = db.execute("MATCH (a {title: 'A'})-[r:LINKS_TO]->(b) RETURN r.anchor AS anchor")
    assert rows[0]["anchor"] == "Beta"


def test_wikilink_heading_fragment_stripped(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [[b#Some Section]].\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nx\n", encoding="utf-8")
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, wikilinks=True)
    assert summary["stubs"] == 0
    rows = db.execute("MATCH (a {title: 'A'})-[:LINKS_TO]->(b) RETURN b.title AS t")
    assert rows[0]["t"] == "B"


def test_wikilink_unresolved_creates_stub(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [[Future Note]].\n", encoding="utf-8"
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, wikilinks=True)
    assert summary["stubs"] == 1
    stub = _node_by_uri(db, "okf:Future Note")
    assert stub.properties.get("stub") is True


def test_wikilink_ambiguous_basename_skipped(db, tmp_path):
    sub1, sub2 = tmp_path / "one", tmp_path / "two"
    sub1.mkdir()
    sub2.mkdir()
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [[dup]].\n", encoding="utf-8"
    )
    (sub1 / "dup.md").write_text("---\ntype: Doc\ntitle: Dup1\n---\nx\n", encoding="utf-8")
    (sub2 / "dup.md").write_text("---\ntype: Doc\ntitle: Dup2\n---\nx\n", encoding="utf-8")
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, wikilinks=True)
    assert summary["relationships"] == 0
    assert summary["stubs"] == 0


def test_wikilinks_excluded_from_citations_section(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nx\n\n# Citations\n[[b]]\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nx\n", encoding="utf-8")
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, wikilinks=True)
    assert summary["relationships"] == 0
    assert summary["stubs"] == 0


def test_wikilinks_respect_typed_links(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Table\ntitle: A\n---\n# Joins with\n\n[[b]] on id.\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text("---\ntype: Table\ntitle: B\n---\nx\n", encoding="utf-8")
    db.import_okf_bundle(str(tmp_path), configure_fts=False, wikilinks=True, typed_links=True)
    rows = db.execute("MATCH (a {title: 'A'})-[r]->(b) RETURN type(r) AS t")
    assert rows[0]["t"] == "JOINS_WITH"


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


# --- Incremental import (content-hash skip / update-in-place) ----------------


class _CountingEmbedder(EmbeddingFunction):
    """Trivial embedder that counts how many documents it was asked to embed."""

    def __init__(self) -> None:
        self.total_docs = 0

    def __call__(self, input: list[str]) -> list[list[float]]:
        self.total_docs += len(input)
        return [[1.0, 0.0] for _ in input]

    @staticmethod
    def name() -> str:
        return "counting_test_embedder"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    @staticmethod
    def build_from_config(config: dict) -> "_CountingEmbedder":
        return _CountingEmbedder()

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    @property
    def dimension(self) -> int:
        return 2


def test_incremental_second_import_skips_unchanged(db, tmp_path):
    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody A.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nBody B.\n", encoding="utf-8")

    first = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert first["nodes"] == 2
    assert first["unchanged"] == 0
    a_id = _node_by_uri(db, "okf:a").id
    b_id = _node_by_uri(db, "okf:b").id

    second = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert second["nodes"] == 0
    assert second["updated"] == 0
    assert second["unchanged"] == 2
    assert db.get_node_count() == 2
    assert _node_by_uri(db, "okf:a").id == a_id
    assert _node_by_uri(db, "okf:b").id == b_id


def test_incremental_updates_changed_file_in_place(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [b](/b.md).\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nBody B.\n", encoding="utf-8")
    db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    a_id = _node_by_uri(db, "okf:a").id
    b_id = _node_by_uri(db, "okf:b").id

    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A Updated\n---\nSee [b](/b.md) again.\n", encoding="utf-8"
    )
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert summary["updated"] == 1
    assert summary["unchanged"] == 1
    assert summary["nodes"] == 1

    a = _node_by_uri(db, "okf:a")
    assert a.id == a_id  # same node, updated in place
    assert a.properties["title"] == "A Updated"
    assert _node_by_uri(db, "okf:b").id == b_id  # unchanged concept untouched

    links = db.match_relationships(source_id=a_id, rel_type="LINKS_TO")
    assert len(links) == 1  # stale relationship replaced, not duplicated
    assert links[0].target_id == b_id


def test_incremental_new_file_added(db, tmp_path):
    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody A.\n", encoding="utf-8")
    db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)

    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nBody B.\n", encoding="utf-8")
    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert summary["nodes"] == 1
    assert summary["unchanged"] == 1
    assert db.get_node_count() == 2


def test_incremental_preserves_trust_model_edges(db, tmp_path):
    (tmp_path / "old.md").write_text(
        "---\ntype: Doc\ntitle: Old\n---\nBody old.\n", encoding="utf-8"
    )
    (tmp_path / "new.md").write_text(
        "---\ntype: Doc\ntitle: New\n---\nBody new.\n", encoding="utf-8"
    )
    db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    old_id = _node_by_uri(db, "okf:old").id
    new_id = _node_by_uri(db, "okf:new").id
    db.create_relationship(new_id, old_id, "SUPERSEDES")

    (tmp_path / "new.md").write_text(
        "---\ntype: Doc\ntitle: New Updated\n---\nBody new updated.\n", encoding="utf-8"
    )
    db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)

    rels = db.match_relationships(source_id=new_id, rel_type="SUPERSEDES")
    assert len(rels) == 1
    assert rels[0].target_id == old_id


def test_incremental_promotes_existing_stub(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nSee [b](/b.md).\n", encoding="utf-8"
    )
    first = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert first["stubs"] == 1
    stub_id = _node_by_uri(db, "okf:b").id

    (tmp_path / "b.md").write_text(
        "---\ntype: Doc\ntitle: B\n---\nReal body B.\n", encoding="utf-8"
    )
    second = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert second["stubs"] == 0  # no *new* stub created; the existing one was promoted

    b = _node_by_uri(db, "okf:b")
    assert b.id == stub_id  # same node, promoted in place
    assert b.properties.get("stub") is None
    assert b.properties["title"] == "B"
    assert db.get_node_count() == 2  # no duplicate node for b

    links = db.match_relationships(rel_type="LINKS_TO")
    assert len(links) == 1
    assert links[0].target_id == stub_id


def test_incremental_prune_removes_missing_file_node(db, tmp_path):
    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody A.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nBody B.\n", encoding="utf-8")
    db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    (tmp_path / "b.md").unlink()

    summary = db.import_okf_bundle(
        str(tmp_path), configure_fts=False, incremental=True, prune=True
    )
    assert summary["pruned"] == 1
    assert db.get_node_count() == 1
    assert not [n for n in db.match_nodes() if n.uri == "okf:b"]


def test_incremental_without_prune_keeps_missing_file_node(db, tmp_path):
    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody A.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nBody B.\n", encoding="utf-8")
    db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    (tmp_path / "b.md").unlink()

    summary = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert summary["pruned"] == 0
    assert db.get_node_count() == 2


def test_prune_requires_incremental(db, tmp_path):
    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody A.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        db.import_okf_bundle(str(tmp_path), configure_fts=False, prune=True)


def test_incremental_reuses_reference_node_across_imports(db, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A\n---\nx\n\n# Citations\n- https://example.com/spec\n",
        encoding="utf-8",
    )
    first = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert first["references"] == 1

    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A Updated\n---\nx\n\n# Citations\n- https://example.com/spec\n",
        encoding="utf-8",
    )
    second = db.import_okf_bundle(str(tmp_path), configure_fts=False, incremental=True)
    assert second["references"] == 0  # reused, not duplicated
    assert second["updated"] == 1
    refs = [n for n in db.match_nodes() if "Reference" in n.labels]
    assert len(refs) == 1


def test_incremental_skips_reembedding_unchanged(db, tmp_path):
    (tmp_path / "a.md").write_text("---\ntype: Doc\ntitle: A\n---\nBody A.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\ntype: Doc\ntitle: B\n---\nBody B.\n", encoding="utf-8")

    embedder = _CountingEmbedder()
    first = db.import_okf_bundle(
        str(tmp_path), configure_fts=False, incremental=True, embed=embedder
    )
    assert first["embedded"] == 2
    assert embedder.total_docs == 2

    (tmp_path / "a.md").write_text(
        "---\ntype: Doc\ntitle: A Updated\n---\nBody A updated.\n", encoding="utf-8"
    )
    second = db.import_okf_bundle(
        str(tmp_path), configure_fts=False, incremental=True, embed=embedder
    )
    assert second["embedded"] == 1  # only the changed concept was re-embedded
    assert second["unchanged"] == 1
    assert embedder.total_docs == 3  # 2 from the first import + 1 re-embed


# --- diff_okf_bundles ---------------------------------------------------------


def _write_bundle(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_diff_detects_added_removed_changed(tmp_path):
    from grafito.okf import diff_okf_bundles

    base = _write_bundle(
        tmp_path / "base",
        {
            "keep.md": "---\ntype: Doc\ntitle: Keep\n---\nUnchanged body.\n",
            "edit.md": "---\ntype: Doc\ntitle: Edit\n---\nOld body.\n",
            "gone.md": "---\ntype: Doc\ntitle: Gone\n---\nBye.\n",
        },
    )
    cand = _write_bundle(
        tmp_path / "cand",
        {
            "keep.md": "---\ntype: Doc\ntitle: Keep\n---\nUnchanged body.\n",
            "edit.md": "---\ntype: Doc\ntitle: Edit\n---\nNew body.\n",
            "new.md": "---\ntype: Doc\ntitle: New\n---\nHello.\n",
        },
    )

    diff = diff_okf_bundles(base, cand)
    assert diff.added == ["new.md"]
    assert diff.removed == ["gone.md"]
    assert list(diff.changed) == ["edit.md"]
    assert "keep.md" not in diff.changed  # identical content is not a change
    assert diff.has_changes is True
    assert diff.summary() == {
        "added": 1,
        "removed": 1,
        "changed": 1,
        "invalid": 0,
        "broken_links": 0,
    }


def test_diff_reports_field_level_delta(tmp_path):
    from grafito.okf import diff_okf_bundles

    base = _write_bundle(
        tmp_path / "base",
        {"c.md": "---\ntype: ADR\ntitle: Old\nstatus: draft\n---\nBody.\n"},
    )
    cand = _write_bundle(
        tmp_path / "cand",
        {"c.md": "---\ntype: ADR\ntitle: New\nowner: jp\n---\nBody changed.\n"},
    )

    delta = diff_okf_bundles(base, cand).changed["c.md"]
    assert delta.body_changed is True
    assert delta.frontmatter_changed == {"title": ("Old", "New")}
    assert delta.frontmatter_added == {"owner": "jp"}
    assert delta.frontmatter_removed == {"status": "draft"}


def test_diff_type_change_shows_as_frontmatter_delta(tmp_path):
    from grafito.okf import diff_okf_bundles

    base = _write_bundle(tmp_path / "base", {"c.md": "---\ntype: Doc\n---\nBody.\n"})
    cand = _write_bundle(tmp_path / "cand", {"c.md": "---\ntype: ADR\n---\nBody.\n"})

    delta = diff_okf_bundles(base, cand).changed["c.md"]
    assert delta.body_changed is False
    assert delta.frontmatter_changed == {"type": ("Doc", "ADR")}


def test_diff_surfaces_invalid_and_broken_links_in_candidate(tmp_path):
    from grafito.okf import diff_okf_bundles

    base = _write_bundle(tmp_path / "base", {"a.md": "---\ntype: Doc\n---\nBody.\n"})
    cand = _write_bundle(
        tmp_path / "cand",
        {
            "a.md": "---\ntype: Doc\n---\nSee [x](/nowhere.md).\n",
            "no-type.md": "---\ntitle: X\n---\nBody.\n",
        },
    )

    diff = diff_okf_bundles(base, cand)
    assert "missing or empty required field: type" in diff.invalid["no-type.md"]
    assert ("a.md", "nowhere") in diff.broken_links
    assert diff.conformant is False  # a hard conformance error is present


def test_diff_ignores_reserved_files(tmp_path):
    from grafito.okf import diff_okf_bundles

    base = _write_bundle(
        tmp_path / "base",
        {
            "c.md": "---\ntype: Doc\n---\nBody.\n",
            "log.md": "## 2026-01-01\n- **Creation**: c.\n",
            "index.md": "---\nokf_version: '0.1'\n---\n# Index\n",
        },
    )
    cand = _write_bundle(
        tmp_path / "cand",
        {
            "c.md": "---\ntype: Doc\n---\nBody.\n",
            "log.md": "## 2026-02-02\n- **Update**: changed log only.\n",
            "index.md": "---\nokf_version: '0.9'\n---\n# Different index\n",
        },
    )

    diff = diff_okf_bundles(base, cand)
    assert diff.has_changes is False  # only reserved files differ
    assert diff.summary()["changed"] == 0


def test_diff_hash_matches_importer_okf_hash(db, tmp_path):
    """The diff's content hash is byte-for-byte the importer's ``okf_hash``."""
    from grafito.importers.okf import _concept_hashes

    _write_bundle(
        tmp_path,
        {
            "a.md": "---\ntype: Doc\ntitle: A\n---\nBody A.\n",
            "sub/b.md": "---\ntype: Doc\ntitle: B\n---\nBody B.\n",
        },
    )
    db.import_okf_bundle(str(tmp_path), configure_fts=False)

    hashes = _concept_hashes(tmp_path)
    for node in db.match_nodes():
        cid = node.properties.get("concept_id")
        okf_hash = node.properties.get("okf_hash")
        if cid and okf_hash:
            assert hashes[f"{cid}.md"] == okf_hash


def test_diff_missing_bundle_raises(tmp_path):
    from grafito.okf import diff_okf_bundles

    real = _write_bundle(tmp_path / "real", {"c.md": "---\ntype: Doc\n---\nBody.\n"})
    with pytest.raises(NotADirectoryError):
        diff_okf_bundles(tmp_path / "nope", real)
    with pytest.raises(NotADirectoryError):
        diff_okf_bundles(real, tmp_path / "nope")
