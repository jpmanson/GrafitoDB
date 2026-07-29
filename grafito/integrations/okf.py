"""Open Knowledge Format (OKF) bundle exporter.

Serializes a GrafitoDB graph back into an OKF bundle: a directory tree of
markdown files with YAML frontmatter (see
https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf). This is the
inverse of :func:`grafito.importers.okf.import_bundle`.

Each node becomes one concept document:

- the first node label -> frontmatter ``type``
- node properties (minus internal keys) -> frontmatter
- ``CITES`` edges -> the ``sources`` frontmatter block (SPEC sec. 5.1)
- the ``body`` property -> markdown body (or a synthesized ``# Links`` section
  when no body is stored)
- the concept's file path is derived from its ``uri`` (``<uri_prefix><id>``)

A bundle-root ``index.md`` is generated for progressive disclosure (SPEC sec. 8).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import yaml

from ..importers.okf import (
    DEFAULT_LABEL,
    SOURCE_SIGNALS,
    VIA_CITATIONS,
    classify_source,
)

if TYPE_CHECKING:
    from ..database import GrafitoDatabase

# Properties that are GrafitoDB bookkeeping, not OKF frontmatter.
_INTERNAL_PROPS = ("body", "concept_id", "stub", "okf_hash")

# Recommended frontmatter order (SPEC sec. 4.1, then the trust and lifecycle
# families of sec. 5). `type` is always emitted first; the legacy v0.1
# `timestamp` trails the block it was superseded by (sec. 13.1).
_RECOMMENDED_ORDER = (
    "title",
    "description",
    "resource",
    "tags",
    "status",
    "generated",
    "verified",
    "stale_after",
    "timestamp",
)

# Provenance keys, emitted last and in this order (SPEC sec. 5.1 examples):
# the `sources` list, then the `usage_window` that frames its usage counts.
_PROVENANCE_ORDER = ("sources", "usage_window")

# Key order within one `sources` entry (SPEC sec. 5.1).
_SOURCE_ENTRY_ORDER = ("id", "resource", "title", *SOURCE_SIGNALS)


def _concept_id(node, uri_prefix: str) -> str:
    if node.uri and node.uri.startswith(uri_prefix):
        return node.uri[len(uri_prefix):]
    cid = node.properties.get("concept_id")
    if isinstance(cid, str) and cid:
        return cid
    return f"node-{node.id}"


def _ordered_frontmatter(
    db: "GrafitoDatabase", node, rels: list, concept_id: str, uri_prefix: str
) -> dict:
    """Build an ordered frontmatter dict: type, recommended keys, rest, provenance."""
    label = node.labels[0] if node.labels else DEFAULT_LABEL
    props = {k: v for k, v in node.properties.items() if k not in _INTERNAL_PROPS}
    provenance = {key: props.pop(key) for key in _PROVENANCE_ORDER if key in props}

    frontmatter: dict = {"type": label}
    for key in _RECOMMENDED_ORDER:
        if key in props:
            frontmatter[key] = props.pop(key)
    for key in sorted(props):
        frontmatter[key] = props[key]

    sources = _sources_frontmatter(
        db,
        rels,
        concept_id,
        uri_prefix,
        provenance.get("sources"),
        provenance.get("usage_window"),
    )
    if sources:
        provenance["sources"] = sources
    for key in _PROVENANCE_ORDER:
        if key in provenance:
            frontmatter[key] = provenance[key]
    return frontmatter


def _source_entry(rel, target, uri_prefix: str, shared_window) -> dict | None:
    """One ``sources`` entry from a ``CITES`` edge, or ``None`` if it names nothing.

    The edge carries what the entry said: ``source_id`` is the entry's ``id``
    (the footnote join key, SPEC sec. 5.1), ``anchor`` its ``title``, and the
    credibility signals are stored per edge because they describe *this*
    concept's use of the source, not the source in the abstract.
    """
    if target.properties.get("okf_auto"):
        # A Reference node: an external URL, a followable artifact, or a scope
        # descriptor — all of them keep their authored resource string.
        resource = target.properties.get("resource") or target.properties.get("url")
    else:
        # Another concept: the recommended bundle-relative form (sec. 6.1).
        resource = f"/{_concept_id(target, uri_prefix)}.md"
    if not resource:
        return None

    entry = {"resource": resource}
    if rel.properties.get("source_id"):
        entry["id"] = rel.properties["source_id"]
    if rel.properties.get("anchor"):
        entry["title"] = rel.properties["anchor"]
    for signal in SOURCE_SIGNALS:
        value = rel.properties.get(signal)
        # A window equal to the concept-wide one is implied by the sibling
        # `usage_window` key, so re-emitting it per entry would be noise.
        if value is not None and not (signal == "usage_window" and value == shared_window):
            entry[signal] = value
    return {key: entry[key] for key in _SOURCE_ENTRY_ORDER if key in entry}


def _sources_frontmatter(
    db: "GrafitoDatabase", rels: list, concept_id: str, uri_prefix: str, stored, shared_window
) -> list | None:
    """The concept's ``sources`` block: what it was authored with, plus new edges.

    The authored ``sources`` property (present on any imported v0.2 concept) is
    kept verbatim so a round-trip is byte-stable. ``CITES`` edges that no entry
    already covers — a programmatic ``cite()``, or an edge added straight to the
    graph — are appended, so provenance added through the graph reaches
    markdown. Edges from a legacy ``# Citations`` body list are skipped: that
    list is still in the body, and lifting it into frontmatter too would double
    every citation on the next import.
    """
    if stored is not None and not isinstance(stored, (list, dict)):
        return stored  # not a shape we can merge into; round-trip it untouched
    entries = [stored] if isinstance(stored, dict) else list(stored or [])
    # Resource identity, not string equality: an entry may name a concept as
    # `../x.md` where the synthesized form is `/x.md`.
    seen = {
        classify_source(entry["resource"], concept_id)
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("resource"), str)
    }

    for rel in rels:
        if rel.type != "CITES" or rel.properties.get("via") == VIA_CITATIONS:
            continue
        target = db.get_node(rel.target_id)
        if target is None:
            continue
        entry = _source_entry(rel, target, uri_prefix, shared_window)
        if entry is None:
            continue
        key = classify_source(entry["resource"], concept_id)
        if key not in seen:
            seen.add(key)
            entries.append(entry)
    return entries or None


def _heading_from_rel_type(rel_type: str) -> str:
    """Inverse of the importer's heading normalization (``JOINS_WITH`` -> ``Joins with``)."""
    return rel_type.replace("_", " ").capitalize()


def _synthesize_body(
    db: "GrafitoDatabase", rels: list, uri_prefix: str, links_heading: str
) -> str:
    """Synthesize a body for a node with none stored, from its outgoing edges.

    ``LINKS_TO`` edges become a ``# {links_heading}`` section; other typed
    edges each get a heading derived from their type (``JOINS_WITH`` ->
    ``# Joins with``), so bundles imported with ``typed_links=True`` round-trip.

    Only edges this function *owns* are written. An edge's ``via`` property
    records which frontmatter block produced it — ``sources`` (SPEC sec. 5.1),
    or a computation field such as ``attester`` (sec. 10.2) — and those are
    re-emitted from their own block, so writing them here too would duplicate
    every one of them on the next import. The single exception is
    ``via="citations"``: an edge from a legacy v0.1 ``# Citations`` body list,
    written back in the form it was authored in.
    """
    link_sections: dict[str, list[str]] = {}  # rel type -> lines
    cite_lines: list[str] = []
    for rel in rels:
        via = rel.properties.get("via")
        if rel.type == "CITES":
            if via != VIA_CITATIONS:
                continue  # v0.2 provenance: lives in frontmatter, not the body
        elif via:
            continue  # derived from a frontmatter field that re-emits it itself
        target = db.get_node(rel.target_id)
        if target is None:
            continue
        anchor = rel.properties.get("anchor")
        if rel.type == "CITES":
            if target.properties.get("okf_auto"):
                url = target.properties.get("url")
                cite_lines.append(f"- [{anchor}]({url})" if anchor else f"- {url}")
            else:
                tid = _concept_id(target, uri_prefix)
                label = anchor or target.properties.get("title") or tid
                cite_lines.append(f"- [{label}](/{tid}.md)")
        else:
            tid = _concept_id(target, uri_prefix)
            label = anchor or target.properties.get("title") or tid
            link_sections.setdefault(rel.type, []).append(f"- [{label}](/{tid}.md)")

    sections: list[str] = []
    # The default link type first (under the conventional heading), then the
    # typed sections alphabetically, citations last.
    for rel_type in sorted(link_sections, key=lambda t: (t != "LINKS_TO", t)):
        heading = links_heading if rel_type == "LINKS_TO" else _heading_from_rel_type(rel_type)
        sections.append(f"# {heading}\n\n" + "\n".join(link_sections[rel_type]) + "\n")
    if cite_lines:
        sections.append("# Citations\n\n" + "\n".join(cite_lines) + "\n")
    return "\n".join(sections)


def export_bundle(
    db: "GrafitoDatabase",
    root: str | Path,
    *,
    uri_prefix: str = "okf:",
    write_index: bool = True,
    write_viz: bool = False,
    write_log: bool = True,
    prune: bool = False,
    links_heading: str = "Links",
    okf_version: str | None = None,
) -> dict:
    """Export the graph in ``db`` to an OKF bundle directory at ``root``.

    Args:
        db: Source database.
        root: Destination bundle directory (created if missing).
        uri_prefix: Prefix used to recover concept IDs from node URIs. Should
            match the value used at import time.
        write_index: Generate per-directory ``index.md`` files for progressive
            disclosure (SPEC sec. 8), following the convention used by reference
            bundles: a root index linking to child directory indexes, and each
            directory index grouping its concepts by ``type``.
        okf_version: Declare the OKF version the bundle targets, written as
            ``okf_version`` frontmatter in the root ``index.md`` — the only
            index file allowed to carry frontmatter (SPEC sec. 12). ``None``
            (the default) writes no declaration; ``OKFBundle.save()`` passes
            through whatever the bundle declared when it was loaded, so a
            version survives a round-trip instead of being dropped.
        write_viz: Also emit a self-contained ``viz.html`` graph viewer at the
            bundle root (best-effort; mirrors the reference bundles).
        write_log: Regenerate per-scope ``log.md`` files (SPEC sec. 9) from the
            ``LogEntry`` nodes in the graph (present when the bundle was
            imported with ``import_log=True`` or entries were added via
            ``OKFBundle.log_entry``). Scopes without entries are left alone —
            an existing ``log.md`` is never deleted or blanked.
        prune: Delete concept ``.md`` files under ``root`` that no longer
            correspond to a node — so deleting a concept from the graph deletes
            its file on re-export. Reserved files (``log.md``), ``index.md``
            files, and non-markdown files are never touched; directories left
            empty are removed.
        links_heading: Heading used for the synthesized links section when a
            node has no stored ``body``.

    Returns:
        Summary dict: ``{"concepts", "skipped", "pruned", "logs", "viz"}``.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    concepts = 0
    skipped = 0
    written: set[Path] = set()  # concept files written this export
    # (concept_id, title, description, type) for index generation.
    index_entries: list[tuple[str, str, str, str]] = []

    for node in db.match_nodes():
        # Skip derived/unapproved nodes: stubs (broken links), auto Reference
        # nodes (external citations), Directory nodes, LogEntry nodes (all
        # re-synthesized on (re-)import), and concepts still awaiting review
        # via OKFBundle.propose(). None are written as their own concept file.
        if (
            node.properties.get("stub") is True
            or node.properties.get("okf_auto") is True
            or node.properties.get("directory") is True
            or node.properties.get("log") is True
            or node.properties.get("pending_review") is True
        ):
            skipped += 1
            continue

        concept_id = _concept_id(node, uri_prefix)
        # Fetched once: both the `sources` block and a synthesized body read the
        # concept's outgoing edges.
        rels = db.match_relationships(source_id=node.id)
        frontmatter = _ordered_frontmatter(db, node, rels, concept_id, uri_prefix)

        body = node.properties.get("body")
        if not isinstance(body, str) or not body.strip():
            body = _synthesize_body(db, rels, uri_prefix, links_heading)

        fm_text = yaml.safe_dump(
            frontmatter,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,  # block style throughout (matches real bundles)
        ).strip()
        document = f"---\n{fm_text}\n---\n\n{body}"
        if not document.endswith("\n"):
            document += "\n"

        file_path = root_path / PurePosixPath(f"{concept_id}.md")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(document, encoding="utf-8")
        written.add(file_path)
        concepts += 1

        index_entries.append(
            (
                concept_id,
                str(frontmatter.get("title") or concept_id),
                str(frontmatter.get("description") or ""),
                str(frontmatter["type"]),
            )
        )

    pruned = 0
    if prune:
        pruned = _prune_orphans(root_path, written)

    if write_index:
        _write_indexes(root_path, index_entries, okf_version)

    logs = 0
    if write_log:
        logs = _write_logs(db, root_path)

    viz_written = False
    if write_viz:
        viz_written = _write_viz(db, root_path)

    return {
        "concepts": concepts,
        "skipped": skipped,
        "pruned": pruned,
        "logs": logs,
        "viz": viz_written,
    }


def _write_logs(db: "GrafitoDatabase", root_path: Path) -> int:
    """Serialize ``LogEntry`` nodes to per-scope ``log.md`` files (SPEC sec. 9).

    Entries are grouped by scope (the directory the log belongs to) and date,
    newest date first; within a date, creation order is kept. Returns the number
    of ``log.md`` files written. Scopes without entries are not touched.
    """
    # scope -> date -> [(node_id, text)]
    scopes: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for node in db.match_nodes(properties={"log": True}):
        props = node.properties
        date, text = props.get("date"), props.get("text")
        if not isinstance(date, str) or not date or not isinstance(text, str) or not text:
            continue
        scope = props.get("scope") or ""
        scopes.setdefault(scope, {}).setdefault(date, []).append((node.id, text))

    for scope, by_date in scopes.items():
        lines = ["# Update Log", ""]
        for date in sorted(by_date, reverse=True):
            lines.append(f"## {date}")
            for _node_id, text in sorted(by_date[date]):
                lines.append(f"* {text}")
            lines.append("")
        target = root_path / scope / "log.md" if scope else root_path / "log.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return len(scopes)


def _prune_orphans(root_path: Path, written: set[Path]) -> int:
    """Delete concept files not written by this export; drop emptied directories.

    Only non-reserved ``.md`` files are candidates — ``index.md``/``log.md``
    and non-markdown files (images, ``viz.html``, ...) are never removed.
    """
    from ..importers.okf import RESERVED_FILENAMES

    pruned = 0
    for path in sorted(root_path.rglob("*.md")):
        if path.name in RESERVED_FILENAMES or path in written:
            continue
        path.unlink()
        pruned += 1
    # Remove directories the pruning emptied (deepest first). An index.md alone
    # does not keep a directory alive — it described the pruned concepts.
    for directory in sorted(
        (p for p in root_path.rglob("*") if p.is_dir()), reverse=True
    ):
        entries = list(directory.iterdir())
        if all(e.name == "index.md" for e in entries):
            for e in entries:
                e.unlink()
            directory.rmdir()
    return pruned


def _write_indexes(
    root_path: Path,
    entries: list[tuple[str, str, str, str]],
    okf_version: str | None = None,
) -> None:
    """Write one ``index.md`` per directory (recursive, SPEC sec. 8 convention).

    Each directory index lists its immediate child directories under a
    ``# Subdirectories`` heading and its own concepts grouped by ``type``.
    ``okf_version``, when given, is declared in the *root* index only — the one
    index file the SPEC lets carry frontmatter (sec. 12).
    """
    # dir -> {"subdirs": set[str], "concepts": list[(filename, title, desc, type)]}
    dirs: dict[str, dict] = {}

    def ensure(dirpath: str) -> dict:
        return dirs.setdefault(dirpath, {"subdirs": set(), "concepts": []})

    ensure(".")
    for concept_id, title, description, ctype in entries:
        path = PurePosixPath(concept_id)
        parent = path.parent.as_posix()
        if parent == ".":
            parent_dir = "."
        else:
            parent_dir = parent
        ensure(parent_dir)["concepts"].append((f"{path.name}.md", title, description, ctype))
        # Register the chain of ancestor -> child relationships.
        current = "."
        for part in path.parent.parts:
            child = part if current == "." else f"{current}/{part}"
            ensure(current)["subdirs"].add(child)
            ensure(child)
            current = child

    for dirpath, content in dirs.items():
        lines: list[str] = []
        if dirpath == "." and okf_version:
            lines.append("---")
            lines.append(f"okf_version: {yaml.safe_dump(str(okf_version)).strip()}")
            lines.append("---")
            lines.append("")
        if content["subdirs"]:
            lines.append("# Subdirectories")
            lines.append("")
            for child in sorted(content["subdirs"]):
                name = child.rsplit("/", 1)[-1]
                lines.append(f"* [{name}]({name}/index.md)")
            lines.append("")
        by_type: dict[str, list[tuple[str, str, str]]] = {}
        for filename, title, description, ctype in content["concepts"]:
            by_type.setdefault(ctype, []).append((filename, title, description))
        for ctype in sorted(by_type):
            lines.append(f"# {ctype}")
            lines.append("")
            for filename, title, description in sorted(by_type[ctype]):
                suffix = f' - "{description}"' if description else ""
                lines.append(f"* [{title}]({filename}){suffix}")
            lines.append("")

        target = root_path / "index.md" if dirpath == "." else root_path / dirpath / "index.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_viz(db: "GrafitoDatabase", root_path: Path) -> bool:
    """Emit a self-contained viz.html using the d3 backend. Best-effort."""
    try:
        from .viz import export_graph

        graph = db.to_networkx()
        export_graph(
            graph,
            str(root_path / "viz.html"),
            backend="d3",
            node_label="label_and_name",
        )
        return True
    except Exception:
        return False
