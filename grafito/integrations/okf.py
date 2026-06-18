"""Open Knowledge Format (OKF) bundle exporter.

Serializes a GrafitoDB graph back into an OKF bundle: a directory tree of
markdown files with YAML frontmatter (see ``todo/okf/SPEC.md``). This is the
inverse of :func:`grafito.importers.okf.import_bundle`.

Each node becomes one concept document:

- the first node label -> frontmatter ``type``
- node properties (minus internal keys) -> frontmatter
- the ``body`` property -> markdown body (or a synthesized ``# Links`` section
  when no body is stored)
- the concept's file path is derived from its ``uri`` (``<uri_prefix><id>``)

A bundle-root ``index.md`` is generated for progressive disclosure (SPEC sec. 6).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import yaml

from ..importers.okf import DEFAULT_LABEL

if TYPE_CHECKING:
    from ..database import GrafitoDatabase

# Properties that are GrafitoDB bookkeeping, not OKF frontmatter.
_INTERNAL_PROPS = ("body", "concept_id", "stub")

# Recommended frontmatter order (SPEC sec. 4.1). `type` is always emitted first.
_RECOMMENDED_ORDER = ("title", "description", "resource", "tags", "timestamp")


def _concept_id(node, uri_prefix: str) -> str:
    if node.uri and node.uri.startswith(uri_prefix):
        return node.uri[len(uri_prefix):]
    cid = node.properties.get("concept_id")
    if isinstance(cid, str) and cid:
        return cid
    return f"node-{node.id}"


def _ordered_frontmatter(node) -> dict:
    """Build an ordered frontmatter dict: type, recommended keys, then the rest."""
    label = node.labels[0] if node.labels else DEFAULT_LABEL
    props = {k: v for k, v in node.properties.items() if k not in _INTERNAL_PROPS}

    frontmatter: dict = {"type": label}
    for key in _RECOMMENDED_ORDER:
        if key in props:
            frontmatter[key] = props.pop(key)
    for key in sorted(props):
        frontmatter[key] = props[key]
    return frontmatter


def _synthesize_links_section(
    db: "GrafitoDatabase", node, uri_prefix: str, heading: str
) -> str:
    """Render outgoing relationships as an OKF ``# Links`` markdown section."""
    rels = db.match_relationships(source_id=node.id)
    lines: list[str] = []
    for rel in rels:
        target = db.get_node(rel.target_id)
        if target is None:
            continue
        target_id = _concept_id(target, uri_prefix)
        anchor = rel.properties.get("anchor") or target.properties.get("title") or target_id
        lines.append(f"- [{anchor}](/{target_id}.md)")
    if not lines:
        return ""
    return f"# {heading}\n\n" + "\n".join(lines) + "\n"


def export_bundle(
    db: "GrafitoDatabase",
    root: str | Path,
    *,
    uri_prefix: str = "okf:",
    write_index: bool = True,
    write_viz: bool = False,
    links_heading: str = "Links",
) -> dict:
    """Export the graph in ``db`` to an OKF bundle directory at ``root``.

    Args:
        db: Source database.
        root: Destination bundle directory (created if missing).
        uri_prefix: Prefix used to recover concept IDs from node URIs. Should
            match the value used at import time.
        write_index: Generate per-directory ``index.md`` files for progressive
            disclosure (SPEC sec. 6), following the convention used by reference
            bundles: a root index linking to child directory indexes, and each
            directory index grouping its concepts by ``type``.
        write_viz: Also emit a self-contained ``viz.html`` graph viewer at the
            bundle root (best-effort; mirrors the reference bundles).
        links_heading: Heading used for the synthesized links section when a
            node has no stored ``body``.

    Returns:
        Summary dict: ``{"concepts", "skipped", "viz"}``.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    concepts = 0
    skipped = 0
    # (concept_id, title, description, type) for index generation.
    index_entries: list[tuple[str, str, str, str]] = []

    for node in db.match_nodes():
        # Skip derived nodes: stubs (broken links) and auto-created Reference
        # nodes for external citations. Both are re-derived from concept bodies
        # on re-import, so they are not written as their own concept files.
        if node.properties.get("stub") is True or node.properties.get("okf_auto") is True:
            skipped += 1
            continue

        concept_id = _concept_id(node, uri_prefix)
        frontmatter = _ordered_frontmatter(node)

        body = node.properties.get("body")
        if not isinstance(body, str) or not body.strip():
            body = _synthesize_links_section(db, node, uri_prefix, links_heading)

        fm_text = yaml.safe_dump(
            frontmatter,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=None,  # scalar-only lists render inline: [a, b]
        ).strip()
        document = f"---\n{fm_text}\n---\n\n{body}"
        if not document.endswith("\n"):
            document += "\n"

        file_path = root_path / PurePosixPath(f"{concept_id}.md")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(document, encoding="utf-8")
        concepts += 1

        index_entries.append(
            (
                concept_id,
                str(frontmatter.get("title") or concept_id),
                str(frontmatter.get("description") or ""),
                str(frontmatter["type"]),
            )
        )

    if write_index:
        _write_indexes(root_path, index_entries)

    viz_written = False
    if write_viz:
        viz_written = _write_viz(db, root_path)

    return {"concepts": concepts, "skipped": skipped, "viz": viz_written}


def _write_indexes(root_path: Path, entries: list[tuple[str, str, str, str]]) -> None:
    """Write one ``index.md`` per directory (recursive, SPEC sec. 6 convention).

    Each directory index lists its immediate child directories under a
    ``# Subdirectories`` heading and its own concepts grouped by ``type``.
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
