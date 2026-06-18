"""Open Knowledge Format (OKF) bundle importer.

OKF is a directory tree of UTF-8 markdown files with YAML frontmatter. Each
non-reserved ``.md`` file is a *concept*; its path within the bundle (minus the
``.md`` suffix) is its *concept ID*. Markdown links between concepts express
untyped, directed relationships.

This importer maps an OKF bundle onto the Property Graph Model:

- concept ``type`` -> node label (falls back to ``Concept`` when absent)
- remaining frontmatter keys -> node properties
- markdown body -> ``body`` property (feeds full-text search)
- concept ID -> node ``uri`` (``<uri_prefix><concept-id>``)
- markdown links -> relationships (default type ``LINKS_TO``)

See ``todo/okf/SPEC.md`` for the format specification.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ..database import GrafitoDatabase

# Reserved filenames that are not concept documents (SPEC sec. 3.1).
RESERVED_FILENAMES = {"index.md", "log.md"}

# Default label for concepts lacking a `type` (permissive consumption, sec. 9)
# and for stub nodes created from links to not-yet-written concepts (sec. 5.3).
DEFAULT_LABEL = "Concept"

# Markdown inline link: [anchor](target)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Schemes treated as external citations/resources rather than intra-bundle links.
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "ftp://", "//", "#")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown document into (frontmatter dict, body).

    Returns an empty dict when no frontmatter block is present.
    """
    if not text.startswith("---"):
        return {}, text
    # Frontmatter is delimited by `---` lines at the very top.
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            block = "".join(lines[1:idx])
            body = "".join(lines[idx + 1 :])
            # Strip a single leading newline left after the closing delimiter.
            if body.startswith("\n"):
                body = body[1:]
            data = yaml.safe_load(block) or {}
            if not isinstance(data, dict):
                data = {}
            return data, body
    # No closing delimiter: treat the whole file as body.
    return {}, text


def _normalize_target(target: str, source_id: str) -> str | None:
    """Resolve a markdown link target to a concept ID, or None if external.

    `source_id` is the concept ID of the document containing the link, used to
    resolve relative paths.
    """
    target = target.strip()
    # Drop a fragment/anchor and surrounding angle brackets or quotes.
    target = target.split()[0] if target else target
    target = target.strip("<>")
    if not target or target.startswith(_EXTERNAL_PREFIXES):
        return None
    # Strip any in-page fragment.
    target = target.split("#", 1)[0]
    if not target:
        return None
    if target.startswith("/"):
        # Bundle-relative (absolute) link (SPEC sec. 5.1).
        concept_id = target.lstrip("/")
    else:
        # Relative link, resolved against the source concept's directory.
        base = PurePosixPath(source_id).parent
        concept_id = _posix_join(base, target)
    if concept_id.endswith(".md"):
        concept_id = concept_id[: -len(".md")]
    return concept_id or None


def _posix_join(base: PurePosixPath, rel: str) -> str:
    parts = list(base.parts)
    for segment in rel.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if parts:
                parts.pop()
            continue
        parts.append(segment)
    return "/".join(parts)


def extract_links(body: str, source_id: str) -> list[tuple[str, str]]:
    """Return [(anchor, target_concept_id), ...] for intra-bundle links."""
    links: list[tuple[str, str]] = []
    for match in _LINK_RE.finditer(body):
        anchor, raw_target = match.group(1), match.group(2)
        concept_id = _normalize_target(raw_target, source_id)
        if concept_id is not None:
            links.append((anchor, concept_id))
    return links


def _concept_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return rel[: -len(".md")] if rel.endswith(".md") else rel


def import_bundle(
    db: "GrafitoDatabase",
    root: str | Path,
    *,
    link_type: str = "LINKS_TO",
    configure_fts: bool = True,
    uri_prefix: str = "okf:",
) -> dict:
    """Import an OKF bundle directory into ``db``.

    Args:
        db: Target database.
        root: Path to the bundle root directory.
        link_type: Relationship type created for intra-bundle markdown links.
        configure_fts: Configure full-text search over title/description/body
            (best-effort; skipped if SQLite lacks FTS5).
        uri_prefix: Prefix prepended to each concept ID to form the node ``uri``.

    Returns:
        Summary dict: ``{"nodes", "relationships", "stubs", "skipped"}``.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"OKF bundle root not found: {root_path}")

    concept_to_node: dict[str, int] = {}
    pending_links: list[tuple[str, str, str]] = []  # (source_id, anchor, target_id)
    nodes = 0
    skipped = 0

    for path in sorted(root_path.rglob("*.md")):
        if path.name in RESERVED_FILENAMES:
            skipped += 1
            continue
        text = path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(text)
        concept_id = _concept_id_for(path, root_path)

        concept_type = frontmatter.get("type")
        label = concept_type if isinstance(concept_type, str) and concept_type else DEFAULT_LABEL

        properties = {k: v for k, v in frontmatter.items() if k != "type"}
        properties["body"] = body
        properties.setdefault("concept_id", concept_id)

        node = db.create_node(
            labels=[label],
            properties=properties,
            uri=f"{uri_prefix}{concept_id}",
        )
        concept_to_node[concept_id] = node.id
        nodes += 1

        for anchor, target_id in extract_links(body, concept_id):
            pending_links.append((concept_id, anchor, target_id))

    # Second pass: resolve links, creating stubs for missing targets (sec. 5.3).
    relationships = 0
    stubs = 0
    for source_id, anchor, target_id in pending_links:
        if target_id not in concept_to_node:
            stub = db.create_node(
                labels=[DEFAULT_LABEL],
                properties={"concept_id": target_id, "stub": True},
                uri=f"{uri_prefix}{target_id}",
            )
            concept_to_node[target_id] = stub.id
            stubs += 1
        db.create_relationship(
            concept_to_node[source_id],
            concept_to_node[target_id],
            link_type,
            properties={"anchor": anchor} if anchor else {},
        )
        relationships += 1

    if configure_fts and db.has_fts5():
        # OKF `type` values are free-form (may contain spaces), so index across
        # all node labels rather than per-label.
        db.create_text_index("node", None, ["title", "description", "body"])
        db.rebuild_text_index()

    return {
        "nodes": nodes,
        "relationships": relationships,
        "stubs": stubs,
        "skipped": skipped,
    }
