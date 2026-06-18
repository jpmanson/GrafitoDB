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
- links under a ``# Citations`` heading -> ``CITES`` relationships, to concepts
  (intra-bundle) or to auto-created ``Reference`` nodes (external URLs)

See ``todo/okf/SPEC.md`` for the format specification.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from ..database import GrafitoDatabase

# Reserved filenames that are not concept documents (SPEC sec. 3.1).
RESERVED_FILENAMES = {"index.md", "log.md"}

# Default label for concepts lacking a `type` (permissive consumption, sec. 9)
# and for stub nodes created from links to not-yet-written concepts (sec. 5.3).
DEFAULT_LABEL = "Concept"

# Label for auto-created nodes representing external citation sources (sec. 8).
REFERENCE_LABEL = "Reference"

# Markdown inline link: [anchor](target)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Bare URL (citations in real bundles often list plain URLs, not markdown links).
_BARE_URL_RE = re.compile(r"https?://[^\s<>\]\)]+")

# Schemes treated as external citations/resources rather than intra-bundle links.
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "ftp://", "//")

# `# Citations` section heading (any level), SPEC sec. 8.
_CITATIONS_HEADING_RE = re.compile(r"^(#{1,6})\s+Citations\s*$", re.IGNORECASE | re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)


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


def classify_target(target: str, source_id: str) -> tuple[str, str] | None:
    """Classify a markdown link target.

    Returns ``("concept", concept_id)`` for an intra-bundle link,
    ``("external", url)`` for an external URL, or ``None`` for a pure in-page
    anchor / empty target. `source_id` resolves relative paths.
    """
    target = target.strip()
    # Drop a title/whitespace and surrounding angle brackets.
    target = target.split()[0] if target else target
    target = target.strip("<>")
    if not target or target.startswith("#"):
        return None
    if target.startswith(_EXTERNAL_PREFIXES):
        return "external", target
    # Intra-bundle path; strip any in-page fragment.
    path = target.split("#", 1)[0]
    if not path:
        return None
    if path.startswith("/"):
        # Bundle-relative (absolute) link (SPEC sec. 5.1).
        concept_id = path.lstrip("/")
    else:
        # Relative link, resolved against the source concept's directory.
        base = PurePosixPath(source_id).parent
        concept_id = _posix_join(base, path)
    if concept_id.endswith(".md"):
        concept_id = concept_id[: -len(".md")]
    return ("concept", concept_id) if concept_id else None


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
        classified = classify_target(raw_target, source_id)
        if classified is not None and classified[0] == "concept":
            links.append((anchor, classified[1]))
    return links


def split_citations(body: str) -> tuple[str, str]:
    """Split a body into (main_body, citations_block).

    The citations block spans from a ``# Citations`` heading to the next heading
    of the same or higher level (or end of file). Returns ``(body, "")`` when no
    citations section is present.
    """
    match = _CITATIONS_HEADING_RE.search(body)
    if not match:
        return body, ""
    start = match.start()
    level = len(match.group(1))
    end = len(body)
    for heading in _HEADING_RE.finditer(body, match.end()):
        if len(heading.group(1)) <= level:
            end = heading.start()
            break
    citations_block = body[start:end]
    main_body = body[:start] + body[end:]
    return main_body, citations_block


def extract_citations(citations_block: str, source_id: str) -> list[tuple[str, str, str]]:
    """Return [(anchor, kind, value), ...] for links inside a citations block.

    ``kind`` is ``"concept"`` (value is a concept ID) or ``"external"`` (value
    is a URL). Handles both markdown links and bare URLs.
    """
    cites: list[tuple[str, str, str]] = []
    link_spans: list[tuple[int, int]] = []
    for match in _LINK_RE.finditer(citations_block):
        anchor, raw_target = match.group(1), match.group(2)
        link_spans.append(match.span())
        classified = classify_target(raw_target, source_id)
        if classified is not None:
            cites.append((anchor, classified[0], classified[1]))
    # Bare URLs that are not already part of a markdown link.
    for match in _BARE_URL_RE.finditer(citations_block):
        start = match.start()
        if any(span_start <= start < span_end for span_start, span_end in link_spans):
            continue
        url = match.group(0).rstrip(".,;")
        cites.append(("", "external", url))
    return cites


def _concept_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return rel[: -len(".md")] if rel.endswith(".md") else rel


# Default concept fields combined into the document embedded for semantic search.
DEFAULT_EMBED_FIELDS = ("title", "description", "body")


def concept_document(properties: dict, fields: tuple[str, ...] = DEFAULT_EMBED_FIELDS) -> str:
    """Build the text embedded for a concept by joining selected fields."""
    parts = []
    for field in fields:
        value = properties.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def import_bundle(
    db: "GrafitoDatabase",
    root: str | Path,
    *,
    link_type: str = "LINKS_TO",
    citations: bool = True,
    citation_type: str = "CITES",
    configure_fts: bool = True,
    embed: "Any" = None,
    embed_index: str = "okf",
    embed_fields: tuple[str, ...] = DEFAULT_EMBED_FIELDS,
    embed_backend: str = "bruteforce",
    uri_prefix: str = "okf:",
) -> dict:
    """Import an OKF bundle directory into ``db``.

    Args:
        db: Target database.
        root: Path to the bundle root directory.
        link_type: Relationship type created for intra-bundle markdown links.
        citations: Parse the ``# Citations`` section into ``citation_type``
            relationships (to concepts or auto-created ``Reference`` nodes).
        citation_type: Relationship type created for citations.
        configure_fts: Configure full-text search over title/description/body
            (best-effort; skipped if SQLite lacks FTS5).
        embed: Optional ``EmbeddingFunction``. When provided, each concept is
            embedded for semantic search into a vector index. Query later with
            ``db.semantic_search("text", index=embed_index)``.
        embed_index: Name of the vector index created for concept embeddings.
        embed_fields: Concept fields concatenated into the embedded document.
        embed_backend: Vector index backend (default ``bruteforce``, no extra
            dependencies).
        uri_prefix: Prefix prepended to each concept ID to form the node ``uri``.

    Returns:
        Summary dict with
        ``nodes``, ``relationships``, ``citations``, ``references``,
        ``stubs``, ``skipped`` and ``embedded`` counts.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"OKF bundle root not found: {root_path}")

    concept_to_node: dict[str, int] = {}
    pending_links: list[tuple[str, str, str]] = []  # (source_id, anchor, target_id)
    # (source_id, anchor, kind, value)
    pending_citations: list[tuple[str, str, str, str]] = []
    embed_docs: list[tuple[int, str]] = []  # (node_id, document) for real concepts
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

        if embed is not None:
            embed_docs.append((node.id, concept_document(properties, embed_fields)))

        # Citation links are excluded from LINKS_TO so they only yield CITES.
        main_body, citations_block = split_citations(body) if citations else (body, "")
        for anchor, target_id in extract_links(main_body, concept_id):
            pending_links.append((concept_id, anchor, target_id))
        for anchor, kind, value in extract_citations(citations_block, concept_id):
            pending_citations.append((concept_id, anchor, kind, value))

    # Second pass: resolve links, creating stubs for missing targets (sec. 5.3).
    relationships = 0
    stubs = 0

    def resolve_concept(target_id: str) -> int:
        nonlocal stubs
        if target_id not in concept_to_node:
            stub = db.create_node(
                labels=[DEFAULT_LABEL],
                properties={"concept_id": target_id, "stub": True},
                uri=f"{uri_prefix}{target_id}",
            )
            concept_to_node[target_id] = stub.id
            stubs += 1
        return concept_to_node[target_id]

    for source_id, anchor, target_id in pending_links:
        db.create_relationship(
            concept_to_node[source_id],
            resolve_concept(target_id),
            link_type,
            properties={"anchor": anchor} if anchor else {},
        )
        relationships += 1

    # Citations: link to concepts (intra-bundle) or to Reference nodes (external).
    reference_nodes: dict[str, int] = {}  # url -> node_id
    citation_count = 0
    for source_id, anchor, kind, value in pending_citations:
        if kind == "concept":
            target = resolve_concept(value)
        else:
            if value not in reference_nodes:
                ref = db.create_node(
                    labels=[REFERENCE_LABEL],
                    properties={"title": anchor or value, "url": value, "okf_auto": True},
                    uri=value,
                )
                reference_nodes[value] = ref.id
            target = reference_nodes[value]
        db.create_relationship(
            concept_to_node[source_id],
            target,
            citation_type,
            properties={"anchor": anchor} if anchor else {},
        )
        citation_count += 1

    if configure_fts and db.has_fts5():
        # OKF `type` values are free-form (may contain spaces), so index across
        # all node labels rather than per-label.
        db.create_text_index("node", None, ["title", "description", "body"])
        db.rebuild_text_index()

    embedded = 0
    if embed is not None and embed_docs:
        node_ids = [node_id for node_id, _ in embed_docs]
        documents = [doc for _, doc in embed_docs]
        dim = getattr(embed, "dimension", None) or len(embed([documents[0] or " "])[0])
        db.create_vector_index(
            embed_index,
            dim=dim,
            backend=embed_backend,
            embedding_function=embed,
            if_not_exists=True,
        )
        db.upsert_embeddings(node_ids, documents, index=embed_index)
        embedded = len(embed_docs)

    return {
        "nodes": nodes,
        "relationships": relationships,
        "citations": citation_count,
        "references": len(reference_nodes),
        "stubs": stubs,
        "skipped": skipped,
        "embedded": embedded,
    }
