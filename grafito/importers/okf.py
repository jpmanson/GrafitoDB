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

See https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf for
the format specification.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

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

# Relationship types added by OKFBundle's trust model (supersede/conflicts_with),
# not reproduced by re-parsing a concept file. Preserved across incremental updates.
_TRUST_REL_TYPES = frozenset({"SUPERSEDES", "CONFLICTS_WITH"})

# Markdown inline link: [anchor](target)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Bare URL (citations in real bundles often list plain URLs, not markdown links).
_BARE_URL_RE = re.compile(r"https?://[^\s<>\]\)]+")

# Schemes treated as external citations/resources rather than intra-bundle links.
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "ftp://", "//")

# `# Citations` section heading (any level), SPEC sec. 8.
_CITATIONS_HEADING_RE = re.compile(r"^(#{1,6})\s+Citations\s*$", re.IGNORECASE | re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)

# Heading with its text, for typed-link extraction.
_HEADING_TEXT_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$", re.MULTILINE)

_REL_TYPE_RE = re.compile(r"[A-Z_][A-Z0-9_]*")

# Obsidian wikilink: [[Target]], [[Target|Alias]], [[Target#Heading]],
# [[Target#Heading|Alias]] (also matches the `![[Target]]` embed form).
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown document into (frontmatter dict, body).

    Returns an empty dict when no frontmatter block is present. Raises
    ``yaml.YAMLError`` when a block is present but is not valid YAML — callers
    that want permissive consumption (SPEC sec. 9) catch it and fall back to
    treating the whole file as body (see ``import_bundle``).
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
    return [(anchor, target) for anchor, target, _ in extract_typed_links(body, source_id)]


def _headings_before(body: str) -> list[tuple[int, str]]:
    """[(position, heading_text), ...] for every markdown heading in ``body``."""
    return [(m.start(), m.group(1)) for m in _HEADING_TEXT_RE.finditer(body)]


def _heading_at(headings: list[tuple[int, str]], pos: int) -> str | None:
    """The text of the closest heading at or before ``pos``, or ``None``."""
    heading = None
    for hpos, text in headings:
        if hpos > pos:
            break
        heading = text
    return heading


def extract_typed_links(body: str, source_id: str) -> list[tuple[str, str, str | None]]:
    """Return [(anchor, target_concept_id, heading), ...] for intra-bundle links.

    ``heading`` is the text of the closest markdown heading above the link, or
    ``None`` for links before any heading.
    """
    headings = _headings_before(body)
    links: list[tuple[str, str, str | None]] = []
    for match in _LINK_RE.finditer(body):
        anchor, raw_target = match.group(1), match.group(2)
        classified = classify_target(raw_target, source_id)
        if classified is None or classified[0] != "concept":
            continue
        links.append((anchor, classified[1], _heading_at(headings, match.start())))
    return links


def parse_wikilink(raw: str) -> tuple[str, str | None]:
    """Split a wikilink's inner text into ``(target, alias)``.

    Handles ``Target``, ``Target|Alias``, ``Target#Heading``, and
    ``Target#Heading|Alias`` — the heading fragment (an in-page anchor) is
    dropped, same as :func:`classify_target` does for markdown links.
    """
    target_part, _, alias = raw.partition("|")
    target_part = target_part.split("#", 1)[0].strip()
    return target_part, (alias.strip() or None)


def resolve_wikilink(
    target: str, concept_ids: "set[str]", basename_index: dict[str, list[str]]
) -> str | None:
    """Resolve an Obsidian wikilink target to a concept ID.

    Obsidian links by note title/filename, not by path, so this tries an exact
    concept-ID match first (``[[decisions/0001-use-sqlite]]``), then a
    case-insensitive match against every concept's basename. A basename shared
    by more than one concept is ambiguous and left unresolved (``None``) rather
    than guessed — same as a target matching nothing at all, both cases the
    caller treats as "not found" (see :func:`extract_typed_wikilinks`).
    """
    if target in concept_ids:
        return target
    candidates = basename_index.get(target.lower(), [])
    return candidates[0] if len(candidates) == 1 else None


def extract_typed_wikilinks(
    body: str, concept_ids: "set[str]", basename_index: dict[str, list[str]]
) -> list[tuple[str, str, str | None]]:
    """Return [(anchor, target_concept_id, heading), ...] for Obsidian wikilinks.

    Unlike markdown links (path-resolved, always kept — an unknown target
    becomes a stub per SPEC sec. 5.3), a wikilink is only kept when it
    resolves unambiguously via :func:`resolve_wikilink`: to an existing
    concept, or verbatim as a not-yet-written concept ID (matching Obsidian's
    own "red link" convention) when nothing matches its basename at all. An
    *ambiguous* basename (shared by more than one concept) is dropped instead
    of guessed. Citations sections are not scanned — wikilink support only
    covers ``LINKS_TO``/typed body links.
    """
    headings = _headings_before(body)
    links: list[tuple[str, str, str | None]] = []
    for match in _WIKILINK_RE.finditer(body):
        target, alias = parse_wikilink(match.group(1))
        if not target:
            continue
        resolved = resolve_wikilink(target, concept_ids, basename_index)
        if resolved is None and target.lower() in basename_index:
            continue  # ambiguous basename: skip rather than guess
        target_id = resolved or target  # unresolved -> verbatim, becomes a stub
        links.append((alias or target, target_id, _heading_at(headings, match.start())))
    return links


def rel_type_from_heading(heading: str | None) -> str | None:
    """Normalize a heading into a relationship type (``Joins with`` -> ``JOINS_WITH``).

    Returns ``None`` when the heading is absent or does not normalize to a valid
    type identifier — callers fall back to the generic link type. ``Links`` (the
    conventional heading synthesized by the exporter) also maps to ``None`` so
    the default section round-trips to the default ``link_type``.
    """
    if heading is None:
        return None
    rel_type = re.sub(r"[^A-Za-z0-9]+", "_", heading).strip("_").upper()
    if rel_type == "LINKS" or not _REL_TYPE_RE.fullmatch(rel_type):
        return None
    return rel_type


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


# Log entry: a list bullet under a `## YYYY-MM-DD` date heading (SPEC sec. 7).
_LOG_DATE_RE = re.compile(r"^#{1,6}\s+(\d{4}-\d{2}-\d{2})\s*$")
_LOG_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
_LOG_KIND_RE = re.compile(r"^\*\*([^*]+)\*\*:?\s*(.*)$")


def parse_log_entries(text: str) -> list[tuple[str, str | None, str]]:
    """Parse a ``log.md`` into ``[(date, kind, text), ...]`` (SPEC sec. 7).

    ``kind`` is the leading bold word (``Update``/``Creation``/...) when present.
    """
    entries: list[tuple[str, str | None, str]] = []
    current_date: str | None = None
    for line in text.splitlines():
        date_match = _LOG_DATE_RE.match(line)
        if date_match:
            current_date = date_match.group(1)
            continue
        bullet = _LOG_BULLET_RE.match(line)
        if bullet and current_date:
            entry = bullet.group(1).strip()
            kind_match = _LOG_KIND_RE.match(entry)
            kind = kind_match.group(1).strip() if kind_match else None
            entries.append((current_date, kind, entry))
    return entries


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
    typed_links: bool = False,
    citations: bool = True,
    citation_type: str = "CITES",
    configure_fts: bool = True,
    embed: "Any" = None,
    embed_index: str = "okf",
    embed_fields: tuple[str, ...] = DEFAULT_EMBED_FIELDS,
    embed_backend: str = "bruteforce",
    embed_options: dict | None = None,
    progress_every: int | None = None,
    progress: Callable[[str, int], None] | None = None,
    directory_nodes: bool = False,
    directory_label: str = "Directory",
    contains_type: str = "CONTAINS",
    import_log: bool = False,
    log_label: str = "LogEntry",
    mentions_type: str = "MENTIONS",
    uri_prefix: str = "okf:",
    incremental: bool = False,
    prune: bool = False,
    wikilinks: bool = False,
) -> dict:
    """Import an OKF bundle directory into ``db``.

    Args:
        db: Target database.
        root: Path to the bundle root directory.
        link_type: Relationship type created for intra-bundle markdown links.
        typed_links: Derive the relationship type from the markdown heading a
            link sits under (a link under ``# Joins with`` becomes a
            ``JOINS_WITH`` relationship). Links before any heading, under
            ``# Links``, or under headings that do not normalize to a valid
            type keep ``link_type``. Off by default.
        wikilinks: Also resolve Obsidian-style ``[[Note]]``/``[[Note|Alias]]``
            wikilinks in the main body (not the Citations section) into the
            same relationship types as markdown links — Obsidian vaults are
            OKF-compatible without a plugin, but wikilinks name a note by
            title rather than path, so they need vault-wide resolution: an
            exact concept-ID match first, then a case-insensitive match
            against every concept's filename (basename). A basename shared by
            more than one concept is ambiguous and skipped; one matching
            nothing becomes a stub keyed by the literal link text (Obsidian's
            own "red link" — not-yet-written note — convention). Off by
            default.
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
        embed_options: Extra options forwarded to ``create_vector_index`` —
            e.g. ``{"store_embeddings": True}`` persists the vectors in the
            database so a file-backed db reuses them across sessions without
            re-embedding, and ``{"index_path": ...}`` sets the on-disk location
            for file-backed backends (faiss/hnswlib/...).
        progress_every: Print a progress line every N concept files (and at
            the end of each phase). Mirrors ``import_neo4j_dump``.
        progress: Optional callback ``(phase, count)`` invoked at the same
            cadence instead of printing — for programmatic progress reporting.
            Phases: ``concepts``, ``links``, ``citations``, ``embedded``,
            ``done``.
        directory_nodes: Synthesize ``directory_label`` nodes + ``contains_type``
            edges from concept paths, enabling top-down graph traversal
            (root -> subdir -> concept). Off by default.
        directory_label: Label for synthesized directory nodes.
        contains_type: Relationship type for directory containment.
        import_log: Import ``log.md`` entries as ``log_label`` nodes, linked to
            mentioned concepts via ``mentions_type``. Off by default.
        log_label: Label for log-entry nodes.
        mentions_type: Relationship type from a log entry to a mentioned concept.
        uri_prefix: Prefix prepended to each concept ID to form the node ``uri``.
        incremental: Skip re-parsing and re-embedding concept files whose
            content hasn't changed since the last import (tracked via a
            content hash stored on each node, ``okf_hash``). A changed file
            updates its node in place (same node ID, so incoming links from
            unchanged concepts stay valid) rather than creating a duplicate;
            a link to a not-yet-imported concept reuses an existing stub
            instead of creating a second one. Trust-model edges added via
            ``OKFBundle.supersede``/``conflicts_with`` (``SUPERSEDES``,
            ``CONFLICTS_WITH``) are preserved on updated concepts. Off by
            default (matches the original always-reimport behavior). Note:
            ``directory_nodes``/``import_log`` are not incremental-aware —
            combining them with ``incremental=True`` duplicates directory/log
            nodes on repeated imports.
        prune: Requires ``incremental=True``. Delete nodes for concept files
            that existed in a prior import but are no longer present in the
            bundle. Mirrors the exporter's ``prune`` option.

    Returns:
        Summary dict with ``nodes`` (created or updated this run),
        ``relationships``, ``citations``, ``references`` (new ``Reference``
        nodes created this run), ``stubs``, ``skipped``, ``embedded``,
        ``directories``, ``log_entries``, ``malformed`` (concept IDs whose
        frontmatter was not valid YAML and was treated as body text),
        ``unchanged`` (hash-matched files skipped), ``updated`` (pre-existing
        nodes whose content changed), and ``pruned`` (nodes removed because
        their file disappeared from the bundle).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"OKF bundle root not found: {root_path}")
    if prune and not incremental:
        raise ValueError("prune=True requires incremental=True")

    concept_to_node: dict[str, int] = {}
    real_concepts: dict[str, int] = {}  # concept_id -> node_id (file concepts only)
    # (source_id, anchor, target_id, rel_type)
    pending_links: list[tuple[str, str, str, str]] = []
    # (source_id, anchor, kind, value)
    pending_citations: list[tuple[str, str, str, str]] = []
    embed_docs: list[tuple[int, str]] = []  # (node_id, document) for real concepts
    wikilink_bodies: list[tuple[str, str]] = []  # (concept_id, main_body), only if wikilinks
    nodes = 0
    skipped = 0
    malformed: list[str] = []
    processed = 0
    unchanged = 0
    updated = 0
    pruned = 0

    def report(phase: str, count: int, *, end: bool = False) -> None:
        if progress is not None:
            progress(phase, count)
        elif progress_every:
            if end:
                print(f"\rImported {count} {phase}." + " " * 20)
            else:
                print(f"\rImporting {phase}: {count}", end="", flush=True)

    # In-loop reporting cadence: every `progress_every` files, or every file
    # when only a callback is given.
    stride = progress_every or (1 if progress is not None else 0)

    # Concept lookups (`concept_id`) are served by an expression index. Created
    # before the transaction because index creation commits unconditionally.
    db.create_node_index(None, "concept_id")

    # Incremental mode: seed lookups from the graph's current state so unchanged
    # concepts, existing stubs, and existing Reference nodes are reused by ID
    # instead of duplicated.
    existing_real: dict[str, Any] = {}  # concept_id -> Node, real (non-stub) concepts only
    reference_nodes: dict[str, int] = {}  # url -> node_id
    if incremental:
        for existing_node in db.match_nodes():
            props = existing_node.properties
            cid = props.get("concept_id")
            if not isinstance(cid, str) or not cid:
                continue
            if props.get("stub") is True:
                concept_to_node[cid] = existing_node.id
            elif (
                props.get("okf_auto")
                or props.get("directory")
                or props.get("log")
                or props.get("pending_review")
            ):
                # Reference/Directory/LogEntry nodes, and concepts staged via
                # OKFBundle.propose() awaiting review, are not file-backed
                # concepts — never reused/promoted by the file importer.
                continue
            else:
                existing_real[cid] = existing_node
                concept_to_node[cid] = existing_node.id
        for ref_node in db.match_nodes(labels=[REFERENCE_LABEL]):
            url = ref_node.properties.get("url")
            if isinstance(url, str) and url:
                reference_nodes[url] = ref_node.id

    # One transaction for the whole import: per-node autocommit dominates load
    # time on large bundles. Respect a transaction the caller already opened.
    txn = nullcontext(db) if getattr(db, "_in_transaction", False) else db
    with txn:
        for path in sorted(root_path.rglob("*.md")):
            if path.name in RESERVED_FILENAMES:
                skipped += 1
                continue
            text = path.read_text(encoding="utf-8")
            concept_id = _concept_id_for(path, root_path)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            processed += 1

            prior_real = None
            node_id = None
            if incremental:
                prior_real = existing_real.pop(concept_id, None)
                if prior_real is not None and prior_real.properties.get("okf_hash") == content_hash:
                    # Unchanged since the last import: keep the node as-is,
                    # skip re-parsing its links/citations and re-embedding it.
                    real_concepts[concept_id] = prior_real.id
                    unchanged += 1
                    if stride and processed % stride == 0:
                        report("concepts", processed)
                    continue
                node_id = concept_to_node.get(concept_id)  # prior real (changed) or a stub

            try:
                frontmatter, body = parse_frontmatter(text)
            except yaml.YAMLError:
                # Permissive consumption (SPEC sec. 9): one bad file must not
                # abort the bundle. Keep the full text as body and report it.
                frontmatter, body = {}, text
                malformed.append(concept_id)

            concept_type = frontmatter.get("type")
            label = (
                concept_type if isinstance(concept_type, str) and concept_type else DEFAULT_LABEL
            )

            properties = {k: v for k, v in frontmatter.items() if k != "type"}
            properties["body"] = body
            properties.setdefault("concept_id", concept_id)
            properties["okf_hash"] = content_hash

            if node_id is not None:
                db.replace_node_properties(node_id, properties)
                current_node = db.get_node(node_id)
                if current_node.labels != [label]:
                    if current_node.labels:
                        db.remove_labels(node_id, list(current_node.labels))
                    db.add_labels(node_id, [label])
                if prior_real is not None:
                    # Re-derive this concept's own links/citations below; leave
                    # trust-model edges (added via OKFBundle) untouched.
                    for rel in db.match_relationships(source_id=node_id):
                        if rel.type not in _TRUST_REL_TYPES:
                            db.delete_relationship(rel.id)
                    updated += 1
            else:
                node = db.create_node(
                    labels=[label],
                    properties=properties,
                    uri=f"{uri_prefix}{concept_id}",
                )
                node_id = node.id

            concept_to_node[concept_id] = node_id
            real_concepts[concept_id] = node_id
            nodes += 1
            if stride and processed % stride == 0:
                report("concepts", processed)

            if embed is not None:
                embed_docs.append((node_id, concept_document(properties, embed_fields)))

            # Citation links are excluded from LINKS_TO so they only yield CITES.
            main_body, citations_block = split_citations(body) if citations else (body, "")
            for anchor, target_id, heading in extract_typed_links(main_body, concept_id):
                rel_type = (rel_type_from_heading(heading) if typed_links else None) or link_type
                pending_links.append((concept_id, anchor, target_id, rel_type))
            for anchor, kind, value in extract_citations(citations_block, concept_id):
                pending_citations.append((concept_id, anchor, kind, value))
            if wikilinks:
                wikilink_bodies.append((concept_id, main_body))

        if stride:
            report("concepts", processed, end=True)

        if incremental and prune:
            for stale_id, stale_node in existing_real.items():
                db.delete_node(stale_node.id)
                concept_to_node.pop(stale_id, None)
                pruned += 1

        if wikilinks and wikilink_bodies:
            # Resolved after the whole bundle's real_concepts is known — a
            # wikilink names a note by title, not by the path of the file
            # currently being parsed, so it needs vault-wide context.
            concept_id_set = set(real_concepts)
            basename_index: dict[str, list[str]] = {}
            for cid in real_concepts:
                basename_index.setdefault(cid.rsplit("/", 1)[-1].lower(), []).append(cid)
            for concept_id, main_body in wikilink_bodies:
                for anchor, target_id, heading in extract_typed_wikilinks(
                    main_body, concept_id_set, basename_index
                ):
                    rel_type = (rel_type_from_heading(heading) if typed_links else None) or link_type
                    pending_links.append((concept_id, anchor, target_id, rel_type))

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

        for source_id, anchor, target_id, rel_type in pending_links:
            db.create_relationship(
                concept_to_node[source_id],
                resolve_concept(target_id),
                rel_type,
                properties={"anchor": anchor} if anchor else {},
            )
            relationships += 1
        if stride:
            report("links", relationships, end=True)

        # Citations: link to concepts (intra-bundle) or to Reference nodes (external).
        # `reference_nodes` is pre-seeded with existing Reference nodes in
        # incremental mode, so only genuinely new URLs create a node here.
        citation_count = 0
        new_references = 0
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
                    new_references += 1
                target = reference_nodes[value]
            db.create_relationship(
                concept_to_node[source_id],
                target,
                citation_type,
                properties={"anchor": anchor} if anchor else {},
            )
            citation_count += 1
        if stride:
            report("citations", citation_count, end=True)

        # Optional: synthesize a directory tree (Directory nodes + CONTAINS edges).
        directories = 0
        if directory_nodes:
            dir_paths: set[str] = set()
            for concept_id in real_concepts:
                parts = concept_id.split("/")
                for i in range(1, len(parts)):
                    dir_paths.add("/".join(parts[:i]))
            dir_node: dict[str, int] = {}
            root_node = db.create_node(
                labels=[directory_label],
                properties={"path": "", "name": "", "directory": True},
                uri=uri_prefix,
            )
            dir_node[""] = root_node.id
            directories += 1
            for path in sorted(dir_paths):
                node = db.create_node(
                    labels=[directory_label],
                    properties={"path": path, "name": path.rsplit("/", 1)[-1], "directory": True},
                    uri=f"{uri_prefix}{path}/",
                )
                dir_node[path] = node.id
                directories += 1
            for path in sorted(dir_paths):
                parent = path.rsplit("/", 1)[0] if "/" in path else ""
                db.create_relationship(dir_node[parent], dir_node[path], contains_type)
            for concept_id, node_id in real_concepts.items():
                parent = concept_id.rsplit("/", 1)[0] if "/" in concept_id else ""
                db.create_relationship(dir_node[parent], node_id, contains_type)

        # Optional: import log.md entries as LogEntry nodes + MENTIONS edges.
        log_entries = 0
        if import_log:
            for log_path in sorted(root_path.rglob("log.md")):
                scope = log_path.parent.relative_to(root_path).as_posix()
                scope = "" if scope == "." else scope
                source_id = f"{scope}/_log" if scope else "_log"
                log_text = log_path.read_text(encoding="utf-8")
                for date, kind, entry_text in parse_log_entries(log_text):
                    entry_node = db.create_node(
                        labels=[log_label],
                        properties={
                            "date": date,
                            "kind": kind,
                            "text": entry_text,
                            "scope": scope,
                            "log": True,
                        },
                    )
                    log_entries += 1
                    for _anchor, target_id in extract_links(entry_text, source_id):
                        if target_id in concept_to_node:
                            db.create_relationship(
                                entry_node.id, concept_to_node[target_id], mentions_type
                            )

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
                options=dict(embed_options) if embed_options else None,
                embedding_function=embed,
                if_not_exists=True,
            )
            db.upsert_embeddings(node_ids, documents, index=embed_index)
            embedded = len(embed_docs)
            if stride:
                report("embedded", embedded, end=True)

    if progress is not None:
        progress("done", processed)

    return {
        "nodes": nodes,
        "relationships": relationships,
        "citations": citation_count,
        "references": new_references,
        "stubs": stubs,
        "skipped": skipped,
        "embedded": embedded,
        "directories": directories,
        "log_entries": log_entries,
        "malformed": malformed,
        "unchanged": unchanged,
        "updated": updated,
        "pruned": pruned,
    }


def validate_bundle(root: str | Path) -> dict:
    """Validate a bundle against the OKF v0.1 conformance rules (SPEC sec. 9).

    Reports problems without importing anything and without aborting on the
    first bad file — the linter counterpart to the importer's permissive
    consumption.

    Errors (conformance failures):

    - a non-reserved ``.md`` file with no frontmatter block;
    - a frontmatter block that is not parseable YAML;
    - a missing, empty, or non-string ``type`` field.

    Warnings (soft guidance a consumer must tolerate):

    - intra-bundle links whose target concept does not exist (not-yet-written
      knowledge, SPEC sec. 5.3);
    - frontmatter in a non-root ``index.md`` (only the root index may carry
      frontmatter, SPEC sec. 11).

    Returns:
        ``{"conformant": bool, "files": int, "errors": [{"path", "error"}],
        "warnings": [{"path", "warning"}]}`` — ``files`` counts the concept
        documents examined; ``path`` values are bundle-relative.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"OKF bundle root not found: {root_path}")

    errors: list[dict] = []
    warnings: list[dict] = []
    concept_ids: set[str] = set()
    pending_links: list[tuple[str, str]] = []  # (path, target_concept_id)
    files = 0

    for path in sorted(root_path.rglob("*.md")):
        rel = path.relative_to(root_path).as_posix()
        text = path.read_text(encoding="utf-8")

        if path.name in RESERVED_FILENAMES:
            if path.name == "index.md" and rel != "index.md" and text.startswith("---"):
                warnings.append(
                    {"path": rel, "warning": "frontmatter is only permitted in the root index.md"}
                )
            continue

        files += 1
        concept_id = _concept_id_for(path, root_path)
        concept_ids.add(concept_id)

        try:
            frontmatter, body = parse_frontmatter(text)
        except yaml.YAMLError as exc:
            errors.append({"path": rel, "error": f"frontmatter is not valid YAML: {exc}"})
            continue

        if not text.startswith("---"):
            errors.append({"path": rel, "error": "missing frontmatter block"})
        else:
            concept_type = frontmatter.get("type")
            if not isinstance(concept_type, str) or not concept_type.strip():
                errors.append({"path": rel, "error": "missing or empty required field: type"})

        for _anchor, target_id in extract_links(body, concept_id):
            pending_links.append((rel, target_id))

    for rel, target_id in pending_links:
        if target_id not in concept_ids:
            warnings.append({"path": rel, "warning": f"broken link to unknown concept: {target_id}"})

    return {
        "conformant": not errors,
        "files": files,
        "errors": errors,
        "warnings": warnings,
    }


# --- Layered linting (Core / Profile / Hygiene) -------------------------------
#
# `validate_bundle` above is the Core layer: hard SPEC conformance, never
# customizable. `lint_bundle` adds two more layers on top of it, inspired by
# okflint's three-tier model:
#
# - Profile: bundle-specific rules from an optional YAML manifest (e.g. "ADR
#   concepts require a status field"). A rule's `severity` decides whether it
#   blocks `conformant`.
# - Hygiene: built-in best-practice checks for a *knowledge graph* specifically
#   (missing title/description, very short bodies, concepts disconnected from
#   the graph, duplicate titles) — always advisory, never blocks `conformant`.

# Rule keys `_check_profile_rule` understands, beyond `id`/`description`/
# `applies_to`/`severity`.
_PROFILE_RULE_CHECKS = frozenset(
    {"require_field", "forbid_field", "max_length", "allowed_values", "pattern"}
)


def _load_profile(profile: "str | Path | dict | None") -> list[dict]:
    """Load a Profile manifest: a dict, or a path to a YAML file with a `rules` list."""
    if profile is None:
        return []
    if isinstance(profile, dict):
        data = profile
    else:
        path = Path(profile)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = data.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("profile manifest must have a top-level 'rules' list")
    for rule in rules:
        if not isinstance(rule, dict) or "id" not in rule:
            raise ValueError("every profile rule needs an 'id'")
        if not _PROFILE_RULE_CHECKS.intersection(rule):
            raise ValueError(
                f"profile rule {rule['id']!r} has no recognized check "
                f"(expected one of {sorted(_PROFILE_RULE_CHECKS)})"
            )
    return rules


def _rule_applies(rule: dict, concept_type: str | None) -> bool:
    applies_to = rule.get("applies_to", "*")
    if applies_to in (None, "*"):
        return True
    if isinstance(applies_to, list):
        return concept_type in applies_to
    return concept_type == applies_to


def _check_profile_rule(rule: dict, frontmatter: dict) -> list[str]:
    """Return violation messages for `rule` against a concept's frontmatter."""
    messages: list[str] = []
    if "require_field" in rule:
        field = rule["require_field"]
        value = frontmatter.get(field)
        empty = value in ("", [], {}) if rule.get("non_empty", True) else False
        if value is None or empty:
            messages.append(f"missing required field: {field}")
    if "forbid_field" in rule and rule["forbid_field"] in frontmatter:
        messages.append(f"forbidden field present: {rule['forbid_field']}")
    if "max_length" in rule and "field" in rule:
        value = frontmatter.get(rule["field"])
        if isinstance(value, str) and len(value) > rule["max_length"]:
            messages.append(
                f"{rule['field']!r} exceeds max_length {rule['max_length']} ({len(value)} chars)"
            )
    if "allowed_values" in rule and "field" in rule:
        value = frontmatter.get(rule["field"])
        if value is not None and value not in rule["allowed_values"]:
            messages.append(
                f"{rule['field']!r} value {value!r} not in allowed_values {rule['allowed_values']}"
            )
    if "pattern" in rule and "field" in rule:
        value = frontmatter.get(rule["field"])
        if isinstance(value, str) and not re.search(rule["pattern"], value):
            messages.append(f"{rule['field']!r} does not match pattern {rule['pattern']!r}")
    return messages


def lint_bundle(
    root: str | Path,
    *,
    profile: "str | Path | dict | None" = None,
    mode: str = "audit",
    short_body_chars: int = 40,
) -> dict:
    """Lint a bundle in three layers: Core, Profile, and Hygiene.

    Core reuses :func:`validate_bundle` verbatim (hard SPEC conformance,
    sec. 9) — see its docstring for exactly what it checks.

    Profile applies custom rules from ``profile`` (a dict, or a path to a
    YAML file) shaped like::

        rules:
          - id: adr-requires-status
            applies_to: ADR          # a type name, a list of types, or "*" (default)
            require_field: status    # missing, or "" / [] / {} unless non_empty: false
            severity: error          # "error" (blocks `conformant`) or "warning" (default)
          - id: title-max-length
            max_length: 80
            field: title
            severity: warning

    Supported checks (a rule may combine more than one): ``require_field``
    (+ optional ``non_empty``, default ``true``), ``forbid_field``,
    ``max_length`` (+ ``field``), ``allowed_values`` (+ ``field``), ``pattern``
    (a regex, + ``field``).

    Hygiene is a fixed, non-customizable set of best-practice checks for a
    knowledge graph specifically — always advisory, never affects
    ``conformant``: ``missing-title``, ``missing-description``, ``short-body``
    (main body under ``short_body_chars``, excluding the citations section),
    ``orphan-concept`` (no intra-bundle links in or out), and
    ``duplicate-title`` (two concepts sharing a title).

    ``mode="audit"`` (default) returns all three layers — the observational,
    human-facing report. ``mode="validate"`` omits Hygiene (a CI gate cares
    about conformance, not style nits); in both modes ``conformant`` is
    ``True`` only when Core has no errors and no Profile rule with
    ``severity="error"`` was violated — Profile warnings never block it.

    Returns:
        ``{"conformant", "files", "core": {"errors", "warnings"},
        "profile": [{"path", "rule", "message", "severity"}],
        "hygiene": [{"path", "rule", "message"}]}``.
    """
    if mode not in ("audit", "validate"):
        raise ValueError(f"Unknown lint mode: {mode!r} (expected 'audit' or 'validate')")

    root_path = Path(root)
    core_report = validate_bundle(root_path)
    rules = _load_profile(profile)

    # Single extra pass to gather what Profile/Hygiene need: frontmatter, body,
    # and the intra-bundle link graph (in/out degree per concept).
    concepts: list[tuple[str, str, dict, str]] = []  # (rel, concept_id, frontmatter, body)
    id_to_rel: dict[str, str] = {}
    outgoing: dict[str, int] = {}
    incoming: dict[str, int] = {}
    titles: dict[str, list[str]] = {}

    for path in sorted(root_path.rglob("*.md")):
        if path.name in RESERVED_FILENAMES:
            continue
        rel = path.relative_to(root_path).as_posix()
        try:
            frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue  # already reported by the Core layer
        concept_id = _concept_id_for(path, root_path)
        concepts.append((rel, concept_id, frontmatter, body))
        id_to_rel[concept_id] = rel
        outgoing.setdefault(concept_id, 0)
        incoming.setdefault(concept_id, 0)
        main_body, _citations_block = split_citations(body)
        for _anchor, target_id in extract_links(main_body, concept_id):
            outgoing[concept_id] += 1
            incoming[target_id] = incoming.get(target_id, 0) + 1
        title = frontmatter.get("title")
        if isinstance(title, str) and title.strip():
            titles.setdefault(title, []).append(concept_id)

    profile_findings: list[dict] = []
    hygiene_findings: list[dict] = []

    for rel, concept_id, frontmatter, body in concepts:
        concept_type = frontmatter.get("type")
        concept_type = concept_type if isinstance(concept_type, str) else None

        for rule in rules:
            if not _rule_applies(rule, concept_type):
                continue
            for message in _check_profile_rule(rule, frontmatter):
                profile_findings.append(
                    {
                        "path": rel,
                        "rule": rule["id"],
                        "message": message,
                        "severity": rule.get("severity", "warning"),
                    }
                )

        if mode == "audit":
            if not isinstance(frontmatter.get("title"), str) or not frontmatter["title"].strip():
                hygiene_findings.append(
                    {"path": rel, "rule": "missing-title", "message": "no title field"}
                )
            if not isinstance(frontmatter.get("description"), str) or not frontmatter[
                "description"
            ].strip():
                hygiene_findings.append(
                    {"path": rel, "rule": "missing-description", "message": "no description field"}
                )
            main_body, _citations_block = split_citations(body)
            if len(main_body.strip()) < short_body_chars:
                hygiene_findings.append(
                    {
                        "path": rel,
                        "rule": "short-body",
                        "message": f"body is under {short_body_chars} characters",
                    }
                )
            if outgoing[concept_id] == 0 and incoming.get(concept_id, 0) == 0:
                hygiene_findings.append(
                    {"path": rel, "rule": "orphan-concept", "message": "no outgoing or incoming links"}
                )

    if mode == "audit":
        for title, ids in titles.items():
            if len(ids) > 1:
                for concept_id in ids:
                    others = [i for i in ids if i != concept_id]
                    hygiene_findings.append(
                        {
                            "path": id_to_rel[concept_id],
                            "rule": "duplicate-title",
                            "message": f"title {title!r} also used by {others}",
                        }
                    )

    profile_errors = [f for f in profile_findings if f["severity"] == "error"]
    conformant = not core_report["errors"] and not profile_errors

    return {
        "conformant": conformant,
        "files": core_report["files"],
        "core": {"errors": core_report["errors"], "warnings": core_report["warnings"]},
        "profile": profile_findings,
        "hygiene": hygiene_findings if mode == "audit" else [],
    }
