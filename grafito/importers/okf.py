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
- ``sources`` frontmatter -> ``CITES`` relationships, to concepts (intra-bundle)
  or to auto-created ``Reference`` nodes (external URLs, followable artifacts,
  and scope descriptors), carrying the entry's credibility signals
- links under a legacy ``# Citations`` heading -> the same ``CITES``
  relationships (OKF v0.1 form, still consumed per SPEC sec. 13.1)

See https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf for
the format specification.
"""

from __future__ import annotations

import datetime
import hashlib
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

import yaml

if TYPE_CHECKING:
    from ..database import GrafitoDatabase

# Reserved filenames that are not concept documents (SPEC sec. 3.1).
RESERVED_FILENAMES = {"index.md", "log.md"}

# Default label for concepts lacking a `type` (permissive consumption, sec. 11)
# and for stub nodes created from links to not-yet-written concepts (sec. 6.1).
DEFAULT_LABEL = "Concept"

# Label for auto-created nodes representing sources outside the bundle (sec. 5.1).
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

# Legacy `# Citations` section heading (any level). Superseded by the `sources`
# frontmatter in v0.2 (SPEC sec. 13.1), still parsed for v0.1 documents.
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
    that want permissive consumption (SPEC sec. 11) catch it and fall back to
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
        # Bundle-relative (absolute) link (SPEC sec. 6.1).
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
    becomes a stub per SPEC sec. 6.1), a wikilink is only kept when it
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


# --- Provenance: the `sources` frontmatter (SPEC sec. 5.1) --------------------
#
# v0.2 moves provenance out of the body and into frontmatter. `sources` is a
# list of entries, each naming a `resource` plus optional per-source credibility
# signals; `usage_window` is written once as a sibling of `sources` and frames
# every entry's `usage_count`. The importer turns each entry into the same
# `CITES` edge the legacy `# Citations` list produced, so provenance is
# graph-queryable however it was authored.

# Optional per-source credibility signals carried on the edge (sec. 5.1).
SOURCE_SIGNALS = ("author", "usage_count", "last_modified", "usage_window")

# Values of the `via` property on a CITES edge: which form the provenance was
# authored in. The exporter reads it to write each edge back the same way,
# instead of emitting a v0.1 citation list and a v0.2 `sources` block for the
# same fact (which would double the edges on the next import).
VIA_SOURCES = "sources"  # frontmatter `sources` (v0.2), or a programmatic cite()
VIA_CITATIONS = "citations"  # legacy body `# Citations` list (v0.1)


def normalize_sources(frontmatter: dict) -> list[dict]:
    """Return the ``sources`` frontmatter as a list of normalized entries.

    Applies the shared ``usage_window`` sibling as each entry's default window,
    tolerates a lone entry written as a bare mapping, and drops anything that is
    not a mapping carrying a non-empty ``resource`` (REQUIRED within an entry).
    Permissive consumption (sec. 11): one malformed entry never costs the others.
    """
    raw = frontmatter.get("sources")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    shared_window = frontmatter.get("usage_window")
    entries: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        resource = item.get("resource")
        if not isinstance(resource, str) or not resource.strip():
            continue
        entry: dict = {"resource": resource.strip()}
        for key in ("id", "title", *SOURCE_SIGNALS):
            value = item.get(key)
            if value is not None:
                entry[key] = value
        if "usage_window" not in entry and shared_window is not None:
            entry["usage_window"] = shared_window
        entries.append(entry)
    return entries


def classify_source(resource: str, source_id: str) -> tuple[str, str]:
    """Classify a ``sources[].resource`` for edge resolution (sec. 5.1).

    Returns ``("external", url)``, ``("concept", concept_id)`` for a path into
    the bundle, or ``("scope", descriptor)`` for a population or scope
    descriptor a consumer cannot follow (``all queries in BigQuery project X``).
    A descriptor is told apart from a path by containing whitespace — paths and
    URLs never do, and the SPEC gives no other marker.
    """
    resource = resource.strip()
    if not resource:
        return "scope", resource
    if resource.startswith(_EXTERNAL_PREFIXES):
        return "external", resource
    if any(char.isspace() for char in resource):
        return "scope", resource
    classified = classify_target(resource, source_id)
    return classified if classified is not None else ("scope", resource)


# --- Trust and lifecycle: `generated`, `verified`, `status`, `stale_after` ----
#
# These families (SPEC sec. 5.2-5.5) answer "who wrote this", "who confirmed
# it", "is it current", and "is it still true". They are read, never computed
# and stored: a trust tier derived at read time cannot go stale against the
# `verified` list it came from, and OKF deliberately stores the signals rather
# than a verdict. Every helper here takes raw frontmatter — the same dict shape
# whether it came from YAML (where dates parse to `date`/`datetime`) or from a
# node's properties (where grafito has already normalized them to ISO text).

# Lifecycle vocabulary (sec. 5.4). An absent `status` reads as `stable`.
LIFECYCLE_STATUSES = ("draft", "stable", "deprecated")
DEFAULT_STATUS = "stable"

# Trust tiers, lowest to highest (sec. 5.3).
TRUST_TIERS = ("unverified", "machine-confirmed", "human-reviewed")

# The actor prefix trust classification keys off (sec. 7).
HUMAN_ACTOR_PREFIX = "human:"


def _iso_text(value: Any) -> str | None:
    """ISO text for a date-ish frontmatter value, or ``None`` when unusable."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return None


def _as_date(value: Any) -> "datetime.date | None":
    """A calendar date from a `date`, `datetime`, or ISO string; else ``None``."""
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        try:
            return datetime.date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def normalize_event(value: Any) -> dict | None:
    """One ``{by, at}`` trust event (sec. 5.2), or ``None`` if it names no actor.

    ``by`` is REQUIRED within the mapping; ``at`` is optional, so an event that
    records only who confirmed something is still an event.
    """
    if not isinstance(value, dict):
        return None
    by = value.get("by")
    if not isinstance(by, str) or not by.strip():
        return None
    event = {"by": by.strip()}
    at = _iso_text(value.get("at"))
    if at:
        event["at"] = at
    return event


def normalize_verified(frontmatter: dict) -> list[dict]:
    """``verified`` as a list of ``{by, at}`` events (sec. 5.2).

    A single verifier may be written as a bare mapping without the list dash;
    consumers MUST read that as a one-element list (sec. 11). Multiple entries
    capture independent checks, for example a human sign-off plus a nightly
    process.
    """
    raw = frontmatter.get("verified")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [event for event in (normalize_event(item) for item in raw) if event is not None]


def trust_tier(frontmatter: dict) -> str:
    """The concept's trust tier, derived from ``verified`` (sec. 5.3).

    ``unverified`` with no ``verified`` key, ``human-reviewed`` when any
    verifier is a ``human:`` actor, ``machine-confirmed`` otherwise. Advisory
    only: a concept with no trust frontmatter is still consumable (sec. 11).
    """
    events = normalize_verified(frontmatter)
    if not events:
        return "unverified"
    if any(event["by"].startswith(HUMAN_ACTOR_PREFIX) for event in events):
        return "human-reviewed"
    return "machine-confirmed"


def verified_at(frontmatter: dict) -> str | None:
    """When the concept was most recently confirmed — the latest ``at`` (sec. 5.2)."""
    stamps = [event["at"] for event in normalize_verified(frontmatter) if "at" in event]
    return max(stamps) if stamps else None


def generated_at(frontmatter: dict) -> str | None:
    """When the content last meaningfully changed (sec. 5.2).

    Falls back to a legacy v0.1 ``timestamp`` when ``generated`` is absent —
    the fallback v0.2 explicitly allows for v0.1 documents (sec. 13.1).
    """
    generated = frontmatter.get("generated")
    if isinstance(generated, dict):
        at = _iso_text(generated.get("at"))
        if at:
            return at
    return _iso_text(frontmatter.get("timestamp"))


def generated_by(frontmatter: dict) -> str | None:
    """The actor that produced the current content (sec. 5.2), or ``None``."""
    generated = frontmatter.get("generated")
    if not isinstance(generated, dict):
        return None
    by = generated.get("by")
    return by.strip() if isinstance(by, str) and by.strip() else None


def is_stale(frontmatter: dict, *, today: "datetime.date | None" = None) -> bool:
    """Whether ``today >= stale_after`` (sec. 5.5).

    ``False`` when the concept declares no ``stale_after``, or declares one that
    is not a readable date — staleness is an assertion the producer makes, never
    something a consumer infers from silence.
    """
    deadline = _as_date(frontmatter.get("stale_after"))
    if deadline is None:
        return False
    return (today or datetime.date.today()) >= deadline


# --- Attested computations: the computation family (SPEC sec. 10) ------------
#
# An Attested Computation concept carries a sanctioned way to compute a value
# plus the means to confirm a run produced it that way. Three of its fields name
# paths (sec. 6.2) pointing outside the concept: the computation file, the
# executor's run instructions, and the attester's code. Materializing them as
# edges makes "what runs this" and "what checks it" traversable, the same way
# `sources` makes provenance traversable — grafito never executes any of it.

# The concept type these fields belong to (sec. 10.1).
COMPUTATION_TYPE = "Attested Computation"

# Path-valued computation field -> the relationship it produces. `computation`
# holds the path itself; `executor`/`attester` are mappings whose `resource`
# key holds it (sec. 10.2).
COMPUTATION_REL_TYPES = {
    "computation": "HAS_COMPUTATION",
    "executor": "EXECUTED_BY",
    "attester": "ATTESTED_BY",
}

# `# Computation` body fence heading (sec. 4.2), the inline alternative to a
# `computation` path.
_COMPUTATION_HEADING_RE = re.compile(r"^#{1,6}\s+Computation\s*$", re.IGNORECASE | re.MULTILINE)


def computation_paths(frontmatter: dict) -> list[tuple[str, str]]:
    """Return ``[(field, path), ...]`` for the computation family (sec. 10.2).

    Driven by field presence rather than ``type``: ``type`` is free-form, and a
    concept declaring an ``attester`` is making the same assertion whatever it
    calls itself. Values that name nothing followable are skipped, so a
    ``computation`` holding inline text, or an ``executor`` mapping with no
    ``resource``, simply stays an ordinary property.
    """
    paths: list[tuple[str, str]] = []
    for field in COMPUTATION_REL_TYPES:
        value = frontmatter.get(field)
        candidate = value if field == "computation" else (
            value.get("resource") if isinstance(value, dict) else None
        )
        if isinstance(candidate, str) and candidate.strip():
            paths.append((field, candidate.strip()))
    return paths


def has_inline_computation(body: str) -> bool:
    """Whether the body carries a ``# Computation`` section (sec. 10.3)."""
    return bool(_COMPUTATION_HEADING_RE.search(body or ""))


def _concept_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return rel[: -len(".md")] if rel.endswith(".md") else rel


# Log entry: a list bullet under a `## YYYY-MM-DD` date heading (SPEC sec. 9).
_LOG_DATE_RE = re.compile(r"^#{1,6}\s+(\d{4}-\d{2}-\d{2})\s*$")
_LOG_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
_LOG_KIND_RE = re.compile(r"^\*\*([^*]+)\*\*:?\s*(.*)$")


def parse_log_entries(text: str) -> list[tuple[str, str | None, str]]:
    """Parse a ``log.md`` into ``[(date, kind, text), ...]`` (SPEC sec. 9).

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
        citations: Import provenance as ``citation_type`` relationships (to
            concepts or auto-created ``Reference`` nodes) — from the ``sources``
            frontmatter (SPEC v0.2 sec. 5.1) and from a legacy ``# Citations``
            body section (v0.1, sec. 13.1). Each edge records which form it came
            from in its ``via`` property, and a ``sources`` edge also carries the
            entry's ``source_id`` and credibility signals (``author``,
            ``usage_count``, ``last_modified``, ``usage_window``).
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
        ``relationships``, ``citations`` (every citation edge), ``sources``
        (how many of those came from the ``sources`` frontmatter rather than a
        legacy body list), ``computations`` (edges from the computation family,
        sec. 10.2), ``references`` (new ``Reference``
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
    # (source_id, normalized `sources` entry)
    pending_sources: list[tuple[str, dict]] = []
    # (source_id, computation field, path)
    pending_computations: list[tuple[str, str, str]] = []
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
            if citations:
                for entry in normalize_sources(frontmatter):
                    pending_sources.append((concept_id, entry))
            for field, computation_path in computation_paths(frontmatter):
                pending_computations.append((concept_id, field, computation_path))
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

        # Second pass: resolve links, creating stubs for missing targets (sec. 6.1).
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

        # Citations: link to concepts (intra-bundle) or to Reference nodes
        # (external URLs, followable artifacts, scope descriptors).
        # `reference_nodes` is pre-seeded with existing Reference nodes in
        # incremental mode, so only genuinely new resources create a node here.
        citation_count = 0
        new_references = 0

        def resolve_reference(resource: str, title: str | None, *, scope: bool = False) -> int:
            nonlocal new_references
            if resource not in reference_nodes:
                properties: dict[str, Any] = {
                    "title": title or resource,
                    "url": resource,
                    "resource": resource,
                    "okf_auto": True,
                }
                if scope:
                    # A population descriptor, not an artifact to follow (sec. 5.1).
                    properties["scope_descriptor"] = True
                ref = db.create_node(labels=[REFERENCE_LABEL], properties=properties, uri=resource)
                reference_nodes[resource] = ref.id
                new_references += 1
            return reference_nodes[resource]

        def resolve_path_field(raw: str, source_id: str, title: str | None = None) -> int:
            """Resolve a path-valued field (sec. 6.2) to the node it names.

            A path is a concept when the bundle has one, or when it is
            explicitly a ``.md`` document not written yet (sec. 6.1). Anything
            else followable — an attester's ``.py``, a dashboard path — becomes
            a ``Reference``, so these fields never litter the graph with stub
            *concepts* for files that are not concepts.

            A plain relative path is tried twice: once relative to the citing
            concept, then once from the bundle root. The SPEC writes executors
            and attesters as ``references/skills/run-on-bq.md`` from a concept
            inside ``computations/``, while placing that tree at the root
            (sec. 6.3) — only the second reading finds it, and a path that
            resolves under neither reading is unaffected by trying both.
            """
            kind, value = classify_source(raw, source_id)
            if kind == "concept":
                if value in concept_to_node:
                    return concept_to_node[value]
                if not raw.startswith(("/", ".")):
                    rooted = raw[: -len(".md")] if raw.endswith(".md") else raw
                    if rooted in concept_to_node:
                        return concept_to_node[rooted]
                if raw.endswith(".md"):
                    return resolve_concept(value)
            return resolve_reference(raw, title, scope=kind == "scope")

        for source_id, anchor, kind, value in pending_citations:
            target = resolve_concept(value) if kind == "concept" else resolve_reference(value, anchor)
            edge_props = {"via": VIA_CITATIONS}
            if anchor:
                edge_props["anchor"] = anchor
            db.create_relationship(concept_to_node[source_id], target, citation_type, edge_props)
            citation_count += 1

        # Frontmatter provenance (sec. 5.1): the same `citation_type` edges, but
        # tagged `via=VIA_SOURCES` so the exporter writes them back to
        # frontmatter, and carrying the entry's id, title, and credibility
        # signals — a source's authority, adoption, and recency are per *edge*,
        # since two concepts can cite one resource over different usage windows.
        source_edges = 0
        for source_id, entry in pending_sources:
            resource = entry["resource"]
            target = resolve_path_field(resource, source_id, entry.get("title"))
            edge_props = {"via": VIA_SOURCES}
            if entry.get("title"):
                edge_props["anchor"] = entry["title"]
            if entry.get("id"):
                edge_props["source_id"] = entry["id"]
            for signal in SOURCE_SIGNALS:
                if signal in entry:
                    edge_props[signal] = entry[signal]
            db.create_relationship(concept_to_node[source_id], target, citation_type, edge_props)
            citation_count += 1
            source_edges += 1
        if stride:
            report("citations", citation_count, end=True)

        # The computation family (sec. 10.2): the paths naming what runs a
        # computation and what checks it become edges, resolved exactly like a
        # source path — a known concept, a `.md` document not written yet, or a
        # Reference for anything else (an attester is usually a `.py` file).
        computation_edges = 0
        for source_id, field, computation_path in pending_computations:
            if classify_source(computation_path, source_id)[0] == "scope":
                continue  # not a path at all: inline content, left as a property
            target = resolve_path_field(computation_path, source_id)
            db.create_relationship(
                concept_to_node[source_id],
                target,
                COMPUTATION_REL_TYPES[field],
                {"via": field},
            )
            computation_edges += 1

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
        "sources": source_edges,
        "computations": computation_edges,
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


# Prefix of the Core-layer warning emitted for an intra-bundle link whose target
# concept is absent. Shared with :func:`diff_okf_bundles`, which recovers the
# target id from the warning rather than re-deriving broken links itself.
_BROKEN_LINK_WARNING = "broken link to unknown concept: "


def _source_issues(frontmatter: dict) -> list[str]:
    """Soft problems in a concept's ``sources`` block (SPEC sec. 5.1).

    Never an error: the family is optional and a consumer must not reject a
    concept over it (sec. 11). But an entry with no ``resource`` names nothing
    followable, so it is silently dropped on import — worth reporting.
    """
    raw = frontmatter.get("sources")
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return ["'sources' is not a list of entries"]
    issues: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            issues.append(f"sources[{index}] is not a mapping")
        elif not isinstance(item.get("resource"), str) or not item["resource"].strip():
            issues.append(f"sources[{index}] is missing the required field: resource")
    return issues


def _trust_issues(frontmatter: dict) -> list[str]:
    """Soft problems in the trust and lifecycle families (SPEC sec. 5.2-5.5).

    All warnings, never errors: these families are optional and a consumer must
    not reject a concept over them (sec. 11). What is worth reporting is a field
    that is *present but unreadable*, since it silently stops carrying the
    meaning its author intended — an event with no actor cannot set a trust
    tier, a `stale_after` that is not a date can never make a concept stale, and
    a `status` outside the vocabulary is not the lifecycle signal it looks like.
    """
    issues: list[str] = []

    status = frontmatter.get("status")
    if status is not None and status not in LIFECYCLE_STATUSES:
        issues.append(
            f"'status' value {status!r} is not one of {list(LIFECYCLE_STATUSES)}"
        )

    generated = frontmatter.get("generated")
    if generated is not None:
        if not isinstance(generated, dict):
            issues.append("'generated' is not a mapping")
        elif normalize_event(generated) is None:
            issues.append("'generated' is missing the required field: by")

    verified = frontmatter.get("verified")
    if verified is not None:
        raw = [verified] if isinstance(verified, dict) else verified
        if not isinstance(raw, list):
            issues.append("'verified' is not a list of events")
        else:
            for index, item in enumerate(raw):
                if normalize_event(item) is None:
                    issues.append(f"verified[{index}] is missing the required field: by")

    stale_after = frontmatter.get("stale_after")
    if stale_after is not None and _as_date(stale_after) is None:
        issues.append(f"'stale_after' value {stale_after!r} is not a YYYY-MM-DD date")

    return issues


def _computation_issues(frontmatter: dict, body: str) -> list[str]:
    """Soft problems in the computation family (SPEC sec. 10.2-10.3).

    Warnings, not errors: conformance (sec. 11) asks only for a parseable
    frontmatter with a ``type``, and says a producer SHOULD follow sec. 10 when
    the family is present. What is reported is a contract a consumer could not
    honour — an Attested Computation with no ``runtime`` cannot be interpreted,
    and one with neither an inline fence nor a ``computation`` path has nothing
    to run.
    """
    issues: list[str] = []

    for field in ("executor", "attester"):
        value = frontmatter.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            issues.append(f"{field!r} is not a mapping")
        elif not isinstance(value.get("resource"), str) or not value["resource"].strip():
            issues.append(f"{field!r} is missing the required field: resource")

    parameters = frontmatter.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, list):
            issues.append("'parameters' is not a list")
        else:
            for index, item in enumerate(parameters):
                if not isinstance(item, dict) or not item.get("name"):
                    issues.append(f"parameters[{index}] is missing the required field: name")

    if frontmatter.get("type") == COMPUTATION_TYPE:
        runtime = frontmatter.get("runtime")
        if not isinstance(runtime, str) or not runtime.strip():
            issues.append(
                f"missing the required field: runtime (required for type {COMPUTATION_TYPE!r})"
            )
        if not frontmatter.get("computation") and not has_inline_computation(body):
            issues.append(
                "no computation: expected a 'computation' path or a '# Computation' body section"
            )

    return issues


def validate_bundle(root: str | Path) -> dict:
    """Validate a bundle against the OKF v0.2 conformance rules (SPEC sec. 11).

    Reports problems without importing anything and without aborting on the
    first bad file — the linter counterpart to the importer's permissive
    consumption.

    Errors (conformance failures):

    - a non-reserved ``.md`` file with no frontmatter block;
    - a frontmatter block that is not parseable YAML;
    - a missing, empty, or non-string ``type`` field.

    Warnings (soft guidance a consumer must tolerate):

    - intra-bundle links whose target concept does not exist (not-yet-written
      knowledge, SPEC sec. 6.1);
    - a ``sources`` entry that is malformed or carries no ``resource``
      (sec. 5.1) — dropped on import;
    - a trust or lifecycle field that is present but unreadable: a ``status``
      outside the vocabulary, a ``generated``/``verified`` event with no
      ``by`` actor, or a ``stale_after`` that is not a date (sec. 5.2-5.5);
    - a computation contract a consumer could not honour: an Attested
      Computation with no ``runtime`` or no computation at all, a malformed
      ``executor``/``attester``, or a parameter with no ``name`` (sec. 10.2);
    - frontmatter in a non-root ``index.md`` (only the root index may carry
      frontmatter, and only ``okf_version``, SPEC sec. 12).

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

        issues = (
            _source_issues(frontmatter)
            + _trust_issues(frontmatter)
            + _computation_issues(frontmatter, body)
        )
        for issue in issues:
            warnings.append({"path": rel, "warning": issue})

        for _anchor, target_id in extract_links(body, concept_id):
            pending_links.append((rel, target_id))

    for rel, target_id in pending_links:
        if target_id not in concept_ids:
            warnings.append({"path": rel, "warning": f"{_BROKEN_LINK_WARNING}{target_id}"})

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


# --- Bundle diff (read-only preview) ------------------------------------------
#
# `diff_okf_bundles` previews what would change if `candidate` replaced `base`,
# without importing either side. It is a pure, read-only dry-run of the
# *incremental* importer: the content hash it compares is byte-for-byte the same
# `okf_hash` the importer stores and re-checks on re-import — `sha256` over the
# newline-normalized file text (`read_text` translates CRLF -> LF, so the hash
# already matches regardless of platform line endings). That lockstep is the
# whole point: "changed here" iff "the importer would update that node". The
# `invalid` and `broken_links` fields are delegated to `validate_bundle` so the
# OKF conformance rules keep living in exactly one place.


@dataclass(frozen=True)
class ConceptDelta:
    """Field-level change for a concept whose file differs between two trees.

    Frontmatter is diffed key-by-key (``type`` included, so a retype shows up as
    a ``frontmatter_changed`` entry); ``body_changed`` covers the markdown body
    below the frontmatter block.
    """

    frontmatter_added: dict[str, Any]  # key -> value present only in candidate
    frontmatter_removed: dict[str, Any]  # key -> value present only in base
    frontmatter_changed: dict[str, tuple[Any, Any]]  # key -> (base, candidate)
    body_changed: bool


@dataclass(frozen=True)
class BundleDiff:
    """The read-only preview of replacing ``base`` with ``candidate``.

    Paths are bundle-relative POSIX strings (e.g. ``decisions/0001-x.md``),
    matching :func:`validate_bundle`'s reporting. Reserved files (``index.md``,
    ``log.md``) are excluded, exactly as the importer excludes them from
    concepts. ``invalid`` and ``broken_links`` describe the *candidate* — the
    tree about to be adopted.
    """

    added: list[str]  # concepts present only in candidate
    removed: list[str]  # concepts present only in base
    changed: dict[str, ConceptDelta]  # concept path -> what changed
    invalid: dict[str, str]  # candidate path -> conformance error
    broken_links: list[tuple[str, str]]  # (candidate path, missing target id)

    def summary(self) -> dict[str, int]:
        """Counts for a one-line report."""
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "changed": len(self.changed),
            "invalid": len(self.invalid),
            "broken_links": len(self.broken_links),
        }

    @property
    def has_changes(self) -> bool:
        """Whether adopting the candidate would add, remove, or change anything."""
        return bool(self.added or self.removed or self.changed)

    @property
    def conformant(self) -> bool:
        """Whether the candidate is free of hard conformance errors.

        Broken links are warnings (SPEC sec. 9), so they never make a candidate
        non-conformant — mirroring :func:`validate_bundle`.
        """
        return not self.invalid


def _concept_hashes(root: Path) -> dict[str, str]:
    """Map every non-reserved concept path to the importer's ``okf_hash``.

    Read and hashed exactly as :func:`import_bundle` does, so the result is
    comparable to the ``okf_hash`` stored on imported nodes.
    """
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*.md")):
        if path.name in RESERVED_FILENAMES:
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        hashes[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashes


def _safe_parse(text: str) -> tuple[dict, str]:
    """Parse frontmatter permissively (SPEC sec. 9), like the importer does.

    A candidate file with malformed YAML is still reported in ``invalid`` by
    :func:`validate_bundle`; here we only need a best-effort delta, so an
    unparseable block degrades to an empty frontmatter with the whole text as
    body rather than raising.
    """
    try:
        return parse_frontmatter(text)
    except yaml.YAMLError:
        return {}, text


def _concept_delta(base_text: str, candidate_text: str) -> ConceptDelta:
    base_fm, base_body = _safe_parse(base_text)
    cand_fm, cand_body = _safe_parse(candidate_text)
    added = {k: v for k, v in cand_fm.items() if k not in base_fm}
    removed = {k: v for k, v in base_fm.items() if k not in cand_fm}
    changed = {
        k: (base_fm[k], cand_fm[k])
        for k in base_fm
        if k in cand_fm and base_fm[k] != cand_fm[k]
    }
    return ConceptDelta(
        frontmatter_added=added,
        frontmatter_removed=removed,
        frontmatter_changed=changed,
        body_changed=base_body != cand_body,
    )


def diff_okf_bundles(base: str | Path, candidate: str | Path) -> BundleDiff:
    """Preview what adopting ``candidate`` in place of ``base`` would change.

    A pure, read-only, domain-agnostic diff between two OKF trees on disk —
    neither is imported, no graph, no LLM, no network. It is the safe-replace
    companion to :func:`validate_bundle`/:func:`lint_bundle`: build a candidate
    bundle into a staging directory, diff it against the live one, show a human
    the ``added``/``removed``/``changed`` concepts plus any ``invalid`` files or
    ``broken_links``, and only then swap the directories.

    Because the content hash matches the importer's ``okf_hash`` byte-for-byte,
    the ``changed`` set is exactly the set of concepts a subsequent incremental
    ``import_okf_bundle`` would re-process — this preview never disagrees with
    what the import actually does.

    Args:
        base: the current bundle (the "before" tree).
        candidate: the proposed bundle (the "after" tree).

    Returns:
        A :class:`BundleDiff`. See its fields for the exact shape.

    Raises:
        NotADirectoryError: if either path is not an existing directory.
    """
    base_root = Path(base)
    cand_root = Path(candidate)
    for role, root in (("base", base_root), ("candidate", cand_root)):
        if not root.is_dir():
            raise NotADirectoryError(f"OKF {role} bundle root not found: {root}")

    base_hashes = _concept_hashes(base_root)
    cand_hashes = _concept_hashes(cand_root)
    base_paths, cand_paths = set(base_hashes), set(cand_hashes)

    added = sorted(cand_paths - base_paths)
    removed = sorted(base_paths - cand_paths)

    # Field-level delta only for the (usually small) set of files whose content
    # actually moved — re-read just those, rather than holding every body in RAM.
    changed: dict[str, ConceptDelta] = {}
    for rel in sorted(base_paths & cand_paths):
        if base_hashes[rel] == cand_hashes[rel]:
            continue
        changed[rel] = _concept_delta(
            (base_root / rel).read_text(encoding="utf-8"),
            (cand_root / rel).read_text(encoding="utf-8"),
        )

    # Reuse the Core conformance pass for the candidate: `invalid` and
    # `broken_links` are exactly its errors and its broken-link warnings, so the
    # rules are never re-implemented here.
    report = validate_bundle(cand_root)
    invalid = {e["path"]: e["error"] for e in report["errors"]}
    broken_links = [
        (w["path"], w["warning"][len(_BROKEN_LINK_WARNING) :])
        for w in report["warnings"]
        if w["warning"].startswith(_BROKEN_LINK_WARNING)
    ]

    return BundleDiff(
        added=added,
        removed=removed,
        changed=changed,
        invalid=invalid,
        broken_links=broken_links,
    )
