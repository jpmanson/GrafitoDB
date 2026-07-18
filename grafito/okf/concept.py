"""Lightweight OKF views over GrafitoDB nodes.

A :class:`Concept` wraps a grafito :class:`~grafito.models.Node` and speaks the
OKF vocabulary (concept id, type, title, body, links, citations). The raw node
is always available via ``concept.node`` for full graph access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..models import Node

if TYPE_CHECKING:
    from .bundle import OKFBundle


class Concept:
    """An OKF concept: a thin, OKF-flavored view over a graph node."""

    __slots__ = ("node", "_bundle")

    def __init__(self, bundle: "OKFBundle", node: Node) -> None:
        self.node = node
        self._bundle = bundle

    # --- derived accessors -------------------------------------------------

    @property
    def id(self) -> str:
        """Concept ID (the node URI minus the bundle prefix, or its property)."""
        prefix = self._bundle.uri_prefix
        if self.node.uri and self.node.uri.startswith(prefix):
            return self.node.uri[len(prefix):]
        return str(self.node.properties.get("concept_id") or self.node.id)

    @property
    def type(self) -> str:
        return self.node.labels[0] if self.node.labels else "Concept"

    @property
    def title(self) -> str | None:
        return self.node.properties.get("title")

    @property
    def description(self) -> str | None:
        return self.node.properties.get("description")

    @property
    def body(self) -> str:
        return self.node.properties.get("body", "")

    @property
    def tags(self) -> list[str]:
        tags = self.node.properties.get("tags")
        return list(tags) if isinstance(tags, list) else []

    @property
    def properties(self) -> dict[str, Any]:
        return self.node.properties

    @property
    def status(self) -> str | None:
        return self.node.properties.get("status")

    @property
    def is_superseded(self) -> bool:
        return self.status == "superseded"

    @property
    def superseded_by(self) -> str | None:
        return self.node.properties.get("superseded_by")

    @property
    def supersedes(self) -> list[str]:
        """Concept IDs this concept replaces (SPEC trust model)."""
        value = self.node.properties.get("supersedes")
        if isinstance(value, list):
            return list(value)
        return [value] if value else []

    # --- navigation --------------------------------------------------------

    def links(self, *, type: str | None = None) -> list["Concept"]:
        """Concepts this concept links to.

        Follows every outgoing relationship type except ``CITES`` (see
        :meth:`cites`); pass ``type=`` to restrict to one type — useful for
        bundles imported with ``typed_links=True`` (e.g. ``type="JOINS_WITH"``).
        """
        return self._bundle._neighbors(self.id, type, "out")

    def linked_by(self, *, type: str | None = None) -> list["Concept"]:
        """Concepts that link to this one (incoming; any type except ``CITES``)."""
        return self._bundle._neighbors(self.id, type, "in")

    def neighbors(self, *, depth: int = 1, type: str | None = None) -> list["Concept"]:
        """Concepts reachable within ``depth`` outgoing hops."""
        return self._bundle._neighbors(self.id, type, "out", depth=depth)

    def cites(self) -> list[dict]:
        """Citations from this concept: ``[{"url"|"concept", "anchor"}, ...]``."""
        return self._bundle._citations_of(self.id)

    def conflicts(self) -> list["Concept"]:
        """Concepts flagged as contradicting this one (see :meth:`OKFBundle.conflicts_with`)."""
        return self._bundle._neighbors(self.id, "CONFLICTS_WITH", "out")

    def __repr__(self) -> str:
        return f"<Concept {self.id!r} ({self.type})>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Concept) and other.node.id == self.node.id

    def __hash__(self) -> int:
        return hash(self.node.id)


class Proposal:
    """A concept staged for review before it lands in the graph.

    Produced by :meth:`OKFBundle.propose` when a new concept is similar enough
    to existing ones that it needs a human (or heuristic) decision rather than
    being written straight away. Approve with :meth:`OKFBundle.approve`, or
    discard with :meth:`OKFBundle.reject`. A thin, read-only view — mutation
    always goes through the bundle, matching :class:`Concept`.
    """

    __slots__ = ("node", "_bundle")

    def __init__(self, bundle: "OKFBundle", node: Node) -> None:
        self.node = node
        self._bundle = bundle

    @property
    def id(self) -> str:
        return str(self.node.properties.get("concept_id") or self.node.id)

    @property
    def type(self) -> str:
        return self.node.labels[0] if self.node.labels else "Concept"

    @property
    def title(self) -> str | None:
        return self.node.properties.get("title")

    @property
    def body(self) -> str:
        return self.node.properties.get("body", "")

    @property
    def properties(self) -> dict[str, Any]:
        return self.node.properties

    @property
    def similar(self) -> list[dict]:
        """Existing concepts that triggered review, most similar first.

        Each entry is ``{"concept_id", "title", "score", "via"}`` (``via`` is
        ``"semantic"`` or ``"text"``, matching :class:`Hit`).
        """
        value = self.node.properties.get("pending_similar")
        return list(value) if isinstance(value, list) else []

    def __repr__(self) -> str:
        return f"<Proposal {self.id!r} ({self.type})>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Proposal) and other.node.id == self.node.id

    def __hash__(self) -> int:
        return hash(self.node.id)


@dataclass
class Hit:
    """A search result: a concept, its score, and which index produced it."""

    concept: Concept
    score: float
    via: str  # "semantic" | "text"


@dataclass
class ContextPack:
    """Grounded context assembled for an agent prompt, within a token budget.

    The product of :meth:`OKFBundle.context`: a block of ``text`` ready to inject
    into a prompt, the ``citations`` backing it (for grounding/attribution), the
    ``concepts`` that were included, and the seed ``hits`` that started retrieval.
    ``str(pack)`` returns ``pack.text`` so it drops straight into an f-string.

    ``omitted`` explains what retrieval reached but left out — each entry is
    ``{"concept_id", "title", "reason", "via"}`` where ``reason`` is
    ``"budget"`` (didn't fit the token budget), ``"superseded"`` (a retracted
    claim dropped during graph expansion), or ``"reranked_out"`` (a reranker
    with a ``top_n`` cut it). ``trace`` is a compact, deterministic step log of
    how the pack was built, populated only when ``context(..., include_trace=True)``
    — otherwise ``None``. Together they make the pack auditable: an agent (or a
    human) can see not just what grounded the answer, but what didn't and why.
    """

    text: str
    citations: list[dict]
    concepts: list[Concept]
    hits: list[Hit]
    tokens: int
    truncated: bool
    omitted: list[dict] = field(default_factory=list)
    trace: list[dict] | None = None

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.concepts)
