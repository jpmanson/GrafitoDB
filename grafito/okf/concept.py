"""Lightweight OKF views over GrafitoDB nodes.

A :class:`Concept` wraps a grafito :class:`~grafito.models.Node` and speaks the
OKF vocabulary (concept id, type, title, body, links, citations). The raw node
is always available via ``concept.node`` for full graph access.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    # --- navigation --------------------------------------------------------

    def links(self, *, type: str = "LINKS_TO") -> list["Concept"]:
        """Concepts this concept links to (outgoing ``LINKS_TO`` by default)."""
        return self._bundle._neighbors(self.id, type, "out")

    def linked_by(self, *, type: str = "LINKS_TO") -> list["Concept"]:
        """Concepts that link to this one (incoming)."""
        return self._bundle._neighbors(self.id, type, "in")

    def neighbors(self, *, depth: int = 1, type: str = "LINKS_TO") -> list["Concept"]:
        """Concepts reachable within ``depth`` outgoing hops."""
        return self._bundle._neighbors(self.id, type, "out", depth=depth)

    def cites(self) -> list[dict]:
        """Citations from this concept: ``[{"url"|"concept", "anchor"}, ...]``."""
        return self._bundle._citations_of(self.id)

    def __repr__(self) -> str:
        return f"<Concept {self.id!r} ({self.type})>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Concept) and other.node.id == self.node.id

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
    """

    text: str
    citations: list[dict]
    concepts: list[Concept]
    hits: list[Hit]
    tokens: int
    truncated: bool

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.concepts)
