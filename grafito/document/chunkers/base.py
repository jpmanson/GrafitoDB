"""Chunker protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..types import ChunkSpec


@runtime_checkable
class Chunker(Protocol):
    """Pure text → list[ChunkSpec] splitter."""

    name: str

    def split(self, text: str, *, meta: dict[str, Any] | None = None) -> list[ChunkSpec]:
        ...
