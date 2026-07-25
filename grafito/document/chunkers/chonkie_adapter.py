"""Optional adapter mapping Chonkie chunkers → ChunkSpec (extra: ``grafito[document-chonkie]``)."""

from __future__ import annotations

from typing import Any

from ..types import ChunkSpec


class ChonkieChunker:
    """Wrap a `Chonkie <https://github.com/chonkie-ai/chonkie>`_ chunker.

    Chonkie provides mature, token-aware splitters (``TokenChunker``,
    ``RecursiveChunker``, ``SentenceChunker``, …). This adapter maps their
    ``Chunk`` objects to Grafito :class:`ChunkSpec` so passages carry text,
    offsets and token counts.

    Two ways to build it:

    - pass an already-constructed chunker::

        from chonkie import RecursiveChunker
        ChonkieChunker(RecursiveChunker(chunk_size=512))

    - or name a chunker class and let the adapter import it lazily::

        ChonkieChunker.from_recipe("recursive", chunk_size=512)

    Any object exposing ``chunk(text) -> Iterable`` of chunk-like items works,
    so tests can inject a stub without installing ``chonkie``.
    """

    def __init__(self, chunker: Any, *, name: str | None = None) -> None:
        if not hasattr(chunker, "chunk"):
            raise TypeError(
                "ChonkieChunker expects a Chonkie chunker (object with a .chunk() method); "
                f"got {type(chunker).__name__}"
            )
        self._chunker = chunker
        self.name = name or f"chonkie:{type(chunker).__name__}"

    @classmethod
    def from_recipe(cls, recipe: str, **kwargs: Any) -> "ChonkieChunker":
        """Build a Chonkie chunker by short name (imports ``chonkie`` lazily).

        ``recipe`` is one of ``token``, ``sentence``, ``recursive``, ``semantic``
        (mapped to ``TokenChunker`` / ``SentenceChunker`` / ``RecursiveChunker`` /
        ``SemanticChunker``). Raises ``ImportError`` if ``chonkie`` is missing.
        """
        try:
            import chonkie  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "ChonkieChunker.from_recipe requires the optional dependency 'chonkie'. "
                "Install it with: pip install 'grafito[document-chonkie]'"
            ) from exc

        classes = {
            "token": "TokenChunker",
            "sentence": "SentenceChunker",
            "recursive": "RecursiveChunker",
            "semantic": "SemanticChunker",
        }
        cls_name = classes.get(recipe.lower())
        if cls_name is None:
            raise ValueError(
                f"Unknown Chonkie recipe {recipe!r}; choose one of {sorted(classes)}"
            )
        chunker_cls = getattr(chonkie, cls_name)
        return cls(chunker_cls(**kwargs), name=f"chonkie:{recipe.lower()}")

    def split(self, text: str, *, meta: dict[str, Any] | None = None) -> list[ChunkSpec]:
        if not text:
            return []
        raw = self._chunker.chunk(text)
        specs: list[ChunkSpec] = []
        for i, chunk in enumerate(raw):
            specs.append(self._to_spec(chunk, i))
        # Renumber ord densely (skip any empties)
        specs = [s for s in specs if s.text]
        for i, s in enumerate(specs):
            s.ord = i
        return specs

    def _to_spec(self, chunk: Any, ord_i: int) -> ChunkSpec:
        # Chonkie Chunk: .text, .start_index, .end_index, .token_count
        text = getattr(chunk, "text", None)
        if text is None:
            text = str(chunk)
        return ChunkSpec(
            text=text,
            ord=ord_i,
            char_start=_int_or_none(getattr(chunk, "start_index", None)),
            char_end=_int_or_none(getattr(chunk, "end_index", None)),
            token_count=_int_or_none(getattr(chunk, "token_count", None)),
            strategy=self.name,
        )


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
