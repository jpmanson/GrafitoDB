"""Pluggable rerankers for OKF retrieval.

A :class:`Reranker` is any callable that re-scores concept candidates against the
query *text* — the precision step a bi-encoder (embedding) retrieval can't do on
its own. It is injected, never imported by the core: :meth:`OKFBundle.context`
takes a ``rerank=`` argument and uses whatever order the reranker returns.

This module ships two references:

* :class:`LexicalReranker` — dependency-free (query-term overlap). A sensible
  offline default/fallback; good enough to demonstrate the hook.
* :class:`CohereReranker` — a thin adapter over Cohere's cross-encoder rerank
  API, the production-grade option. Mirror it for Voyage/Jina/etc.
"""

from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .concept import Concept

_TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_RERANK_FIELDS = ("title", "description", "body")


@runtime_checkable
class Reranker(Protocol):
    """Re-scores concept candidates against a query, most relevant first.

    Returns ``(concept, score)`` pairs in descending relevance. A reranker may
    return fewer candidates than it was given (e.g. its own ``top_n``); callers
    use exactly the order and subset returned.
    """

    def __call__(self, query: str, candidates: list["Concept"]) -> list[tuple["Concept", float]]:
        ...


def concept_text(concept: "Concept", *, fields: tuple[str, ...] = DEFAULT_RERANK_FIELDS,
                 max_chars: int | None = None) -> str:
    """Render a concept to the text a reranker scores (title/description/body)."""
    parts: list[str] = []
    for field in fields:
        value = getattr(concept, field, None)
        if value:
            parts.append(str(value))
    text = "\n".join(parts)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


class LexicalReranker:
    """Dependency-free reranker scoring candidates by query-term overlap.

    A reasonable offline default and a fallback when no rerank service is wired.
    Production setups should inject a cross-encoder (see :class:`CohereReranker`).
    """

    def __init__(self, *, fields: tuple[str, ...] = DEFAULT_RERANK_FIELDS) -> None:
        self._fields = fields

    def __call__(self, query: str, candidates: list["Concept"]) -> list[tuple["Concept", float]]:
        query_terms = set(_TOKEN_RE.findall(query.lower()))
        scored: list[tuple["Concept", float]] = []
        for concept in candidates:
            tokens = _TOKEN_RE.findall(concept_text(concept, fields=self._fields).lower())
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            # Sum of log-weighted term frequencies for matching query terms,
            # length-normalized so long bodies don't win by sheer size.
            raw = sum(1.0 + math.log(tf[t]) for t in query_terms if t in tf)
            score = raw / math.log(2 + len(tokens))
            scored.append((concept, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


class CohereReranker:
    """Adapter over the Cohere rerank API (cross-encoder relevance scoring)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
        *,
        api_key_env_var: str | None = None,
        base_url: str = "https://api.cohere.ai/v1/rerank",
        top_n: int | None = None,
        fields: tuple[str, ...] = DEFAULT_RERANK_FIELDS,
        max_chars: int = 2000,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ValueError("httpx is not installed. Install with `pip install httpx`.") from exc

        if api_key is None:
            api_key = os.environ.get(api_key_env_var) if api_key_env_var else os.environ.get(
                "COHERE_API_KEY"
            )
        if not api_key:
            raise ValueError(
                "Cohere API key not provided. Set COHERE_API_KEY or pass api_key explicitly."
            )

        self.model = model
        self.top_n = top_n
        self._fields = fields
        self._max_chars = max_chars
        self._api_url = base_url
        self._session = httpx.Client()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def __call__(self, query: str, candidates: list["Concept"]) -> list[tuple["Concept", float]]:
        if not candidates:
            return []
        documents = [
            concept_text(c, fields=self._fields, max_chars=self._max_chars) for c in candidates
        ]
        payload: dict[str, Any] = {"model": self.model, "query": query, "documents": documents}
        if self.top_n is not None:
            payload["top_n"] = self.top_n
        response = self._session.post(self._api_url, json=payload)
        data = response.json()
        if isinstance(data, dict) and data.get("message") and "results" not in data:
            raise ValueError(f"Cohere rerank API error: {data['message']}")
        return [
            (candidates[item["index"]], float(item["relevance_score"]))
            for item in data.get("results", [])
        ]
