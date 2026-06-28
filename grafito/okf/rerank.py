"""Pluggable rerankers for OKF retrieval.

A :class:`Reranker` is any callable that re-scores concept candidates against the
query *text* — the precision step a bi-encoder (embedding) retrieval can't do on
its own. It is injected, never imported by the core: :meth:`OKFBundle.context`
takes a ``rerank=`` argument and uses whatever order the reranker returns.

This module ships:

* :class:`LexicalReranker` — dependency-free (query-term overlap). A sensible
  offline default/fallback; good enough to demonstrate the hook.
* :class:`CrossEncoderReranker` — a local HuggingFace cross-encoder via
  sentence-transformers (e.g. ``BAAI/bge-reranker-base``). Offline, no API key.
* :class:`CohereReranker`, :class:`VoyageReranker`, :class:`JinaReranker` —
  thin adapters over the providers' cross-encoder rerank APIs (the
  production-grade option). They differ only in endpoint, API key, the top-N
  parameter name, and the results key, so they share an HTTP base.

Any callable matching :class:`Reranker` works — these are conveniences, not a
requirement. Inject your own ``(query, candidates) -> [(concept, score)]``.
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


class CrossEncoderReranker:
    """Local HuggingFace cross-encoder reranker via sentence-transformers.

    Runs offline (no API key, no network once the model is cached) with reranker
    models such as ``BAAI/bge-reranker-base`` or
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``. Models are cached per name across
    instances, mirroring ``SentenceTransformerEmbeddingFunction``.
    """

    _models: dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "cpu",
        fields: tuple[str, ...] = DEFAULT_RERANK_FIELDS,
        max_chars: int = 2000,
        **kwargs: Any,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ValueError(
                "sentence_transformers is not installed. "
                "Install with `pip install sentence_transformers`."
            ) from exc

        self.model_name = model_name
        self._fields = fields
        self._max_chars = max_chars
        if model_name not in self._models:
            self._models[model_name] = CrossEncoder(model_name, device=device, **kwargs)
        self._model = self._models[model_name]

    def __call__(self, query: str, candidates: list["Concept"]) -> list[tuple["Concept", float]]:
        if not candidates:
            return []
        pairs = [
            (query, concept_text(c, fields=self._fields, max_chars=self._max_chars))
            for c in candidates
        ]
        scores = self._model.predict(pairs)
        scored = [(candidate, float(score)) for candidate, score in zip(candidates, scores)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


def _parse_rerank_results(
    data: dict, candidates: list["Concept"], *, results_key: str, provider: str
) -> list[tuple["Concept", float]]:
    """Map a provider's rerank response onto ``(concept, score)`` pairs.

    Providers return a list of ``{"index", "relevance_score"}`` under different
    keys (Cohere/Jina: ``results``; Voyage: ``data``). Anything else is an error.
    """
    items = data.get(results_key)
    if items is None:
        detail = data.get("message") or data.get("detail") or data.get("error") or data
        raise ValueError(f"{provider} rerank API error: {detail}")
    return [
        (candidates[item["index"]], float(item["relevance_score"]))
        for item in items
    ]


class _HTTPReranker:
    """Shared HTTP plumbing for cross-encoder rerank-API adapters.

    Subclasses set the four provider-specific bits: ``_provider``, ``_api_url``,
    ``_api_key_env``, ``_top_param`` and ``_results_key`` (plus a default model).
    """

    _provider: str = ""
    _api_url: str = ""
    _api_key_env: str = ""
    _top_param: str = "top_n"
    _results_key: str = "results"
    _default_model: str = ""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        api_key_env_var: str | None = None,
        base_url: str | None = None,
        top_n: int | None = None,
        fields: tuple[str, ...] = DEFAULT_RERANK_FIELDS,
        max_chars: int = 2000,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ValueError("httpx is not installed. Install with `pip install httpx`.") from exc

        env_var = api_key_env_var or self._api_key_env
        if api_key is None:
            api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"{self._provider} API key not provided. "
                f"Set {env_var} or pass api_key explicitly."
            )

        self.model = model or self._default_model
        self.top_n = top_n
        self._fields = fields
        self._max_chars = max_chars
        self._url = base_url or self._api_url
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
            payload[self._top_param] = self.top_n
        response = self._session.post(self._url, json=payload)
        return _parse_rerank_results(
            response.json(), candidates, results_key=self._results_key, provider=self._provider
        )


class CohereReranker(_HTTPReranker):
    """Adapter over the Cohere rerank API (cross-encoder relevance scoring)."""

    _provider = "Cohere"
    _api_url = "https://api.cohere.ai/v1/rerank"
    _api_key_env = "COHERE_API_KEY"
    _top_param = "top_n"
    _results_key = "results"
    _default_model = "rerank-english-v3.0"


class VoyageReranker(_HTTPReranker):
    """Adapter over the Voyage AI rerank API (``top_k``; results under ``data``)."""

    _provider = "Voyage"
    _api_url = "https://api.voyageai.com/v1/rerank"
    _api_key_env = "VOYAGE_API_KEY"
    _top_param = "top_k"
    _results_key = "data"
    _default_model = "rerank-2"


class JinaReranker(_HTTPReranker):
    """Adapter over the Jina AI rerank API (multilingual cross-encoder)."""

    _provider = "Jina"
    _api_url = "https://api.jina.ai/v1/rerank"
    _api_key_env = "JINA_API_KEY"
    _top_param = "top_n"
    _results_key = "results"
    _default_model = "jina-reranker-v2-base-multilingual"
