"""Agentic GraphRAG toolkit over an :class:`~grafito.okf.bundle.OKFBundle`.

Where :meth:`OKFBundle.context` is *one-shot* GraphRAG (retrieve, graph-expand,
pack), this module lets a model **drive the exploration itself** through
OpenAI-style tool calls:

* :class:`BundleTools` — the bundle façade exposed as function tools
  (``browse``/``search``/``open``/``follow``/``history``/``remember``), with
  schemas and dispatch. Framework-free: ``tools.schemas`` plus
  ``tools.call(name, args)`` is the pair PydanticAI, CrewAI, LangGraph and MCP
  all ask for, so adapting is a schema translation and a closure rather than a
  rewrite — see ``examples/okf/okf_frameworks.py``. Whatever the framework,
  route through :meth:`BundleTools.call`: rebuilding the tools against
  ``kb.search()``/``kb[concept_id]`` drops the ``where``/``tag`` boundary.
* :class:`ThreadConfinedTools` — the same toolset, reachable from any thread.
  A bundle belongs to the thread that opened it, and frameworks like CrewAI run
  tools elsewhere; this wrapper gives one thread ownership and queues calls onto
  it. The one part of an integration that is not a schema translation.
* :func:`run_agent` — a minimal tool-calling loop: call the model, execute its
  tool calls against the bundle, feed results back, until it answers. It
  returns an :class:`AgentRun`: the answer plus what it cost to get there
  (turns, every tool call, tokens), so agentic retrieval can be measured
  against the one-shot :meth:`OKFBundle.context` instead of guessed at.
* :class:`Chat` — the model contract: **any callable**
  ``(messages, tools) -> assistant message`` in OpenAI chat format. Bring your
  own provider; two conveniences are bundled: :class:`OpenAIChat` for any
  OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, ...)
  and :class:`AnthropicChat` for Claude via the official ``anthropic`` SDK.

Grafito stays model-agnostic on purpose — the LLM client is injected, never
imported by the core (the same pattern as :mod:`grafito.okf.rerank`); the
bundled clients import their dependency lazily. Adapters for other providers
are one-liners; e.g. via litellm::

    import litellm

    def chat(messages, tools):
        response = litellm.completion(
            model="anthropic/claude-sonnet-5", messages=messages, tools=tools
        )
        return response.choices[0].message.model_dump()

    run_agent(kb, question, chat=chat)

See ``examples/okf/okf_agent.py`` for a runnable end-to-end walkthrough.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..filters import PropertyFilterGroup
    from .bundle import OKFBundle

DEFAULT_SYSTEM_PROMPT = """\
You are a knowledge-base assistant working over an OKF bundle (a graph of
markdown concepts). Ground every answer in the bundle:

1. Orient yourself with `browse` and/or `search` before answering.
2. `open` the most promising concepts and `follow` their links when related
   context could change the answer (links are typed relationships).
3. Answer concisely and cite the concept ids you used, e.g. (runbooks/slow-queries).
4. If the user asks you to record what you learned, use `remember` — write a
   short, self-contained note and link it to the concepts it builds on.

Bundle layers: {layers}
"""


@runtime_checkable
class Chat(Protocol):
    """The injected model: OpenAI-format messages + tool schemas in, message out.

    Any callable ``(messages, tools) -> assistant message dict`` works — an
    :class:`OpenAIChat` instance, a litellm wrapper, or a scripted fake in
    tests. The returned message may carry ``tool_calls``; a message without
    them ends the loop and its ``content`` is the final answer.
    """

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        ...


@runtime_checkable
class ToolSet(Protocol):
    """Anything with OpenAI-style tool ``schemas`` plus a matching ``call``.

    :class:`BundleTools` is one such toolset; pass additional instances via
    ``run_agent(..., extra_tools=[...])`` to give the agent app-specific
    tools (send a message, query an internal API, ...) alongside the
    bundle's own browse/search/open/follow/history/remember. No base class
    needed — any object with this shape works, same spirit as :class:`Chat`.
    """

    schemas: list[dict]

    def call(self, name: str, args: dict) -> str:
        ...


class BundleTools:
    """An :class:`OKFBundle` exposed as OpenAI-style function tools.

    ``schemas`` is the tool list to hand to the model; :meth:`call` dispatches
    one tool call and returns its JSON result. Tool errors (e.g. a wrong
    concept id) are returned as ``{"error": ...}`` for the model to react to,
    instead of raising.

    ``where``/``tag`` scope the whole toolset to a subset of the bundle, with the
    same semantics as :meth:`OKFBundle.search`. The filter is **fixed by the
    application, not chosen by the model**: it is not exposed in any tool schema,
    so the agent cannot widen or disable it. Every read path honours it —
    ``browse``, ``search``, ``open``, ``follow`` and ``history`` — because a
    filter that only covered ``search`` would be theatre: the model reads a
    concept id out of a link and opens it directly.

    ::

        tools = BundleTools(kb, where={"confidentiality": "public"})
        run_agent(kb, question, chat=chat, tools=tools)

    A concept the filter excludes is reported exactly like a nonexistent one
    (``Unknown concept: ...``), so the agent cannot probe for the existence of
    hidden concepts by opening ids it saw elsewhere.

    The filter governs *structure and retrieval*: which concepts can be listed,
    matched, opened, and traversed. It cannot redact **prose inside a concept
    you chose to show**. A visible concept whose body links to a hidden one
    (``[Secret](/docs/secret.md)``) still carries that markdown in its ``body``,
    and a ``log.md`` line may name a hidden concept the same way — so the model
    can learn a hidden id exists, even though every tool refuses to open it.
    Treat the filter as an access boundary on knowledge, not as redaction of
    text; if a body must not mention something, that belongs in the bundle.

    ``remember`` writes plain notes: they do not inherit the filter's fields, so
    with a filter active the agent may not be able to read back what it just
    wrote. Grafito does not silently stamp the filter's values onto new notes,
    since that would let an agent mark its own output ``status: approved``.

    Where ``where``/``tag`` scope *what the tools can reach*, ``include``/
    ``exclude`` scope *which tools exist* — the natural way to hand a read-only
    toolset to a framework that has its own write path::

        BundleTools(kb, exclude=["remember"])     # read-only
        BundleTools(kb, include=["search", "open"])  # retrieval only

    :attr:`schemas` reflects the choice, and :meth:`call` enforces it: a
    disabled name is rejected exactly like one that never existed, so a
    framework routing a stale name cannot reach it. :attr:`ALL_SCHEMAS` is
    always the full list, independent of any instance.

    ``raise_errors=True`` makes :meth:`call` propagate the underlying exception
    instead of returning ``{"error": ...}``. The default suits a raw
    tool-calling loop, where an error the model can read beats a crash;
    frameworks with their own tool-error handling (PydanticAI's ``ModelRetry``,
    CrewAI's retries) want the exception instead. Note that :func:`run_agent`
    has no such handling — a toolset with ``raise_errors=True`` will abort the
    loop on the first bad concept id.
    """

    def __init__(
        self,
        kb: "OKFBundle",
        *,
        tag: str | None = None,
        where: "dict | PropertyFilterGroup | None" = None,
        include: "list[str] | None" = None,
        exclude: "list[str] | None" = None,
        raise_errors: bool = False,
    ) -> None:
        self.kb = kb
        self.tag = tag
        self.where = where
        self.raise_errors = raise_errors
        if include is not None and exclude is not None:
            raise ValueError("Pass include or exclude, not both")
        available = {schema["function"]["name"] for schema in self.ALL_SCHEMAS}
        unknown = (set(include or []) | set(exclude or [])) - available
        if unknown:
            raise ValueError(
                f"Unknown tool name(s) {sorted(unknown)}; available: {sorted(available)}"
            )
        if include is not None:
            self.enabled = set(include)
        elif exclude is not None:
            self.enabled = available - set(exclude)
        else:
            self.enabled = available

    @property
    def schemas(self) -> list[dict]:
        """The tool schemas this instance exposes, after ``include``/``exclude``."""
        return [s for s in self.ALL_SCHEMAS if s["function"]["name"] in self.enabled]

    def _visible(self) -> set[str] | None:
        """Concept ids this toolset may expose, or ``None`` when unfiltered.

        Recomputed per call rather than cached: ``remember`` mutates the bundle
        mid-conversation, and a stale allow-list would be wrong in both
        directions.
        """
        return self.kb._filter_concept_ids(tag=self.tag, where=self.where)

    def _require_visible(self, concept_id: str) -> None:
        """Raise as if the concept did not exist when the filter excludes it."""
        visible = self._visible()
        if visible is not None and concept_id not in visible:
            raise ValueError(f"Unknown concept: {concept_id}")

    ALL_SCHEMAS = [
        {
            "type": "function",
            "function": {
                "name": "browse",
                "description": "List a directory of the bundle: subdirectories and "
                "concepts with titles/descriptions (no bodies). Omit layer for the root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "string", "description": "Directory path, e.g. 'decisions'"}
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search concepts by meaning and keywords (hybrid). "
                "Returns ranked ids with titles and descriptions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open",
                "description": "Read one concept in full: frontmatter, body, outgoing "
                "typed links, and citations.",
                "parameters": {
                    "type": "object",
                    "properties": {"concept_id": {"type": "string"}},
                    "required": ["concept_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "follow",
                "description": "Traverse the graph from a concept: outgoing ('out') or "
                "incoming ('in') links, optionally restricted to one relationship type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept_id": {"type": "string"},
                        "direction": {"type": "string", "enum": ["out", "in"], "default": "out"},
                        "type": {"type": "string", "description": "e.g. 'JOINS_WITH'"},
                    },
                    "required": ["concept_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "history",
                "description": "Changelog entries (newest first), optionally only those "
                "mentioning one concept.",
                "parameters": {
                    "type": "object",
                    "properties": {"concept_id": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Save a new note into the bundle (embedded, searchable, "
                "autologged). Link it to the concepts it builds on.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept_id": {"type": "string", "description": "e.g. 'notes/slow-query-checklist'"},
                        "title": {"type": "string"},
                        "body": {"type": "string", "description": "Markdown body of the note"},
                        "description": {"type": "string"},
                        "links": {
                            "type": "array",
                            "description": "Concepts this note builds on",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target": {"type": "string"},
                                    "type": {"type": "string", "default": "LINKS_TO"},
                                },
                                "required": ["target"],
                            },
                        },
                    },
                    "required": ["concept_id", "title", "body"],
                },
            },
        },
    ]

    def call(self, name: str, args: dict) -> str:
        """Execute one tool call and return its JSON result (or an error the
        model can react to — a wrong concept id should not kill the loop).

        A name this instance does not expose — never defined, or dropped by
        ``include``/``exclude`` — is rejected here, not just omitted from
        :attr:`schemas`. An agent framework that routes a stale or invented
        name must not reach a disabled tool.

        With ``raise_errors=True`` the underlying exception propagates instead
        of being wrapped, for frameworks that drive their own retry on it.
        """
        try:
            if name not in self.enabled:
                raise ValueError(f"Unknown tool {name!r}")
            result = getattr(self, f"_{name}")(**args)
        except Exception as exc:  # surface tool misuse to the model, not the caller
            if self.raise_errors:
                raise
            result = {"error": str(exc)}
        return json.dumps(result, ensure_ascii=False)

    def layers(self) -> dict:
        """Top-level layers with concept counts, honouring the toolset's filter.

        Used to orient the model in the system prompt: an unfiltered
        ``kb.layers()`` would leak the size of what the filter hides.
        """
        if self.where is None and self.tag is None:
            return self.kb.layers()
        listing = self.kb.index(tag=self.tag, where=self.where)
        layers = dict(listing["subdirs"])
        if listing["concepts"]:
            # Root-level concepts are their own layer in kb.layers(); keep the
            # filtered view shaped the same so the prompt stays comparable.
            layers["."] = len(listing["concepts"])
        return layers

    def _browse(self, layer: str | None = None) -> dict:
        return self.kb.index(layer, tag=self.tag, where=self.where)

    def _search(self, query: str, k: int = 5) -> list[dict]:
        hits = self.kb.search(query, k=k, mode="hybrid", tag=self.tag, where=self.where)
        return [
            {
                "id": h.concept.id,
                "title": h.concept.title,
                "description": h.concept.description,
                "score": round(h.score, 4),
            }
            for h in hits
        ]

    def _open(self, concept_id: str) -> dict:
        self._require_visible(concept_id)
        concept = self.kb[concept_id]
        edges = self.kb.execute(
            "MATCH (a)-[r]->(b) WHERE a.concept_id = $cid AND b.concept_id IS NOT NULL "
            "RETURN type(r) AS type, b.concept_id AS target",
            cid=concept_id,
        )
        # Links to hidden concepts are dropped too — an edge list is a directory
        # of ids, and leaking one invites the model to try opening it.
        visible = self._visible()
        if visible is not None:
            edges = [e for e in edges if e["target"] in visible]
        return {
            "id": concept.id,
            "type": concept.type,
            "title": concept.title,
            "description": concept.description,
            "tags": concept.tags,
            "body": concept.body,
            "links": [e for e in edges if e["type"] != "CITES"],
            "cites": concept.cites(),
        }

    def _follow(
        self, concept_id: str, direction: str = "out", type: str | None = None
    ) -> list[dict]:
        self._require_visible(concept_id)
        concept = self.kb[concept_id]
        neighbors = concept.links(type=type) if direction == "out" else concept.linked_by(type=type)
        visible = self._visible()
        if visible is not None:
            neighbors = [c for c in neighbors if c.id in visible]
        return [{"id": c.id, "title": c.title, "description": c.description} for c in neighbors]

    def _history(self, concept_id: str | None = None) -> list[dict]:
        if concept_id is not None:
            self._require_visible(concept_id)
        return self.kb.log(concept_id)

    def _remember(
        self,
        concept_id: str,
        title: str,
        body: str,
        description: str | None = None,
        links: list[dict] | None = None,
    ) -> dict:
        self.kb.add_concept(
            concept_id, type="Note", title=title, body=body, description=description
        )
        for link in links or []:
            self.kb.link(concept_id, link["target"], type=link.get("type") or "LINKS_TO")
        return {"saved": concept_id, "linked_to": [link["target"] for link in links or []]}


class ThreadConfinedTools:
    """A :class:`ToolSet` that serializes every call onto one owning thread.

    An :class:`OKFBundle` belongs to the thread that opened it: SQLite
    connections are created with ``check_same_thread=True``, so reaching the
    bundle from anywhere else raises ``sqlite3.ProgrammingError``. Agent
    frameworks routinely run tools off the main thread — CrewAI does, and
    PydanticAI does for *sync* tools — which makes a bare :class:`BundleTools`
    fail on every call there. This wrapper gives one dedicated thread ownership
    of the bundle and hands it each call through a queue.

    ``factory`` runs **on that thread** and must open the bundle *and* return
    the toolset wrapping it. Opening the bundle in the caller and closing over
    it defeats the entire purpose::

        with ThreadConfinedTools(
            lambda: BundleTools(OKFBundle.load(path, autolog=True))
        ) as tools:
            ...                                   # safe from any thread
            tools.run(lambda t: t.kb.save())      # so is this

    :meth:`run` is the escape hatch for everything that is not a tool call.
    Anything touching the bundle has to happen on the owning thread, so
    ``kb.save()`` after an agent's ``remember`` must go through it — calling it
    directly from the main thread raises the same ``ProgrammingError`` the
    wrapper exists to avoid.

    Calls are serialized: this is a correctness device, not a scaling one, and
    concurrent agents queue behind each other. That is the honest tradeoff while
    the bundle is single-threaded — the alternative, opening the connection with
    ``check_same_thread=False``, would share mutable state (transaction flag,
    in-memory vector and text indexes) with no locking at all.

    Errors are re-raised in the calling thread, so a wrapped toolset built with
    ``raise_errors=True`` still behaves as the framework expects.
    """

    def __init__(self, factory: "Callable[[], ToolSet]", *, name: str = "okf-bundle") -> None:
        self._requests: "queue.Queue" = queue.Queue()
        handshake: "queue.Queue" = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._serve, args=(factory, handshake), name=name, daemon=True
        )
        self._thread.start()
        ok, value = handshake.get()
        if not ok:
            raise value
        self.schemas: list[dict] = value

    def _serve(self, factory: "Callable[[], ToolSet]", handshake: "queue.Queue") -> None:
        """Own the bundle for the life of the thread; serve one request at a time."""
        try:
            toolset = factory()
        except Exception as exc:  # the constructor is blocked on the handshake
            handshake.put((False, exc))
            return
        handshake.put((True, list(toolset.schemas)))
        while True:
            request = self._requests.get()
            if request is None:
                return
            work, reply = request
            try:
                reply.put((True, work(toolset)))
            except Exception as exc:  # re-raised in the calling thread
                reply.put((False, exc))

    def run(self, work: "Callable[[Any], Any]") -> Any:
        """Run ``work(toolset)`` on the owning thread and return its result.

        The way to reach the bundle for anything that is not a tool call —
        ``tools.run(lambda t: t.kb.save())`` to persist an agent's notes, or
        ``tools.run(lambda t: len(t.kb))`` to inspect it. Exceptions propagate
        to the caller.
        """
        if self._closed:
            raise RuntimeError("ThreadConfinedTools is closed")
        reply: "queue.Queue" = queue.Queue()
        self._requests.put((work, reply))
        ok, value = reply.get()
        if not ok:
            raise value
        return value

    def call(self, name: str, args: dict) -> str:
        """Dispatch one tool call on the owning thread (the :class:`ToolSet` API)."""
        return self.run(lambda toolset: toolset.call(name, args))

    def layers(self) -> dict:
        """Delegate :meth:`BundleTools.layers`, so this is accepted as
        ``run_agent(..., tools=...)`` wherever a :class:`BundleTools` is —
        the loop builds its system prompt from the toolset's layer counts."""
        return self.run(lambda toolset: toolset.layers())

    def close(self) -> None:
        """Stop the owning thread. Idempotent.

        The bundle is *not* closed for you: it may still be needed (to save, or
        by another wrapper), and only this thread could have done it anyway.
        Close it first if you own it — ``tools.run(lambda t: t.kb.db.close())``.
        """
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join(timeout=5)

    def __enter__(self) -> "ThreadConfinedTools":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _without_private(messages: list[dict]) -> list[dict]:
    """Drop the loop's ``_``-prefixed bookkeeping keys before sending upstream.

    Assistant messages carry private annotations (``_usage``,
    ``_anthropic_content``); an OpenAI-compatible endpoint would reject them as
    unknown fields or, worse, bill for the extra bytes.
    """
    return [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]


def _openai_usage(data: dict) -> dict | None:
    """OpenAI ``usage`` -> the loop's normalized shape (see :class:`AgentRun`).

    ``prompt_tokens`` already includes whatever was served from cache, so it
    maps to ``input_tokens`` directly. OpenAI caches implicitly and never
    reports a write, hence ``cache_write_tokens`` of 0.
    """
    usage = data.get("usage")
    if not usage:
        return None
    details = usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "cached_input_tokens": details.get("cached_tokens", 0),
        "cache_write_tokens": 0,
        "output_tokens": usage.get("completion_tokens", 0),
    }


class OpenAIChat:
    """Minimal :class:`Chat` for any OpenAI-compatible ``/chat/completions``
    endpoint — OpenAI, Ollama, vLLM, LM Studio, OpenRouter, llama.cpp server.

    Reads ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``OPENAI_MODEL`` when the
    matching argument is omitted. Requires ``httpx``. For providers that do not
    speak the OpenAI format, inject your own callable instead (see the litellm
    adapter in the module docstring).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ValueError("httpx is not installed. Install with `pip install httpx`.") from exc

        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": _without_private(messages),
                "tools": tools,
                "tool_choice": "auto",
            },
        )
        data = response.json()
        if "choices" not in data:
            raise RuntimeError(f"Endpoint error: {data.get('error', data)}")
        message = data["choices"][0]["message"]
        usage = _openai_usage(data)
        if usage is not None:
            message["_usage"] = usage
        return message

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "OpenAIChat":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --- Anthropic adapter: OpenAI chat format <-> Anthropic Messages API ----------


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    """OpenAI function-tool schemas -> Anthropic tool definitions."""
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        for tool in tools
    ]


def _anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI-format history -> (system prompt, Anthropic messages).

    Assistant turns produced by :class:`AnthropicChat` carry the raw Anthropic
    content under ``_anthropic_content`` and are replayed verbatim — this
    preserves ``thinking`` blocks, which must be echoed back unchanged.
    Consecutive ``role="tool"`` results merge into a single user message
    (parallel tool results must not be split across messages).
    """
    system: str | None = None
    out: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            system = message.get("content")
        elif role == "assistant":
            if message.get("_anthropic_content"):
                out.append({"role": "assistant", "content": message["_anthropic_content"]})
                continue
            blocks: list[dict] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"] or "{}"),
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
                "content": message.get("content") or "",
            }
            last = out[-1] if out else None
            if (
                last is not None
                and last["role"] == "user"
                and isinstance(last["content"], list)
                and last["content"]
                and last["content"][-1].get("type") == "tool_result"
            ):
                last["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:  # user
            out.append({"role": "user", "content": message.get("content") or ""})
    return system, out


def _anthropic_usage(response: Any) -> dict | None:
    """Anthropic ``usage`` -> the loop's normalized shape (see :class:`AgentRun`).

    The two providers count the prompt differently and the difference is easy
    to get wrong: Anthropic's ``input_tokens`` is the **uncached remainder**,
    with cache reads and writes reported alongside it, while OpenAI's
    ``prompt_tokens`` is the whole prompt. Summing the three buckets here makes
    ``input_tokens`` mean the same thing on both paths, with the cheap
    (``cached_input_tokens``, ~0.1x) and expensive (``cache_write_tokens``,
    ~1.25x) slices kept separate so cost math stays possible.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    uncached = getattr(usage, "input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return {
        "input_tokens": uncached + cache_read + cache_write,
        "cached_input_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }


def _openai_message(response: Any) -> dict:
    """Anthropic Messages API response -> OpenAI-format assistant message.

    The raw content blocks ride along under ``_anthropic_content`` so the next
    request can echo them back exactly (thinking blocks included), and the
    turn's token counts under ``_usage`` for :func:`run_agent` to aggregate.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    raw_blocks: list[dict] = []
    for block in response.content:
        raw_blocks.append(block.model_dump() if hasattr(block, "model_dump") else dict(block))
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
    message: dict = {
        "role": "assistant",
        "content": "\n".join(text_parts) or None,
        "_anthropic_content": raw_blocks,
    }
    usage = _anthropic_usage(response)
    if usage is not None:
        message["_usage"] = usage
    if tool_calls:
        message["tool_calls"] = tool_calls
    if getattr(response, "stop_reason", None) == "refusal" and not tool_calls:
        message["content"] = message["content"] or "(request declined by safety classifiers)"
    return message


class AnthropicChat:
    """:class:`Chat` for Claude, via the official ``anthropic`` SDK.

    Translates between the loop's OpenAI chat format and the Anthropic
    Messages API — system prompt as the ``system`` parameter, tool schemas as
    ``input_schema`` definitions, ``tool_calls``/``role="tool"`` as
    ``tool_use``/``tool_result`` blocks — so ``run_agent`` works unchanged.
    Runs with adaptive thinking; thinking blocks are preserved across turns.

    Requires ``pip install anthropic`` (or ``grafito[anthropic]``). With no
    ``api_key``, the SDK resolves credentials from the environment
    (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile); ``ANTHROPIC_MODEL``
    overrides the default model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        max_tokens: int = 16000,
        timeout: float = 300.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ValueError(
                "anthropic is not installed. "
                "Install with `pip install anthropic` (or `grafito[anthropic]`)."
            ) from exc

        self.model = model or os.environ.get("ANTHROPIC_MODEL") or "claude-opus-4-8"
        self.max_tokens = max_tokens
        client_kwargs: dict[str, Any] = {"timeout": timeout}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**client_kwargs)

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        system, converted = _anthropic_messages(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "adaptive"},
            "messages": converted,
            "tools": _anthropic_tools(tools),
        }
        if system:
            request["system"] = system
        response = self._client.messages.create(**request)
        return _openai_message(response)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "AnthropicChat":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _describe_result(name: str, result: str) -> str:
    """One-line, human-readable summary of a tool result for ``verbose`` logs.

    Raw JSON gets unreadable once truncated mid-string; this describes shape
    (counts, ids, titles) instead of slicing bytes.
    """
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return result[:160]
    if isinstance(data, dict) and set(data) == {"error"}:
        return f"error: {data['error']}"
    if name in ("search", "follow"):
        if not data:
            return "(no results)"
        shown = ", ".join(f"{item['id']} ({item['title']})" for item in data[:3])
        more = f", +{len(data) - 3} more" if len(data) > 3 else ""
        return f"{len(data)} result(s): {shown}{more}"
    if name == "browse":
        return (
            f"layer={data['layer'] or '/'!r}: {len(data['subdirs'])} subdir(s), "
            f"{len(data['concepts'])} concept(s)"
        )
    if name == "open":
        return f"{data['id']} - {data['title']} ({len(data['links'])} link(s), {len(data['cites'])} citation(s))"
    if name == "history":
        return f"{len(data)} entr{'y' if len(data) == 1 else 'ies'}"
    if name == "remember":
        return f"saved {data['saved']}, linked to {data['linked_to']}"
    return json.dumps(data, ensure_ascii=False)[:160]


def _tool_error(result: str) -> str | None:
    """The error message a tool result carries, or ``None`` when it succeeded."""
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and set(data) == {"error"}:
        return data["error"]
    return None


_USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")


def _add_usage(total: dict, usage: dict | None) -> None:
    """Accumulate one turn's normalized usage into the run total, in place."""
    if not usage:
        return
    for name in _USAGE_FIELDS:
        value = usage.get(name)
        if value:
            total[name] = total.get(name, 0) + value
    total["requests"] = total.get("requests", 0) + 1


@dataclass
class ToolCall:
    """One tool invocation the model made, as recorded by :func:`run_agent`.

    ``result_bytes`` is the size of the JSON handed back to the model. It is
    the honest proxy for what agentic exploration costs: tool results stay in
    ``messages`` for the rest of the conversation, so a single ``open`` of a
    long concept is re-sent on every later turn. Unlike token counts it needs
    no cooperation from the model provider — Grafito produces this number
    itself, so it is available even with a :class:`Chat` that reports no usage.

    ``error`` is the message when the tool failed (a bad concept id, a filtered
    one, an unknown tool), ``None`` otherwise.
    """

    turn: int
    name: str
    args: dict
    result_bytes: int
    error: str | None = None


@dataclass
class AgentRun:
    """The result of :func:`run_agent`: the answer, plus what it cost to get it.

    ``str(run)`` returns :attr:`answer`, so it drops straight into an f-string.

    ``tool_calls`` lists every invocation in order and ``usage`` aggregates the
    token counts the :class:`Chat` reported, normalized across providers:

    * ``input_tokens`` — the whole prompt sent that turn, cached parts included
    * ``cached_input_tokens`` — the slice served from cache (~0.1x the price)
    * ``cache_write_tokens`` — the slice written to cache (~1.25x; Anthropic
      only, since OpenAI-compatible endpoints cache implicitly)
    * ``output_tokens``, ``requests``

    Summing ``input_tokens`` across turns is the real billed cost, not the size
    of the context: a tool-calling loop re-sends the full history every turn.
    That is exactly why the cached slice is broken out — read it before
    concluding that a long conversation is expensive.

    ``usage`` is empty when the injected :class:`Chat` reports none; the loop
    never fabricates numbers. ``stopped_early`` is True when ``max_turns`` ran
    out before the model answered, in which case ``answer`` is empty.

    :meth:`summary` folds all of it into the efficiency numbers worth watching.
    """

    answer: str
    messages: list[dict]
    turns: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    stopped_early: bool = False

    def __str__(self) -> str:
        return self.answer

    def summary(self) -> dict:
        """Efficiency numbers for this run, as a plain dict.

        ``repeated_calls`` counts invocations whose ``(name, args)`` the model
        had already issued in this run — the classic failure mode of an agent
        re-reading what is already sitting in its context, and the most
        actionable signal here. ``by_tool`` breaks calls, errors and bytes down
        per tool, which is where ``open`` usually shows up as the cost driver.
        """
        by_tool: dict[str, dict] = {}
        seen: set[tuple[str, str]] = set()
        repeated = 0
        for call in self.tool_calls:
            stats = by_tool.setdefault(call.name, {"calls": 0, "errors": 0, "bytes": 0})
            stats["calls"] += 1
            stats["bytes"] += call.result_bytes
            if call.error is not None:
                stats["errors"] += 1
            key = (call.name, json.dumps(call.args, sort_keys=True, ensure_ascii=False))
            if key in seen:
                repeated += 1
            seen.add(key)
        return {
            "turns": self.turns,
            "tool_calls": len(self.tool_calls),
            "errors": sum(1 for call in self.tool_calls if call.error is not None),
            "repeated_calls": repeated,
            "result_bytes": sum(call.result_bytes for call in self.tool_calls),
            "by_tool": by_tool,
            "usage": dict(self.usage),
        }


def _describe_run(run: AgentRun) -> str:
    """One-line efficiency summary of a finished run, for ``verbose`` logs."""
    stats = run.summary()
    parts = [f"{stats['turns']} turn(s)", f"{stats['tool_calls']} tool call(s)"]
    if stats["errors"]:
        parts.append(f"{stats['errors']} error(s)")
    if stats["repeated_calls"]:
        parts.append(f"{stats['repeated_calls']} repeated")
    parts.append(f"{stats['result_bytes']} tool byte(s)")
    usage = stats["usage"]
    if usage:
        tokens = f"{usage.get('input_tokens', 0)} in"
        if usage.get("cached_input_tokens"):
            tokens += f" ({usage['cached_input_tokens']} cached)"
        parts.append(f"{tokens} / {usage.get('output_tokens', 0)} out")
    return ", ".join(parts)


def run_agent(
    kb: "OKFBundle",
    question: str,
    *,
    chat: Chat,
    tools: "BundleTools | None" = None,
    extra_tools: "list[ToolSet] | None" = None,
    messages: list[dict] | None = None,
    system: str | None = None,
    max_turns: int = 12,
    verbose: bool = False,
) -> AgentRun:
    """Drive a tool-calling loop over the bundle until the model answers.

    ``chat`` is any :class:`Chat` callable. ``system`` overrides the default
    system prompt (which embeds ``kb.layers()`` for orientation), used only
    when the conversation starts. The loop always executes tool calls
    through :class:`BundleTools` — including the ``remember`` write path, so
    with ``autolog=True`` the bundle records what the agent learned.

    Returns an :class:`AgentRun`: the model's final text plus what the run
    cost — turns taken, every tool call with its result size and error, and
    aggregated token usage. ``str(run)`` is the answer, so it still drops
    into an f-string; read ``run.summary()`` when comparing agentic
    exploration against a one-shot :meth:`OKFBundle.context` pack.

    ``tools`` injects a pre-configured :class:`BundleTools` instead of the
    default one — the way to scope an agent to a subset of the bundle, since
    the filter belongs to the application rather than the model::

        run_agent(kb, question, chat=chat,
                  tools=BundleTools(kb, where={"confidentiality": "public"}))

    ``include``/``exclude`` on that same toolset narrow which tools the model
    is offered at all (e.g. ``exclude=["remember"]`` for a read-only agent).
    Leave its ``raise_errors`` at the default: this loop has no tool-error
    handling of its own, so a raising toolset aborts the run on the first bad
    concept id instead of letting the model recover.

    ``extra_tools`` adds app-specific :class:`ToolSet`\\ s (own ``schemas``
    plus a matching ``call``) alongside the bundle's own, for tools that
    have nothing to do with the bundle (send a message, query an internal
    API, ...). Tool names must be unique across every toolset; a collision
    raises ``ValueError`` before the model is ever called. A tool call for
    a name no toolset owns comes back as ``{"error": ...}`` for the model to
    react to, same as any other tool error.

    Pass a list via ``messages`` to carry conversation memory across calls:
    it is extended in place with this turn's question, tool calls, and
    answer, so reusing the same list on the next call continues the same
    conversation. Omit it (the default) for a one-shot question — the list
    is then created here and reachable as ``run.messages``. Note that tool
    results (e.g. full concept bodies from ``open``) accumulate in
    ``messages`` turn over turn, so a long-running conversation costs more
    tokens each turn; ``run.summary()["result_bytes"]`` measures it.
    """
    bundle_tools = tools if tools is not None else BundleTools(kb)
    toolsets: list[ToolSet] = [bundle_tools, *(extra_tools or [])]
    schemas: list[dict] = []
    dispatch: dict[str, ToolSet] = {}
    for toolset in toolsets:
        for schema in toolset.schemas:
            name = schema["function"]["name"]
            if name in dispatch:
                raise ValueError(f"Duplicate tool name {name!r} across toolsets")
            dispatch[name] = toolset
            schemas.append(schema)

    if messages is None:
        messages = []
    if not messages:
        messages.append(
            {
                "role": "system",
                "content": system
                or DEFAULT_SYSTEM_PROMPT.format(layers=bundle_tools.layers()),
            }
        )
    messages.append({"role": "user", "content": question})
    recorded: list[ToolCall] = []
    usage: dict = {}
    for turn in range(1, max_turns + 1):
        message = chat(messages, schemas)
        messages.append(message)
        _add_usage(usage, message.get("_usage"))
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            run = AgentRun(
                answer=message.get("content") or "",
                messages=messages,
                turns=turn,
                tool_calls=recorded,
                usage=usage,
            )
            if verbose:
                print(f"  [{_describe_run(run)}]")
            return run
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            if verbose:
                print(f"  -> {name}({json.dumps(args, ensure_ascii=False)})")
            toolset = dispatch.get(name)
            if toolset is None:
                result = json.dumps({"error": f"Unknown tool {name!r}"})
            else:
                result = toolset.call(name, args)
            if verbose:
                print(f"     {_describe_result(name, result)}")
            recorded.append(
                ToolCall(
                    turn=turn,
                    name=name,
                    args=args,
                    result_bytes=len(result.encode("utf-8")),
                    error=_tool_error(result),
                )
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", name), "content": result}
            )
    run = AgentRun(
        answer="",
        messages=messages,
        turns=max_turns,
        tool_calls=recorded,
        usage=usage,
        stopped_early=True,
    )
    if verbose:
        print(f"  [stopped: max_turns reached — {_describe_run(run)}]")
    return run
