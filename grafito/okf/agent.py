"""Agentic GraphRAG toolkit over an :class:`~grafito.okf.bundle.OKFBundle`.

Where :meth:`OKFBundle.context` is *one-shot* GraphRAG (retrieve, graph-expand,
pack), this module lets a model **drive the exploration itself** through
OpenAI-style tool calls:

* :class:`BundleTools` — the bundle façade exposed as function tools
  (``browse``/``search``/``open``/``follow``/``history``/``remember``), with
  schemas and dispatch. Framework-free: the same tools drop into an OpenAI
  tool-calling loop, LangGraph, CrewAI, or an MCP server.
* :func:`run_agent` — a minimal tool-calling loop: call the model, execute its
  tool calls against the bundle, feed results back, until it answers.
* :class:`Chat` — the model contract: **any callable**
  ``(messages, tools) -> assistant message`` in OpenAI chat format. Bring your
  own provider; :class:`OpenAIChat` is the bundled convenience for any
  OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, ...).

Grafito stays model-agnostic on purpose — the LLM client is injected, never
imported by the core (the same pattern as :mod:`grafito.okf.rerank`). Adapters
for other providers are one-liners; e.g. via litellm::

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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
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


class BundleTools:
    """An :class:`OKFBundle` exposed as OpenAI-style function tools.

    ``schemas`` is the tool list to hand to the model; :meth:`call` dispatches
    one tool call and returns its JSON result. Tool errors (e.g. a wrong
    concept id) are returned as ``{"error": ...}`` for the model to react to,
    instead of raising.
    """

    def __init__(self, kb: "OKFBundle") -> None:
        self.kb = kb

    schemas = [
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
        model can react to — a wrong concept id should not kill the loop)."""
        try:
            result = getattr(self, f"_{name}")(**args)
        except Exception as exc:  # surface tool misuse to the model, not the caller
            result = {"error": str(exc)}
        return json.dumps(result, ensure_ascii=False)

    def _browse(self, layer: str | None = None) -> dict:
        return self.kb.index(layer)

    def _search(self, query: str, k: int = 5) -> list[dict]:
        hits = self.kb.search(query, k=k, mode="hybrid")
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
        concept = self.kb[concept_id]
        edges = self.kb.execute(
            "MATCH (a)-[r]->(b) WHERE a.concept_id = $cid AND b.concept_id IS NOT NULL "
            "RETURN type(r) AS type, b.concept_id AS target",
            cid=concept_id,
        )
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
        concept = self.kb[concept_id]
        neighbors = concept.links(type=type) if direction == "out" else concept.linked_by(type=type)
        return [{"id": c.id, "title": c.title, "description": c.description} for c in neighbors]

    def _history(self, concept_id: str | None = None) -> list[dict]:
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
            json={"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"},
        )
        data = response.json()
        if "choices" not in data:
            raise RuntimeError(f"Endpoint error: {data.get('error', data)}")
        return data["choices"][0]["message"]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "OpenAIChat":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def run_agent(
    kb: "OKFBundle",
    question: str,
    *,
    chat: Chat,
    system: str | None = None,
    max_turns: int = 12,
    verbose: bool = False,
) -> str:
    """Drive a tool-calling loop over the bundle until the model answers.

    ``chat`` is any :class:`Chat` callable. ``system`` overrides the default
    system prompt (which embeds ``kb.layers()`` for orientation). The loop
    executes every tool call through :class:`BundleTools` — including the
    ``remember`` write path, so with ``autolog=True`` the bundle records what
    the agent learned. Returns the model's final text (or a note when
    ``max_turns`` is exhausted).
    """
    tools = BundleTools(kb)
    messages: list[dict] = [
        {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT.format(layers=kb.layers())},
        {"role": "user", "content": question},
    ]
    for _ in range(max_turns):
        message = chat(messages, tools.schemas)
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or ""
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            if verbose:
                print(f"  -> {name}({json.dumps(args, ensure_ascii=False)})")
            result = tools.call(name, args)
            if verbose:
                print(f"     {result[:160]}{'…' if len(result) > 160 else ''}")
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", name), "content": result}
            )
    return "(stopped: max_turns reached without a final answer)"
