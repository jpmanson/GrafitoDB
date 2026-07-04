"""Agentic GraphRAG over an OKF bundle, via any OpenAI-compatible endpoint.

``OKFBundle.context()`` is *one-shot* GraphRAG: retrieve, graph-expand, pack.
This example is the **agentic** variant: the model drives the exploration
itself through tool calls —

- ``browse``   the bundle layer by layer (OKF progressive disclosure),
- ``search``   it by meaning/keywords,
- ``open``     a concept (full body + outgoing edges, typed),
- ``follow``   links across the graph (optionally by relationship type),
- ``history``  a concept's changelog,
- ``remember`` a new note back into the bundle (linked, embedded, autologged)

— then answers with concept citations. The bundle is the agent's *memory*:
mutations are recorded in the changelog (``autolog=True``) and ``save()``
round-trips everything to git-diffable markdown.

Works against any OpenAI-compatible chat-completions endpoint — OpenAI,
Ollama, vLLM, LM Studio, OpenRouter, llama.cpp server, ...:

    export OPENAI_BASE_URL=http://localhost:11434/v1   # e.g. Ollama
    export OPENAI_MODEL=llama3.1
    export OPENAI_API_KEY=sk-...                       # if the endpoint needs one
    python examples/okf/okf_agent.py

Requires ``httpx`` (``pip install httpx``). Retrieval runs on the
dependency-free hashing embedder, so the knowledge side works fully offline.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from grafito.okf import OKFBundle

sys.path.insert(0, str(Path(__file__).parent))
from okf_knowledge_base import HashingEmbeddingFunction  # noqa: E402  (demo embedder)

BUNDLE = Path(__file__).parent / "okf_knowledge_base"

SYSTEM_PROMPT = """\
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


# --- tools over the bundle ----------------------------------------------------


class BundleTools:
    """The OKFBundle façade exposed as OpenAI-style function tools."""

    def __init__(self, kb: OKFBundle) -> None:
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
        except Exception as exc:  # surface tool misuse to the model, not the user
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

    def _follow(self, concept_id: str, direction: str = "out", type: str | None = None) -> list[dict]:
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


# --- the agent loop -------------------------------------------------------------


class OpenAIChat:
    """Minimal client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        import httpx

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


def run_agent(
    kb: OKFBundle,
    question: str,
    *,
    chat,
    max_turns: int = 12,
    verbose: bool = True,
) -> str:
    """Drive a tool-calling loop over the bundle until the model answers.

    ``chat`` is any callable ``(messages, tool_schemas) -> assistant message``
    in OpenAI chat format — inject :class:`OpenAIChat` (or a fake, in tests).
    """
    tools = BundleTools(kb)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(layers=kb.layers())},
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


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_BASE_URL"):
        print(
            "Configure an OpenAI-compatible endpoint first, e.g.\n"
            "  export OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama\n"
            "  export OPENAI_MODEL=llama3.1\n"
            "or\n"
            "  export OPENAI_API_KEY=sk-...                        # OpenAI\n"
            "  export OPENAI_MODEL=gpt-4o-mini"
        )
        sys.exit(0)

    # The bundle is the agent's memory: embedded for search, autologged writes.
    kb = OKFBundle.load(str(BUNDLE), embed=HashingEmbeddingFunction(), autolog=True)

    question = (
        "A production Cypher query got slow after a data load. What should I do, "
        "step by step? Afterwards, remember a short checklist note under notes/ "
        "linked to the concepts you used."
    )
    print(f"Q: {question}\n")
    answer = run_agent(kb, question, chat=OpenAIChat())
    print(f"\nA: {answer}\n")

    # Persist knowledge + changelog to a scratch copy (markdown, git-ready).
    out = Path(tempfile.mkdtemp(prefix="okf_agent_")) / "bundle"
    kb.save(out)
    print(f"Bundle saved to {out}")
    log = out / "log.md"
    if log.exists():
        print("--- log.md ---")
        print(log.read_text(encoding="utf-8"))
    kb.db.close()


if __name__ == "__main__":
    main()
