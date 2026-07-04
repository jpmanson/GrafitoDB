"""Agentic GraphRAG over an OKF bundle, via any OpenAI-compatible endpoint.

``OKFBundle.context()`` is *one-shot* GraphRAG: retrieve, graph-expand, pack.
This example runs the **agentic** variant from ``grafito.okf.agent``: the model
drives the exploration itself through tool calls (browse / search / open /
follow / history) and writes what it learned back into the bundle
(``remember`` + ``autolog=True``), then everything — knowledge *and* its
changelog — round-trips to git-diffable markdown with ``save()``.

Works against any OpenAI-compatible chat-completions endpoint — OpenAI,
Ollama, vLLM, LM Studio, OpenRouter, llama.cpp server, ...:

    export OPENAI_BASE_URL=http://localhost:11434/v1   # e.g. Ollama
    export OPENAI_MODEL=llama3.1
    export OPENAI_API_KEY=sk-...                       # if the endpoint needs one
    python examples/okf/okf_agent.py

— or against Claude via the official SDK (``pip install anthropic``), picked
automatically when only Anthropic credentials are configured:

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/okf/okf_agent.py                   # claude-opus-4-8 by default

Requires ``httpx`` for the OpenAI-compatible path. Retrieval runs on the
dependency-free hashing embedder, so the knowledge side works fully offline.

The model is injected: ``run_agent(kb, question, chat=...)`` takes any
``(messages, tools) -> message`` callable. For a non-OpenAI-format provider,
swap ``OpenAIChat`` for a 4-line adapter, e.g. via litellm::

    import litellm

    def chat(messages, tools):
        response = litellm.completion(
            model="anthropic/claude-sonnet-5", messages=messages, tools=tools
        )
        return response.choices[0].message.model_dump()
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from grafito.okf import AnthropicChat, OKFBundle, OpenAIChat, run_agent

sys.path.insert(0, str(Path(__file__).parent))
from okf_knowledge_base import HashingEmbeddingFunction  # noqa: E402  (demo embedder)

BUNDLE = Path(__file__).parent / "okf_knowledge_base"


def pick_chat():
    """OpenAI-compatible endpoint when configured; else Claude; else explain."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return OpenAIChat()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicChat()
    print(
        "Configure a model endpoint first, e.g.\n"
        "  export OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama\n"
        "  export OPENAI_MODEL=llama3.1\n"
        "or\n"
        "  export OPENAI_API_KEY=sk-...                        # OpenAI\n"
        "or\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...                 # Claude"
    )
    sys.exit(0)


def main() -> None:
    chat = pick_chat()

    # The bundle is the agent's memory: embedded for search, autologged writes.
    kb = OKFBundle.load(str(BUNDLE), embed=HashingEmbeddingFunction(), autolog=True)

    question = (
        "A production Cypher query got slow after a data load. What should I do, "
        "step by step? Afterwards, remember a short checklist note under notes/ "
        "linked to the concepts you used."
    )
    print(f"Q: {question}\n")
    answer = run_agent(kb, question, chat=chat, verbose=True)
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
