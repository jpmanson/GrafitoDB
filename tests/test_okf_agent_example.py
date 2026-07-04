"""Tests for the agentic GraphRAG example (examples/okf/okf_agent.py).

The agent loop is driven with a scripted fake ``chat`` — no endpoint, no
network — so the example's tool schemas, dispatch, and write path stay honest.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

_EXAMPLES = Path("examples") / "okf"


def _load_example():
    spec = importlib.util.spec_from_file_location("okf_agent", _EXAMPLES / "okf_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["okf_agent"] = module
    spec.loader.exec_module(module)
    return module


def _tool_call(name: str, call_id: str, **args) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class ScriptedChat:
    """Replays a fixed sequence of assistant messages; records what it saw."""

    def __init__(self, turns: list[dict]) -> None:
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        assert any(t["function"]["name"] == "remember" for t in tools)
        self.seen.append(list(messages))
        return self.turns.pop(0)


@pytest.fixture
def kb():
    okf_agent = _load_example()
    from okf_knowledge_base import HashingEmbeddingFunction

    from grafito.okf import OKFBundle

    bundle = OKFBundle.load(
        str(_EXAMPLES / "okf_knowledge_base"),
        embed=HashingEmbeddingFunction(),
        autolog=True,
    )
    yield okf_agent, bundle
    bundle.db.close()


def test_agent_loop_explores_and_remembers(kb):
    okf_agent, bundle = kb
    chat = ScriptedChat(
        [
            {"role": "assistant", "content": None, "tool_calls": [
                _tool_call("search", "c1", query="slow query performance"),
                _tool_call("browse", "c2", layer="runbooks"),
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                _tool_call("open", "c3", concept_id="runbooks/slow-queries"),
                _tool_call("open", "c4", concept_id="does/not/exist"),  # must not crash
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                _tool_call(
                    "remember", "c5",
                    concept_id="notes/slow-query-checklist",
                    title="Slow query checklist",
                    body="1. Check indexes. 2. Profile the query.",
                    links=[{"target": "runbooks/slow-queries", "type": "BUILDS_ON"}],
                ),
            ]},
            {"role": "assistant", "content": "Use the runbook (runbooks/slow-queries)."},
        ]
    )
    answer = okf_agent.run_agent(bundle, "query got slow, what do I do?", chat=chat, verbose=False)
    assert answer == "Use the runbook (runbooks/slow-queries)."

    # Tool results reached the model as role=tool messages.
    last_seen = chat.seen[-1]
    tool_messages = [m for m in last_seen if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"c1", "c2", "c3", "c4", "c5"}
    opened = json.loads(next(m for m in tool_messages if m["tool_call_id"] == "c3")["content"])
    assert opened["id"] == "runbooks/slow-queries"
    assert opened["body"]
    failed = json.loads(next(m for m in tool_messages if m["tool_call_id"] == "c4")["content"])
    assert "error" in failed

    # The write path: note created, typed-linked, embedded, and autologged.
    note = bundle.concept("notes/slow-query-checklist")
    assert note is not None and note.type == "Note"
    assert {c.id for c in note.links(type="BUILDS_ON")} == {"runbooks/slow-queries"}
    assert any("slow-query-checklist" in e["text"] for e in bundle.log())
    hits = bundle.search("checklist", mode="text", k=5)
    assert "notes/slow-query-checklist" in {h.concept.id for h in hits}


def test_agent_loop_stops_at_max_turns(kb):
    okf_agent, bundle = kb
    endless = ScriptedChat(
        [
            {"role": "assistant", "content": None,
             "tool_calls": [_tool_call("browse", f"b{i}")]}
            for i in range(5)
        ]
    )
    answer = okf_agent.run_agent(bundle, "loop forever", chat=endless, max_turns=3, verbose=False)
    assert "max_turns" in answer


def test_tool_schemas_match_implementations(kb):
    okf_agent, bundle = kb
    tools = okf_agent.BundleTools(bundle)
    for schema in tools.schemas:
        name = schema["function"]["name"]
        assert callable(getattr(tools, f"_{name}"))
        assert schema["function"]["description"]
