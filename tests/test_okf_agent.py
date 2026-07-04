"""Tests for the agentic GraphRAG toolkit (grafito.okf.agent).

The loop is driven with a scripted fake ``chat`` — no endpoint, no network —
exercising the tool schemas, dispatch, error handling, and the write path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grafito.okf import BundleTools, Chat, OKFBundle, run_agent

pytest.importorskip("yaml")

KB = Path("examples") / "okf" / "okf_knowledge_base"


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
def kb() -> OKFBundle:
    # No embedder: the search tool's hybrid mode degrades to full-text.
    bundle = OKFBundle.load(str(KB), autolog=True)
    yield bundle
    bundle.db.close()


def test_agent_loop_explores_and_remembers(kb):
    chat = ScriptedChat(
        [
            {"role": "assistant", "content": None, "tool_calls": [
                _tool_call("search", "c1", query="slow query"),
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
    answer = run_agent(kb, "query got slow, what do I do?", chat=chat)
    assert answer == "Use the runbook (runbooks/slow-queries)."

    # The default system prompt orients the model with the bundle layers.
    system = chat.seen[0][0]
    assert system["role"] == "system" and "runbooks" in system["content"]

    # Tool results reached the model as role=tool messages.
    tool_messages = [m for m in chat.seen[-1] if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"c1", "c2", "c3", "c4", "c5"}
    opened = json.loads(next(m for m in tool_messages if m["tool_call_id"] == "c3")["content"])
    assert opened["id"] == "runbooks/slow-queries"
    assert opened["body"]
    failed = json.loads(next(m for m in tool_messages if m["tool_call_id"] == "c4")["content"])
    assert "error" in failed

    # The write path: note created, typed-linked, indexed, and autologged.
    note = kb.concept("notes/slow-query-checklist")
    assert note is not None and note.type == "Note"
    assert {c.id for c in note.links(type="BUILDS_ON")} == {"runbooks/slow-queries"}
    assert any("slow-query-checklist" in e["text"] for e in kb.log())
    hits = kb.search("checklist", mode="text", k=5)
    assert "notes/slow-query-checklist" in {h.concept.id for h in hits}


def test_agent_loop_stops_at_max_turns(kb):
    endless = ScriptedChat(
        [
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("browse", f"b{i}")]}
            for i in range(5)
        ]
    )
    answer = run_agent(kb, "loop forever", chat=endless, max_turns=3)
    assert "max_turns" in answer


def test_custom_system_prompt_is_used(kb):
    chat = ScriptedChat([{"role": "assistant", "content": "ok"}])
    run_agent(kb, "hi", chat=chat, system="You are a test fixture.")
    assert chat.seen[0][0]["content"] == "You are a test fixture."


def test_tool_schemas_match_implementations(kb):
    tools = BundleTools(kb)
    for schema in tools.schemas:
        name = schema["function"]["name"]
        assert callable(getattr(tools, f"_{name}"))
        assert schema["function"]["description"]


def test_scripted_chat_satisfies_protocol():
    assert isinstance(ScriptedChat([]), Chat)


def test_openai_chat_requires_httpx_or_builds():
    pytest.importorskip("httpx")
    from grafito.okf import OpenAIChat

    with OpenAIChat(base_url="http://localhost:1/v1", api_key="x", model="m") as chat:
        assert chat.base_url == "http://localhost:1/v1"
        assert chat.model == "m"
        assert isinstance(chat, Chat)
