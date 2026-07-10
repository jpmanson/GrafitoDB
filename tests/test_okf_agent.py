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
        self.seen_tools: list[list[dict]] = []

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        assert any(t["function"]["name"] == "remember" for t in tools)
        self.seen.append(list(messages))
        self.seen_tools.append(list(tools))
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


def test_messages_none_discards_history_between_calls(kb):
    chat = ScriptedChat([{"role": "assistant", "content": "first"}, {"role": "assistant", "content": "second"}])
    run_agent(kb, "one", chat=chat)
    run_agent(kb, "two", chat=chat)
    # Each call started a fresh conversation: one system + one user message.
    assert [m["role"] for m in chat.seen[0]] == ["system", "user"]
    assert [m["role"] for m in chat.seen[1]] == ["system", "user"]


def test_messages_list_threads_conversation_across_calls(kb):
    chat = ScriptedChat([{"role": "assistant", "content": "first"}, {"role": "assistant", "content": "second"}])
    history: list[dict] = []
    first = run_agent(kb, "one", chat=chat, messages=history)
    assert first == "first"
    assert [m["role"] for m in history] == ["system", "user", "assistant"]

    second = run_agent(kb, "two", chat=chat, messages=history)
    assert second == "second"
    # The second call's system prompt is the SAME message object (not re-created)
    # and the full prior turn is still present ahead of the new question.
    assert [m["role"] for m in history] == ["system", "user", "assistant", "user", "assistant"]
    assert chat.seen[1][0] is history[0]
    assert history[1]["content"] == "one"
    assert history[3]["content"] == "two"


def test_messages_empty_list_gets_system_prompt(kb):
    chat = ScriptedChat([{"role": "assistant", "content": "ok"}])
    history: list[dict] = []
    run_agent(kb, "hi", chat=chat, messages=history)
    assert history[0]["role"] == "system"


class FakeToolSet:
    """A minimal custom toolset: no base class, just schemas + call."""

    def __init__(self, name: str = "ping", reply: str = '{"pong": true}') -> None:
        self.name = name
        self.reply = reply
        self.calls: list[tuple[str, dict]] = []
        self.schemas = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "A custom app tool unrelated to the bundle.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def call(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return self.reply


def test_extra_tools_are_offered_to_the_model_and_dispatched(kb):
    fake = FakeToolSet()
    chat = ScriptedChat(
        [
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("ping", "c1")]},
            {"role": "assistant", "content": "done"},
        ]
    )
    answer = run_agent(kb, "use the custom tool", chat=chat, extra_tools=[fake])
    assert answer == "done"
    assert fake.calls == [("ping", {})]
    # The model saw both the bundle's schemas and the extra one.
    schema_names = {s["function"]["name"] for s in chat.seen_tools[0]}
    assert "ping" in schema_names and "search" in schema_names


def test_extra_tools_duplicate_name_raises(kb):
    chat = ScriptedChat([{"role": "assistant", "content": "unreachable"}])
    with pytest.raises(ValueError, match="search"):
        run_agent(kb, "hi", chat=chat, extra_tools=[FakeToolSet(name="search")])


def test_unknown_tool_name_returns_error_without_crashing(kb):
    chat = ScriptedChat(
        [
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("does_not_exist", "c1")]},
            {"role": "assistant", "content": "recovered"},
        ]
    )
    answer = run_agent(kb, "hi", chat=chat, extra_tools=[FakeToolSet()])
    assert answer == "recovered"
    tool_message = next(m for m in chat.seen[1] if m.get("role") == "tool")
    assert "error" in json.loads(tool_message["content"])


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


# --- Anthropic adapter: format conversion (pure, no SDK required) ---------------


def test_anthropic_message_conversion():
    from grafito.okf.agent import _anthropic_messages, _anthropic_tools

    system, converted = _anthropic_messages(
        [
            {"role": "system", "content": "You are a KB assistant."},
            {"role": "user", "content": "query got slow"},
            {"role": "assistant", "content": "Let me look.", "tool_calls": [
                _tool_call("search", "t1", query="slow"),
                _tool_call("browse", "t2"),
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "[hits]"},
            {"role": "tool", "tool_call_id": "t2", "content": "{index}"},
        ]
    )
    assert system == "You are a KB assistant."
    assert [m["role"] for m in converted] == ["user", "assistant", "user"]
    assistant = converted[1]["content"]
    assert assistant[0] == {"type": "text", "text": "Let me look."}
    assert assistant[1]["type"] == "tool_use"
    assert assistant[1]["input"] == {"query": "slow"}
    # Parallel tool results merge into ONE user message.
    results = converted[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result"]
    assert {b["tool_use_id"] for b in results} == {"t1", "t2"}

    tools = _anthropic_tools([{
        "type": "function",
        "function": {"name": "search", "description": "d", "parameters": {"type": "object"}},
    }])
    assert tools == [{"name": "search", "description": "d", "input_schema": {"type": "object"}}]


def test_anthropic_response_conversion_and_replay():
    from types import SimpleNamespace

    from grafito.okf.agent import _anthropic_messages, _openai_message

    response = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="", signature="sig",
                            model_dump=lambda: {"type": "thinking", "thinking": "", "signature": "sig"}),
            SimpleNamespace(type="text", text="Checking the runbook.",
                            model_dump=lambda: {"type": "text", "text": "Checking the runbook."}),
            SimpleNamespace(type="tool_use", id="tu1", name="open",
                            input={"concept_id": "runbooks/slow-queries"},
                            model_dump=lambda: {"type": "tool_use", "id": "tu1", "name": "open",
                                                "input": {"concept_id": "runbooks/slow-queries"}}),
        ],
    )
    message = _openai_message(response)
    assert message["content"] == "Checking the runbook."
    assert message["tool_calls"][0]["id"] == "tu1"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "concept_id": "runbooks/slow-queries"
    }

    # Replay: the raw blocks (thinking included) echo back verbatim.
    _, converted = _anthropic_messages([
        {"role": "user", "content": "q"},
        message,
        {"role": "tool", "tool_call_id": "tu1", "content": "{...}"},
    ])
    assert converted[1]["content"][0]["type"] == "thinking"
    assert converted[1]["content"][2] == {
        "type": "tool_use", "id": "tu1", "name": "open",
        "input": {"concept_id": "runbooks/slow-queries"},
    }


def test_anthropic_chat_requires_sdk_or_builds():
    pytest.importorskip("anthropic")
    from grafito.okf import AnthropicChat

    with AnthropicChat(api_key="x", model="m") as chat:
        assert chat.model == "m"
        assert isinstance(chat, Chat)
