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
    run = run_agent(kb, "query got slow, what do I do?", chat=chat)
    assert run.answer == "Use the runbook (runbooks/slow-queries)."

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
    run = run_agent(kb, "loop forever", chat=endless, max_turns=3)
    assert run.stopped_early and run.answer == ""
    assert run.turns == 3


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
    assert first.answer == "first"
    assert [m["role"] for m in history] == ["system", "user", "assistant"]

    second = run_agent(kb, "two", chat=chat, messages=history)
    assert second.answer == "second"
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
    run = run_agent(kb, "use the custom tool", chat=chat, extra_tools=[fake])
    assert run.answer == "done"
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
    run = run_agent(kb, "hi", chat=chat, extra_tools=[FakeToolSet()])
    assert run.answer == "recovered"
    tool_message = next(m for m in chat.seen[1] if m.get("role") == "tool")
    assert "error" in json.loads(tool_message["content"])


# --- Observability: what the run cost -------------------------------------------


def test_run_records_every_tool_call_with_size_and_error(kb):
    chat = ScriptedChat(
        [
            {"role": "assistant", "tool_calls": [
                _tool_call("open", "c1", concept_id="runbooks/slow-queries"),
                _tool_call("open", "c2", concept_id="does/not/exist"),
            ]},
            {"role": "assistant", "content": "done"},
        ]
    )
    run = run_agent(kb, "hi", chat=chat)

    assert run.turns == 2 and not run.stopped_early
    assert [(c.turn, c.name) for c in run.tool_calls] == [(1, "open"), (1, "open")]

    opened, missing = run.tool_calls
    assert opened.args == {"concept_id": "runbooks/slow-queries"}
    assert opened.error is None
    # The body the model read is what grows the context — it must be counted.
    assert opened.result_bytes > 0
    assert missing.error is not None and "does/not/exist" in missing.error

    stats = run.summary()
    assert stats["tool_calls"] == 2 and stats["errors"] == 1
    assert stats["by_tool"]["open"] == {
        "calls": 2, "errors": 1, "bytes": opened.result_bytes + missing.result_bytes
    }
    assert stats["result_bytes"] == stats["by_tool"]["open"]["bytes"]


def test_summary_counts_repeated_identical_calls(kb):
    """Re-opening the same concept is the signal worth surfacing."""
    chat = ScriptedChat(
        [
            {"role": "assistant", "tool_calls": [_tool_call("browse", "c1", layer="runbooks")]},
            {"role": "assistant", "tool_calls": [_tool_call("browse", "c2", layer="runbooks")]},
            {"role": "assistant", "tool_calls": [_tool_call("browse", "c3", layer="glossary")]},
            {"role": "assistant", "content": "done"},
        ]
    )
    run = run_agent(kb, "hi", chat=chat)
    # Two of the three browses are the same call; only the second one is repeat.
    assert run.summary()["repeated_calls"] == 1


def test_usage_is_aggregated_across_turns_when_the_chat_reports_it(kb):
    turns = [
        {"role": "assistant", "tool_calls": [_tool_call("browse", "c1")], "_usage": {
            "input_tokens": 100, "cached_input_tokens": 0,
            "cache_write_tokens": 80, "output_tokens": 10,
        }},
        {"role": "assistant", "content": "done", "_usage": {
            "input_tokens": 150, "cached_input_tokens": 80,
            "cache_write_tokens": 0, "output_tokens": 20,
        }},
    ]
    run = run_agent(kb, "hi", chat=ScriptedChat(turns))
    assert run.usage == {
        "input_tokens": 250,
        "cached_input_tokens": 80,
        "cache_write_tokens": 80,
        "output_tokens": 30,
        "requests": 2,
    }


def test_usage_stays_empty_when_the_chat_reports_none(kb):
    """A bring-your-own Chat that reports nothing must not produce fake numbers."""
    run = run_agent(kb, "hi", chat=ScriptedChat([{"role": "assistant", "content": "done"}]))
    assert run.usage == {} and run.summary()["usage"] == {}


def test_agent_run_str_is_the_answer(kb):
    run = run_agent(kb, "hi", chat=ScriptedChat([{"role": "assistant", "content": "the answer"}]))
    assert f"{run}" == "the answer"
    assert run.messages is not None and run.messages[-1]["content"] == "the answer"


def test_private_keys_are_stripped_before_hitting_an_openai_endpoint():
    """`_usage` / `_anthropic_content` are bookkeeping, not part of the wire format."""
    from grafito.okf.agent import _without_private

    sent = _without_private([
        {"role": "assistant", "content": "hi", "_usage": {"input_tokens": 1}, "_anthropic_content": []},
        {"role": "user", "content": "q"},
    ])
    assert sent == [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "q"}]


def test_anthropic_usage_sums_the_cache_buckets_into_input_tokens():
    """Anthropic reports the uncached remainder; the loop reports the whole prompt."""
    from grafito.okf.agent import _anthropic_usage

    class Usage:
        input_tokens = 100
        cache_read_input_tokens = 900
        cache_creation_input_tokens = 50
        output_tokens = 7

    class Response:
        usage = Usage()

    assert _anthropic_usage(Response()) == {
        "input_tokens": 1050,
        "cached_input_tokens": 900,
        "cache_write_tokens": 50,
        "output_tokens": 7,
    }


def test_openai_usage_keeps_prompt_tokens_as_the_total():
    from grafito.okf.agent import _openai_usage

    usage = _openai_usage({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 900},
    }})
    assert usage == {
        "input_tokens": 1000,
        "cached_input_tokens": 900,
        "cache_write_tokens": 0,
        "output_tokens": 7,
    }
    assert _openai_usage({}) is None


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


# --- scoped toolsets (where=/tag=) ------------------------------------------


@pytest.fixture
def scoped_kb(tmp_path) -> OKFBundle:
    """Bundle with a public/restricted split, for toolset-scoping tests."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "public.md").write_text(
        "---\ntype: Doc\ntitle: Public doc\nconfidentiality: public\n---\n\n"
        "Deployment runbook.\n\n# Links\n[Secret](/docs/secret.md)\n",
        encoding="utf-8",
    )
    (docs / "secret.md").write_text(
        "---\ntype: Doc\ntitle: Secret doc\nconfidentiality: restricted\n---\n\n"
        "Deployment credentials rotation.\n",
        encoding="utf-8",
    )
    bundle = OKFBundle.load(str(tmp_path), import_log=True)
    yield bundle
    bundle.db.close()


@pytest.fixture
def scoped_tools(scoped_kb) -> BundleTools:
    return BundleTools(scoped_kb, where={"confidentiality": "public"})


def test_scoped_tools_hide_concepts_from_browse_and_search(scoped_tools):
    listing = json.loads(scoped_tools.call("browse", {"layer": "docs"}))
    assert [c["id"] for c in listing["concepts"]] == ["docs/public"]
    hits = json.loads(scoped_tools.call("search", {"query": "deployment"}))
    assert [h["id"] for h in hits] == ["docs/public"]


def test_scoped_tools_refuse_hidden_concept_as_if_absent(scoped_tools):
    """A filtered concept is indistinguishable from a nonexistent one."""
    hidden = json.loads(scoped_tools.call("open", {"concept_id": "docs/secret"}))
    absent = json.loads(scoped_tools.call("open", {"concept_id": "docs/nope"}))
    assert hidden == {"error": "Unknown concept: docs/secret"}
    assert absent == {"error": "Unknown concept: docs/nope"}


def test_scoped_tools_filter_edges_and_traversal(scoped_tools):
    # public links to secret, but neither the edge list nor follow exposes it.
    assert json.loads(scoped_tools.call("open", {"concept_id": "docs/public"}))["links"] == []
    assert json.loads(scoped_tools.call("follow", {"concept_id": "docs/public"})) == []
    assert json.loads(scoped_tools.call("follow", {"concept_id": "docs/secret"})) == {
        "error": "Unknown concept: docs/secret"
    }


def test_scoped_tools_never_expose_hidden_content(scoped_tools):
    """Structure is filtered; hidden titles and bodies never appear."""
    blob = " ".join(
        [
            scoped_tools.call("browse", {"layer": "docs"}),
            scoped_tools.call("search", {"query": "deployment credentials rotation"}),
            scoped_tools.call("follow", {"concept_id": "docs/public"}),
            scoped_tools.call("open", {"concept_id": "docs/public"}),
        ]
    )
    assert "Secret doc" not in blob
    assert "credentials rotation" not in blob


def test_scoped_tools_history_refuses_hidden_concept(scoped_tools):
    assert json.loads(scoped_tools.call("history", {"concept_id": "docs/secret"})) == {
        "error": "Unknown concept: docs/secret"
    }


def test_scoped_tools_layers_do_not_leak_hidden_counts(scoped_kb, scoped_tools):
    assert scoped_kb.layers() == {"docs": 2}      # unfiltered sees both
    assert scoped_tools.layers() == {"docs": 1}   # the prompt sees one


def test_unscoped_tools_see_everything(scoped_kb):
    """No filter means no behaviour change (regression guard)."""
    tools = BundleTools(scoped_kb)
    listing = json.loads(tools.call("browse", {"layer": "docs"}))
    assert [c["id"] for c in listing["concepts"]] == ["docs/public", "docs/secret"]
    assert "error" not in json.loads(tools.call("open", {"concept_id": "docs/secret"}))


def test_run_agent_accepts_injected_scoped_tools(scoped_kb, scoped_tools):
    """The model cannot reach a hidden concept through the loop either."""
    chat = ScriptedChat(
        [
            {"role": "assistant", "tool_calls": [_tool_call("open", "1", concept_id="docs/secret")]},
            {"role": "assistant", "content": "I could not find that concept."},
        ]
    )
    run = run_agent(scoped_kb, "read the secret", chat=chat, tools=scoped_tools)
    assert run.answer == "I could not find that concept."
    tool_messages = [m for m in chat.seen[-1] if m.get("role") == "tool"]
    assert json.loads(tool_messages[0]["content"]) == {"error": "Unknown concept: docs/secret"}
    # The system prompt is built from the filtered layers.
    assert "Secret" not in chat.seen[-1][0]["content"]


def test_scoped_layers_keep_root_concepts_visible(tmp_path):
    """Filtered layers keep kb.layers()' shape, root bucket included."""
    (tmp_path / "d").mkdir()
    (tmp_path / "root.md").write_text(
        "---\ntype: Doc\ntitle: Root\nstatus: approved\n---\n\nbody\n", encoding="utf-8"
    )
    (tmp_path / "d" / "x.md").write_text(
        "---\ntype: Doc\ntitle: X\nstatus: approved\n---\n\nbody\n", encoding="utf-8"
    )
    (tmp_path / "d" / "y.md").write_text(
        "---\ntype: Doc\ntitle: Y\nstatus: draft\n---\n\nbody\n", encoding="utf-8"
    )
    bundle = OKFBundle.load(str(tmp_path))
    try:
        assert bundle.layers() == {".": 1, "d": 2}
        scoped = BundleTools(bundle, where={"status": "approved"})
        assert scoped.layers() == {".": 1, "d": 1}
    finally:
        bundle.db.close()
