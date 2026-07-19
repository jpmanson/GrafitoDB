# Agentic Search Over an OKF Bundle

How does an agent explore an OKF bundle (a directory of markdown concepts)
without the whole knowledge base ending up in the LLM's context window? This
page walks through the mechanism step by step — what gets sent to the model,
when, and why the context stays small even as the bundle grows.

Two APIs implement this; both live in `grafito.okf`:

- **`run_agent()`** — the model drives its own exploration through tool calls,
  turn by turn. This page is mostly about this path.
- **`OKFBundle.context()`** — one deterministic retrieve → expand → pack pass,
  no loop. Covered at the end as the non-agentic alternative.

See [Open Knowledge Format (OKF)](okf.md) for the format itself and the full
API reference; this page is about the *retrieval mechanics*, not the spec.

## The core idea: progressive disclosure

A naive approach would stuff every concept's full markdown body into the
prompt. That doesn't scale past a handful of documents — a 200-concept bundle
could easily be hundreds of thousands of tokens.

Instead, every tool the agent has access to returns **metadata by default**
(id, title, description — a few dozen tokens per concept) and **only one
tool** — `open` — returns the full markdown body of a concept. The agent
decides, concept by concept, whether a piece of content is worth spending
context budget on. Most of the bundle is never read; only what the agent
chooses to `open()` after triaging with cheaper tools.

This mirrors how a person skims a folder of notes: check the directory
listing, skim titles, open the two or three files that look relevant — not
read every file front to back.

Note what this does and does not bound. It bounds how much of the *bundle*
reaches the model — that part holds, and it is why bundle size stops
mattering. It does not make the loop cheap: each turn re-sends everything
read so far, so total spend grows with the number of turns, not with the
size of the knowledge base. [Which one should you use?](#which-one-should-you-use)
has the measured numbers.

## Step by step

### 0. Bootstrap: the system prompt (near-zero cost)

Before any tool call, the agent already knows the *shape* of the bundle, not
its content. `run_agent()`'s default system prompt embeds `kb.layers()`:

```python
kb.layers()
# {'decisions': 3, 'glossary': 3, 'runbooks': 1}
```

That's it — a count per top-level directory. For a bundle with hundreds of
concepts this is still a handful of tokens. The system prompt also tells the
model the exploration protocol (browse/search → open/follow → answer with
citations → optionally `remember`), so it knows the tools exist and roughly
when to reach for each one — see `DEFAULT_SYSTEM_PROMPT` in
`grafito/okf/agent.py`.

### 1. Triage: `browse` and `search` (metadata only, no bodies)

Two tools let the agent narrow down candidates before reading anything:

**`browse(layer=None)`** — the in-memory equivalent of an `index.md`: child
subdirectories and the concepts directly in a directory, titles and
descriptions only.

```python
kb.index("runbooks")
# {"layer": "runbooks", "subdirs": {},
#  "concepts": [{"id": "runbooks/slow-queries", "title": "Triaging a slow graph query",
#                "description": "Steps to diagnose and mitigate...", "type": "Playbook"}]}
```

**`search(query, k=5)`** — hybrid (or semantic/text) ranked search. Same
shape: id, title, description, and a relevance score — never the body.

```python
kb.search("slow query", k=5)
# -> [Hit(concept=<Concept 'runbooks/slow-queries'>, score=0.83), ...]
# the tool serializes each hit as: {"id", "title", "description", "score"}
```

Both are cheap regardless of how big the bundle is — the cost is
`O(concepts touched)`, not `O(bundle size)`, and a "concept touched" here
only costs a title + description, not a body.

### 2. Read: `open` (the only tool that spends real budget)

Only when a concept looks worth it does the agent call:

```python
kb["runbooks/slow-queries"]
# id, type, title, description, tags, body, links (typed, excluding CITES), cites
```

This is the single point in the whole toolset where the full markdown body
enters the context. Everything before this step was metadata; this step is
where the agent actually "reads."

### 3. Traverse: `follow` (graph neighbors, still metadata only)

From an opened concept, the agent can pull in linked concepts — outgoing or
incoming, optionally restricted to one relationship type (for bundles
imported with `typed_links=True`) — without opening them:

```python
kb.concept("runbooks/slow-queries").links()
# -> [Concept, ...]; the tool serializes each as {"id", "title", "description"}
```

This is another triage step: the agent discovers *what's connected* to
something it already read, at metadata cost, and only `open()`s the
neighbors that turn out to matter. This is the graph-traversal edge over a
flat vector store — related concepts are reachable by construction, not by
hoping they also scored high on similarity.

### 4. Provenance (optional): `history`

`history(concept_id=None)` returns changelog entries (date, kind, text,
scope) — small, structured records, not full documents. Useful when the
question is about *when* or *why* something changed, not what it currently
says.

### 5. Write: `remember`

The one tool that isn't about reading: `remember(concept_id, title, body,
links=...)` saves a new concept into the bundle (embedded, searchable,
autologged if `autolog=True`). This is how an agent's conclusions become
part of the knowledge base for next time, rather than being lost when the
conversation ends.

## The loop that ties it together

`run_agent()` is a plain tool-calling loop (`grafito/okf/agent.py`):

```python
for turn in range(1, max_turns + 1):
    message = chat(messages, schemas)      # model sees the conversation + tool schemas
    messages.append(message)
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return AgentRun(answer=message.get("content") or "", ...)  # final answer
    for call in tool_calls:
        result = dispatch[name].call(name, args)   # execute against the bundle
        recorded.append(ToolCall(turn, name, args, len(result), _tool_error(result)))
        messages.append({"role": "tool", "content": result})
```

Each iteration, the model decides — based on what it has read *so far* — what
to do next: browse another directory, run a different search, open one more
concept, follow a link, or stop and answer. `messages` only grows by what was
actually requested; a question that's answerable from two `search` calls and
one `open` never sees the other 500 concepts in the bundle. `max_turns`
(default `12`) is a hard ceiling in case the model keeps calling tools
without converging.

### Worked example

Running `examples/okf/okf_agent.py`'s question against a small demo bundle,
with `verbose=True`:

```
Q: A production Cypher query got slow after a data load. What should I do,
   step by step?

  -> search({"query": "slow query"})
     5 result(s): decisions/0002-cypher-subset (Implement a Cypher subset),
     runbooks/slow-queries (Triaging a slow graph query),
     glossary/semantic-search (Semantic search), +2 more
  -> open({"concept_id": "runbooks/slow-queries"})
     runbooks/slow-queries - Triaging a slow graph query (4 link(s), 2 citation(s))
  -> remember({"concept_id": "notes/slow-query-checklist", ...})
     saved notes/slow-query-checklist, linked to [...]
  [4 turn(s), 3 tool call(s), 6120 tool byte(s), 21400 in (14200 cached) / 512 out]

A: ...(cites runbooks/slow-queries)...
```

That closing line is `run.summary()` — see
[Measuring a run](okf.md#measuring-a-run) for the full breakdown, including
which tool the bytes came from and how many calls were repeats.

Three tool calls. Only **one** (`open`) pulled a full document body into
context — the other four concepts `search` surfaced stayed as one-line
metadata and were never read, because the model judged the top hit was
enough to answer. A bundle with 10 concepts or 10,000 costs the same here:
the `search` call is still 5 short results either way.

## Multi-turn conversations: what accumulates

Pass `messages=history` to `run_agent()` to continue a conversation across
calls (see [Multi-turn conversations](okf.md#multi-turn-conversations)).
Tool results — including any full bodies pulled in via `open` — stay in
`history` and get resent to the model every subsequent turn, since the model
needs its own past reasoning to stay coherent. This is the one place context
*does* grow with usage: not with bundle size, but with how many concepts the
conversation has opened so far. For a long-running session, that's the
signal to eventually trim or summarize older turns — bundle size was never
the bottleneck, accumulated conversation is.

You don't have to guess where that line is: `run.summary()["result_bytes"]`
is how much tool output the run added to the conversation, and
`run.usage["input_tokens"]` is what the model was actually charged for
re-reading it each turn.

## The non-agentic alternative: `context()`

Not every use case needs an iterative agent loop. `OKFBundle.context()` does
the same *kind* of budgeted retrieval in one deterministic pass instead of a
multi-turn tool loop:

1. **seed** with `search()` (semantic/text/hybrid);
2. **graph-expand** — follow each hit's outgoing links within `expand_hops`;
3. **pack** greedily into an explicit `budget_tokens` ceiling, seed hits first
   by score, then expanded neighbors — the top hit is never dropped, only
   truncated if it alone exceeds the budget.

```python
pack = kb.context("how do I make a query run faster", budget_tokens=2000)
str(pack)          # prompt-ready text, guaranteed <= budget_tokens (heuristically)
pack.truncated      # True if anything was cut to fit
```

Here the size control is explicit and up front (`budget_tokens`) rather than
emergent from how many tools the model chose to call — a good fit when you
want a single grounded prompt for *any* downstream model, agentic or not.
Graph-expanded blocks are annotated with the relationship that pulled them in
(`via JOINS_WITH`, etc.) — see
[Grounded context for agents](okf.md#grounded-context-for-agents-context) for
the full option list.

## Which one should you use?

Progressive disclosure bounds what the agent *reads*. It does not bound what
you *pay*, and those come apart in a way that is worth seeing measured. The
Messages API is stateless, so every turn re-sends the whole conversation:
the system prompt, the tool schemas, and every tool result so far are billed
again on each model call.

Measured on the demo bundle against Claude Sonnet 4.6 through an
OpenAI-compatible gateway — same question, same model, both paths calling
the model for real:

| path | input tokens | model calls |
|---|---|---|
| `context()` one-shot | 2 132 | 1 |
| `run_agent()` | 7 961 | 4 |
| PydanticAI over the same `BundleTools` | 9 719 | 4 |

The agentic path cost **~3.7x** the one-shot for the same answer, and a
third-party framework cost more still — it adds prompt scaffolding on top of
the same re-send. On a 202-concept bundle the gap widened to ~5x, because
`context()` is capped by `budget_tokens` and therefore does not grow with the
bundle, while the loop keeps paying its prefix every turn.

`run.summary()` breaks the spend down:

```python
run.summary()["input_per_turn"]        # [1121, 1582, 2187, 3071]
run.summary()["resent_input_tokens"]   # 4890 — 61% of the spend was a repeat
run.usage["cached_input_tokens"]       # 0 on this endpoint
```

**So the agentic path is not the cheap one.** What it buys is the ability to
*write* (`remember`, which `context()` cannot do), to decide what to read
when a fixed `budget_tokens` would be the wrong guess, and to amortize its
exploration across a multi-turn conversation.

Two things genuinely reduce the overhead:

- **Prompt caching.** The re-sent prefix is billed at roughly a tenth when
  the endpoint supports it, which is most of the gap above. Compare
  `resent_input_tokens()` against `usage["cached_input_tokens"]` to see how
  much of that saving you are actually getting — a large gap means caching
  is not reaching your requests.
- **A smaller tool surface.** The schema block is re-sent every turn, so it
  is charged once per model call. Trimming six tools to the two an agent
  actually needs took it from ~29% of the run's spend to ~9%:

  ```python
  BundleTools(kb, include=["search", "open"])
  ```

Rule of thumb: reach for `context()` to *answer* a question, and
`run_agent()` when the agent has to *act* — write back to the bundle,
explore across several turns, or work without a budget you can pick in
advance. Then trim the toolset to what that job needs.

> These are single runs against one model, and turn counts vary between runs
> on the same question — treat the ratios as an order of magnitude, not a
> constant. Re-measure with `run.summary()` on your own bundle and model.

## Summary

| Tool / API | Returns bodies? | Cost per call | When the agent uses it |
|---|---|---|---|
| System prompt (`kb.layers()`) | No | A handful of tokens, fixed | Once, at the start |
| `browse` | No | ~1 line per concept in a directory | Orienting / narrowing by directory |
| `search` | No | ~1 line per hit (`k` hits) | Finding candidates by relevance |
| `open` | **Yes** | One full concept body | Only for concepts worth reading |
| `follow` | No | ~1 line per neighbor | Discovering what's linked, before opening |
| `history` | No (short entries) | ~1 line per log entry | Provenance / "when did this change" |
| `remember` | Writes, doesn't read | Small confirmation | Saving conclusions back to the bundle |
| `context()` | Yes, budgeted | Capped by `budget_tokens` | One-shot prompt assembly, no agent loop |

The bundle can grow indefinitely; what enters the model's context is bounded
by how many concepts the agent (or `context()`'s budget) actually decided
were worth reading — never the whole knowledge base. What *does* grow is the
number of turns, and each one re-sends what came before — which is why
`run_agent()` costs several times a `context()` call for the same answer and
why the choice between them is about what the job needs, not about size.
