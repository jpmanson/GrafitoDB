# Invariant tests

Property tests over generated corpora, as opposed to the example-based tests in
`tests/test_*.py`.

## Why these exist

Every bug fixed in 0.7.1 and 0.7.2 was a violated invariant that the unit tests
did not happen to exercise:

| Bug | Invariant it broke |
| --- | --- |
| Rebuild deleted before computing | a failed build changes nothing |
| Hand-made edge excluded its endpoints | refresh eventually links every node |
| `max_edges` stranded nodes it never searched | refresh converges to the full graph |
| `replace=False` duplicated the graph | appending does not duplicate |

All four were found by reading the code and reasoning about state, not by a
failing test. That does not scale. Each states a property and checks it across a
matrix of `k`, `min_score` and `undirected` on corpora with known structure.

The `max_edges` cap moved from per-edge to per-node because of a property here:
cutting a node off mid-search left it looking processed with an incomplete
neighbourhood, and no later refresh would finish it.

## Layout

- `corpus.py` — deterministic corpora with planted clusters. Vectors are written
  directly rather than through an embedding function, so the geometry under test
  is exactly the geometry described. `Corpus.cluster_of()` is ground truth, which
  is what lets community purity and recall be asserted rather than eyeballed.
- `test_semantic_graph_invariants.py` — how edges are built.
- `test_retrieval_invariants.py` — what subgraph retrieval and analysis return.

Corpora are small on purpose. These properties are structural; they do not become
truer with more nodes, and the matrix already multiplies the run count.

## Verifying the harness still bites

A property test that cannot fail is worse than none — it reports safety it does
not provide. Check it by reverting a fix and confirming something goes red:

```bash
cp grafito/database.py /tmp/database.py.bak

# revert the 0.7.2 fix: count targets as processed, not just sources
python - <<'EOF'
p = "grafito/database.py"
s = open(p).read()
s = s.replace("linked = {source for source, _ in existing}",
              "linked = {n for pair in existing for n in pair}")
open(p, "w").write(s)
EOF

uv run pytest tests/invariants/ -q --no-cov   # expect failures
cp /tmp/database.py.bak grafito/database.py
```

Reverting each of the four fixes above fails 4, 12, 12 and 2 properties
respectively. Do this after adding properties: a new one that survives every
mutation is probably asserting something trivially true.
