# PDF GraphRAG retrieval evaluation

Corpus: Anthropic, *Building Effective AI Agents*. Same pipeline as `pdf_chunking_colab.ipynb`.

This is a retrieval/context evaluation, not a generated-answer benchmark. It asks: do newer graph features add useful evidence, or mostly noise?

## Corpus build

- Sections: 40
- Passages/chunks: 94
- Semantic proximity edges: 218 `SEMANTIC_NEAR` over 94 chunks
- Semantic graph parameters: `k=3`, `min_score=0.45`, `undirected=True`, `labels=['Chunk']`

## Metrics

- `section_hit`: at least one retrieved chunk is in the expected section(s).
- `term_recall`: fraction of manually specified evidence terms present in the retrieved context.
- `ce_max` / `ce_mean`: cross-encoder relevance scores for query/chunk pairs (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Higher is better within a query.
- `chunks` / `chars`: context budget proxy. More is not automatically better.

## Aggregate by strategy

| strategy | section_hit_rate | avg_term_recall | avg_ce_max | avg_ce_mean | avg_chunks | avg_chars |
|---|---:|---:|---:|---:|---:|---:|
| combined_reading_semantic_top1 | 1.00 | 0.92 | 4.39 | -1.76 | 6.2 | 4522 |
| hybrid_reading_top1 | 1.00 | 0.89 | 4.20 | 0.20 | 3.0 | 2259 |
| hybrid_semantic_top1 | 1.00 | 0.92 | 4.39 | -1.31 | 5.2 | 3952 |
| hybrid_top3 | 1.00 | 0.92 | 4.56 | 1.51 | 3.0 | 2438 |
| rerank_vector10_top3 | 1.00 | 0.97 | 4.96 | 3.41 | 3.0 | 2439 |
| semantic_near_top1 | 1.00 | 0.92 | 4.39 | -1.31 | 5.2 | 3952 |
| semantic_near_top3 | 1.00 | 0.97 | 4.79 | -2.76 | 11.5 | 7855 |
| vector_reading_top1 | 1.00 | 0.89 | 4.20 | 0.20 | 3.0 | 2259 |
| vector_reading_top3 | 1.00 | 0.97 | 4.96 | -1.59 | 8.3 | 5854 |
| vector_top3 | 1.00 | 0.92 | 4.56 | 1.51 | 3.0 | 2438 |

## Per-query results

### automation_difference

Query: How do AI agents differ from traditional automation?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 1.00 | 7.33 | 0.07 | 8 | 5925 | 3, 4, 5, 1, 6, 18, 12, 21 | independently, tools, complex, recover, rigid, prewritten |
| hybrid_reading_top1 | True | 1.00 | 7.33 | 3.86 | 3 | 1662 | 3, 4, 5 | independently, tools, complex, recover, rigid, prewritten |
| hybrid_semantic_top1 | True | 1.00 | 7.33 | -0.00 | 7 | 5714 | 4, 1, 6, 5, 18, 12, 21 | independently, tools, complex, recover, rigid, prewritten |
| hybrid_top3 | True | 1.00 | 7.33 | -0.05 | 3 | 1903 | 4, 39, 25 | independently, tools, complex, recover, rigid, prewritten |
| rerank_vector10_top3 | True | 1.00 | 7.33 | 4.75 | 3 | 1978 | 4, 5, 25 | independently, tools, complex, recover, rigid, prewritten |
| semantic_near_top1 | True | 1.00 | 7.33 | -0.00 | 7 | 5714 | 4, 1, 6, 5, 18, 12, 21 | independently, tools, complex, recover, rigid, prewritten |
| semantic_near_top3 | True | 1.00 | 7.33 | -3.46 | 16 | 9531 | 4, 39, 25, 1, 6, 5, 18, 12, 21, 32, 54, 37, 38, 77, 80, 88 | independently, tools, complex, recover, rigid, prewritten |
| vector_reading_top1 | True | 1.00 | 7.33 | 3.86 | 3 | 1662 | 3, 4, 5 | independently, tools, complex, recover, rigid, prewritten |
| vector_reading_top3 | True | 1.00 | 7.33 | -1.25 | 9 | 5831 | 3, 4, 5, 38, 39, 40, 24, 25, 26 | independently, tools, complex, recover, rigid, prewritten |
| vector_top3 | True | 1.00 | 7.33 | -0.05 | 3 | 1903 | 4, 39, 25 | independently, tools, complex, recover, rigid, prewritten |

### single_vs_multi

Query: When should I use a single agent instead of multi-agent orchestration?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 0.67 | 3.03 | -0.92 | 6 | 5253 | 32, 33, 34, 81, 51, 40 | single-agent, multi-agent, complex, specialized |
| hybrid_reading_top1 | True | 0.67 | 3.03 | 0.18 | 3 | 2293 | 32, 33, 34 | single-agent, multi-agent, complex, specialized |
| hybrid_semantic_top1 | True | 0.67 | 3.03 | -0.20 | 5 | 5153 | 33, 34, 81, 51, 40 | single-agent, multi-agent, complex, specialized |
| hybrid_top3 | True | 0.67 | 3.03 | 2.00 | 3 | 3001 | 33, 26, 81 | single-agent, multi-agent, complex, specialized |
| rerank_vector10_top3 | True | 1.00 | 4.03 | 3.21 | 3 | 2030 | 27, 33, 63 | single-agent, multi-agent, complex, specialized, resource, orchestration |
| semantic_near_top1 | True | 0.67 | 3.03 | -0.20 | 5 | 5153 | 33, 34, 81, 51, 40 | single-agent, multi-agent, complex, specialized |
| semantic_near_top3 | True | 0.83 | 3.03 | 0.68 | 12 | 9740 | 33, 26, 81, 90, 31, 34, 51, 40, 89, 82, 83, 84 | single-agent, multi-agent, complex, specialized, resource |
| vector_reading_top1 | True | 0.67 | 3.03 | 0.18 | 3 | 2293 | 32, 33, 34 | single-agent, multi-agent, complex, specialized |
| vector_reading_top3 | True | 0.83 | 4.03 | 0.64 | 9 | 6165 | 32, 33, 34, 25, 26, 27, 80, 81, 82 | single-agent, multi-agent, complex, specialized, resource |
| vector_top3 | True | 0.67 | 3.03 | 2.00 | 3 | 3001 | 33, 26, 81 | single-agent, multi-agent, complex, specialized |

### agent_skills

Query: What are Agent Skills and when should an organization use them?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 1.00 | 4.18 | -0.11 | 4 | 3741 | 20, 21, 22, 4 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| hybrid_reading_top1 | True | 1.00 | 4.18 | 0.17 | 3 | 2641 | 20, 21, 22 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| hybrid_semantic_top1 | True | 1.00 | 4.18 | -0.11 | 4 | 3741 | 21, 22, 20, 4 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| hybrid_top3 | True | 1.00 | 4.18 | 2.57 | 3 | 2929 | 21, 22, 26 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| rerank_vector10_top3 | True | 1.00 | 4.18 | 2.74 | 3 | 3019 | 22, 21, 33 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| semantic_near_top1 | True | 1.00 | 4.18 | -0.11 | 4 | 3741 | 21, 22, 20, 4 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| semantic_near_top3 | True | 1.00 | 4.18 | -2.43 | 10 | 7925 | 21, 22, 26, 20, 4, 85, 84, 81, 90, 31 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| vector_reading_top1 | True | 1.00 | 4.18 | 0.17 | 3 | 2641 | 20, 21, 22 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| vector_reading_top3 | True | 1.00 | 4.18 | -2.09 | 7 | 5179 | 20, 21, 22, 23, 25, 26, 27 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| vector_top3 | True | 1.00 | 4.18 | 2.57 | 3 | 2929 | 21, 22, 26 | skills, domain-specific, standardized workflows, specialized, expertise, integration |

### customer_support

Query: What customer support use cases are described for AI agents?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 1.00 | 3.17 | -2.19 | 6 | 5032 | 11, 12, 13, 2, 1, 4 | customer, support, intercom, fin, resolution, operations |
| hybrid_reading_top1 | True | 0.83 | 3.17 | -4.62 | 3 | 2232 | 11, 12, 13 | customer, support, intercom, fin, resolution |
| hybrid_semantic_top1 | True | 1.00 | 3.17 | 0.97 | 4 | 3526 | 12, 2, 1, 4 | customer, support, intercom, fin, resolution, operations |
| hybrid_top3 | True | 1.00 | 5.56 | 2.19 | 3 | 1977 | 12, 17, 2 | customer, support, intercom, fin, resolution, operations |
| rerank_vector10_top3 | True | 0.83 | 5.56 | 3.59 | 3 | 2477 | 17, 12, 4 | customer, support, intercom, fin, resolution |
| semantic_near_top1 | True | 1.00 | 3.17 | 0.97 | 4 | 3526 | 12, 2, 1, 4 | customer, support, intercom, fin, resolution, operations |
| semantic_near_top3 | True | 1.00 | 5.56 | -3.06 | 10 | 6567 | 12, 17, 2, 1, 6, 89, 4, 86, 31, 78 | customer, support, intercom, fin, resolution, operations |
| vector_reading_top1 | True | 0.83 | 3.17 | -4.62 | 3 | 2232 | 11, 12, 13 | customer, support, intercom, fin, resolution |
| vector_reading_top3 | True | 1.00 | 5.56 | -2.01 | 9 | 6468 | 11, 12, 13, 16, 17, 18, 1, 2, 3 | customer, support, intercom, fin, resolution, operations |
| vector_top3 | True | 1.00 | 5.56 | 2.19 | 3 | 1977 | 12, 17, 2 | customer, support, intercom, fin, resolution, operations |

### evaluator_optimizer

Query: What is the evaluator-optimizer workflow pattern useful for?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 1.00 | 7.27 | -1.03 | 6 | 3256 | 70, 71, 72, 60, 61, 54 | evaluator, optimizer, feedback, iterative, quality, loop |
| hybrid_reading_top1 | True | 1.00 | 7.27 | 5.96 | 3 | 2544 | 70, 71, 72 | evaluator, optimizer, feedback, iterative, quality, loop |
| hybrid_semantic_top1 | True | 1.00 | 7.27 | -2.19 | 5 | 2522 | 71, 70, 60, 61, 54 | evaluator, optimizer, feedback, iterative, quality, loop |
| hybrid_top3 | True | 0.83 | 5.89 | 2.55 | 3 | 2549 | 71, 55, 72 | evaluator, optimizer, feedback, iterative, quality |
| rerank_vector10_top3 | True | 1.00 | 7.27 | 5.96 | 3 | 2544 | 70, 71, 72 | evaluator, optimizer, feedback, iterative, quality, loop |
| semantic_near_top1 | True | 1.00 | 7.27 | -2.19 | 5 | 2522 | 71, 70, 60, 61, 54 | evaluator, optimizer, feedback, iterative, quality, loop |
| semantic_near_top3 | True | 1.00 | 7.27 | -2.28 | 10 | 6334 | 71, 55, 72, 57, 61, 60, 56, 70, 54, 73 | evaluator, optimizer, feedback, iterative, quality, loop |
| vector_reading_top1 | True | 1.00 | 7.27 | 5.96 | 3 | 2544 | 70, 71, 72 | evaluator, optimizer, feedback, iterative, quality, loop |
| vector_reading_top3 | True | 1.00 | 7.27 | 0.14 | 7 | 5055 | 70, 71, 72, 54, 55, 56, 73 | evaluator, optimizer, feedback, iterative, quality, loop |
| vector_top3 | True | 0.83 | 5.89 | 2.55 | 3 | 2549 | 71, 55, 72 | evaluator, optimizer, feedback, iterative, quality |

### observability

Query: Why do agent systems need observability and audit trails?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 0.83 | 1.37 | -6.36 | 7 | 3923 | 22, 23, 24, 74, 0, 61, 35 | observable, explain, trace, decision, black |
| hybrid_reading_top1 | True | 0.83 | 0.23 | -4.35 | 3 | 2182 | 22, 23, 24 | observable, explain, trace, decision, black |
| hybrid_semantic_top1 | True | 0.83 | 1.37 | -6.31 | 6 | 3056 | 23, 74, 0, 61, 24, 35 | observable, explain, trace, decision, black |
| hybrid_top3 | True | 1.00 | 1.37 | -0.18 | 3 | 2271 | 23, 35, 79 | observable, explain, audit, trace, decision, black |
| rerank_vector10_top3 | True | 1.00 | 1.37 | 0.20 | 3 | 2584 | 35, 23, 55 | observable, explain, audit, trace, decision, black |
| semantic_near_top1 | True | 0.83 | 1.37 | -6.31 | 6 | 3056 | 23, 74, 0, 61, 24, 35 | observable, explain, trace, decision, black |
| semantic_near_top3 | True | 1.00 | 1.37 | -6.02 | 11 | 7034 | 23, 35, 79, 74, 0, 61, 24, 89, 80, 85, 84 | observable, explain, audit, trace, decision, black |
| vector_reading_top1 | True | 0.83 | 0.23 | -4.35 | 3 | 2182 | 22, 23, 24 | observable, explain, trace, decision, black |
| vector_reading_top3 | True | 1.00 | 1.37 | -4.98 | 9 | 6427 | 22, 23, 24, 34, 35, 36, 78, 79, 80 | observable, explain, audit, trace, decision, black |
| vector_top3 | True | 1.00 | 1.37 | -0.18 | 3 | 2271 | 23, 35, 79 | observable, explain, audit, trace, decision, black |

## Interpretation

Use this table defensively. A graph-expanded strategy wins only if it improves evidence coverage or relevance at an acceptable context cost. If it adds many chunks with lower `ce_mean`, it is exploration/visualization value, not proof of better answer quality.

For article claims, the honest framing is: semantic proximity edges make vector similarity traversable and explainable as a graph. They can improve recall/context assembly, but they do not automatically improve top-k ranking; reranking or budgeting is still needed to control noise.