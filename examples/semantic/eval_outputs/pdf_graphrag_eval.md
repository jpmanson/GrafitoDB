# PDF GraphRAG retrieval evaluation

Corpus: Anthropic, *Building Effective AI Agents*. Same pipeline as `pdf_chunking_colab.ipynb`.

This is a retrieval/context evaluation, not a generated-answer benchmark. It asks: do newer graph features add useful evidence, or mostly noise?

## Corpus build

- Sections: 40
- Passages/chunks: 92
- Semantic proximity edges: 214 `SEMANTIC_NEAR` over 92 chunks
- Semantic graph parameters: `k=3`, `min_score=0.45`, `undirected=True`, `labels=['Chunk']`

## Metrics

- `section_hit`: at least one retrieved chunk is in the expected section(s).
- `term_recall`: fraction of manually specified evidence terms present in the retrieved context.
- `ce_max` / `ce_mean`: cross-encoder relevance scores for query/chunk pairs (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Higher is better within a query.
- `chunks` / `chars`: context budget proxy. More is not automatically better.

## Aggregate by strategy

| strategy | section_hit_rate | avg_term_recall | avg_ce_max | avg_ce_mean | avg_chunks | avg_chars |
|---|---:|---:|---:|---:|---:|---:|
| combined_reading_semantic_top1 | 1.00 | 0.92 | 4.30 | -1.78 | 6.3 | 4559 |
| hybrid_reading_top1 | 1.00 | 0.89 | 4.20 | 0.48 | 3.0 | 2273 |
| hybrid_semantic_top1 | 1.00 | 0.92 | 4.30 | -1.50 | 5.3 | 3938 |
| hybrid_top3 | 1.00 | 0.89 | 4.53 | 0.17 | 3.0 | 2264 |
| rerank_vector10_top3 | 1.00 | 0.94 | 4.87 | 3.08 | 3.0 | 2517 |
| semantic_near_top1 | 1.00 | 0.92 | 4.30 | -1.50 | 5.3 | 3938 |
| semantic_near_top3 | 1.00 | 0.94 | 4.70 | -2.70 | 11.7 | 7553 |
| vector_reading_top1 | 1.00 | 0.89 | 4.20 | 0.48 | 3.0 | 2273 |
| vector_reading_top3 | 1.00 | 0.97 | 4.87 | -2.08 | 8.5 | 5750 |
| vector_top3 | 1.00 | 0.89 | 4.53 | 0.17 | 3.0 | 2264 |

## Per-query results

### automation_difference

Query: How do AI agents differ from traditional automation?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 1.00 | 7.33 | -0.93 | 9 | 6289 | 3, 4, 5, 1, 2, 6, 18, 12, 21 | independently, tools, complex, recover, rigid, prewritten |
| hybrid_reading_top1 | True | 1.00 | 7.33 | 4.05 | 3 | 1546 | 3, 4, 5 | independently, tools, complex, recover, rigid, prewritten |
| hybrid_semantic_top1 | True | 1.00 | 7.33 | -1.11 | 8 | 6078 | 4, 1, 2, 6, 18, 5, 12, 21 | independently, tools, complex, recover, rigid, prewritten |
| hybrid_top3 | True | 1.00 | 7.33 | -4.76 | 3 | 1397 | 4, 38, 35 | independently, tools, complex, recover, rigid, prewritten |
| rerank_vector10_top3 | True | 1.00 | 7.33 | 2.99 | 3 | 1870 | 4, 25, 88 | independently, tools, complex, recover, rigid, prewritten |
| semantic_near_top1 | True | 1.00 | 7.33 | -1.11 | 8 | 6078 | 4, 1, 2, 6, 18, 5, 12, 21 | independently, tools, complex, recover, rigid, prewritten |
| semantic_near_top3 | True | 1.00 | 7.33 | -3.58 | 18 | 9801 | 4, 38, 35, 0, 1, 2, 6, 18, 5, 12, 14, 20, 21, 23, 25, 31, 43, 36 | independently, tools, complex, recover, rigid, prewritten |
| vector_reading_top1 | True | 1.00 | 7.33 | 4.05 | 3 | 1546 | 3, 4, 5 | independently, tools, complex, recover, rigid, prewritten |
| vector_reading_top3 | True | 1.00 | 7.33 | -3.48 | 9 | 5364 | 3, 4, 5, 37, 38, 39, 34, 35, 36 | independently, tools, complex, recover, rigid, prewritten |
| vector_top3 | True | 1.00 | 7.33 | -4.76 | 3 | 1397 | 4, 38, 35 | independently, tools, complex, recover, rigid, prewritten |

### single_vs_multi

Query: When should I use a single agent instead of multi-agent orchestration?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 0.67 | 3.03 | -0.20 | 6 | 5561 | 31, 32, 33, 79, 49, 39 | single-agent, multi-agent, complex, specialized |
| hybrid_reading_top1 | True | 0.67 | 3.03 | 1.54 | 3 | 2606 | 31, 32, 33 | single-agent, multi-agent, complex, specialized |
| hybrid_semantic_top1 | True | 0.67 | 3.03 | -0.48 | 5 | 5151 | 32, 33, 79, 49, 39 | single-agent, multi-agent, complex, specialized |
| hybrid_top3 | True | 0.67 | 3.03 | 2.00 | 3 | 3001 | 32, 26, 79 | single-agent, multi-agent, complex, specialized |
| rerank_vector10_top3 | True | 0.83 | 4.03 | 3.07 | 3 | 2717 | 27, 32, 26 | single-agent, multi-agent, complex, specialized, resource |
| semantic_near_top1 | True | 0.67 | 3.03 | -0.48 | 5 | 5151 | 32, 33, 79, 49, 39 | single-agent, multi-agent, complex, specialized |
| semantic_near_top3 | True | 0.67 | 3.03 | 0.57 | 11 | 9275 | 32, 26, 79, 88, 31, 33, 49, 39, 87, 81, 82 | single-agent, multi-agent, complex, specialized |
| vector_reading_top1 | True | 0.67 | 3.03 | 1.54 | 3 | 2606 | 31, 32, 33 | single-agent, multi-agent, complex, specialized |
| vector_reading_top3 | True | 0.83 | 4.03 | 1.10 | 9 | 6478 | 31, 32, 33, 25, 26, 27, 78, 79, 80 | single-agent, multi-agent, complex, specialized, resource |
| vector_top3 | True | 0.67 | 3.03 | 2.00 | 3 | 3001 | 32, 26, 79 | single-agent, multi-agent, complex, specialized |

### agent_skills

Query: What are Agent Skills and when should an organization use them?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 1.00 | 4.18 | -0.11 | 4 | 3741 | 20, 21, 22, 4 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| hybrid_reading_top1 | True | 1.00 | 4.18 | 0.17 | 3 | 2641 | 20, 21, 22 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| hybrid_semantic_top1 | True | 1.00 | 4.18 | -0.11 | 4 | 3741 | 21, 22, 20, 4 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| hybrid_top3 | True | 1.00 | 4.18 | 2.57 | 3 | 2929 | 21, 22, 26 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| rerank_vector10_top3 | True | 1.00 | 4.18 | 2.74 | 3 | 3019 | 22, 21, 32 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| semantic_near_top1 | True | 1.00 | 4.18 | -0.11 | 4 | 3741 | 21, 22, 20, 4 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| semantic_near_top3 | True | 1.00 | 4.18 | -2.56 | 11 | 8400 | 21, 22, 26, 20, 4, 83, 82, 79, 88, 31, 81 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| vector_reading_top1 | True | 1.00 | 4.18 | 0.17 | 3 | 2641 | 20, 21, 22 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| vector_reading_top3 | True | 1.00 | 4.18 | -2.09 | 7 | 5179 | 20, 21, 22, 23, 25, 26, 27 | skills, domain-specific, standardized workflows, specialized, expertise, integration |
| vector_top3 | True | 1.00 | 4.18 | 2.57 | 3 | 2929 | 21, 22, 26 | skills, domain-specific, standardized workflows, specialized, expertise, integration |

### customer_support

Query: What customer support use cases are described for AI agents?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 1.00 | 3.17 | -2.07 | 6 | 4912 | 11, 12, 13, 2, 1, 4 | customer, support, intercom, fin, resolution, operations |
| hybrid_reading_top1 | True | 0.83 | 3.17 | -4.62 | 3 | 2232 | 11, 12, 13 | customer, support, intercom, fin, resolution |
| hybrid_semantic_top1 | True | 1.00 | 3.17 | 1.15 | 4 | 3406 | 12, 2, 1, 4 | customer, support, intercom, fin, resolution, operations |
| hybrid_top3 | True | 1.00 | 5.56 | 2.43 | 3 | 1857 | 12, 2, 17 | customer, support, intercom, fin, resolution, operations |
| rerank_vector10_top3 | True | 0.83 | 5.56 | 3.59 | 3 | 2477 | 17, 12, 4 | customer, support, intercom, fin, resolution |
| semantic_near_top1 | True | 1.00 | 3.17 | 1.15 | 4 | 3406 | 12, 2, 1, 4 | customer, support, intercom, fin, resolution, operations |
| semantic_near_top3 | True | 1.00 | 5.56 | -2.74 | 10 | 5580 | 12, 2, 17, 6, 3, 4, 1, 84, 31, 76 | customer, support, intercom, fin, resolution, operations |
| vector_reading_top1 | True | 0.83 | 3.17 | -4.62 | 3 | 2232 | 11, 12, 13 | customer, support, intercom, fin, resolution |
| vector_reading_top3 | True | 1.00 | 5.56 | -1.93 | 9 | 6348 | 11, 12, 13, 1, 2, 3, 16, 17, 18 | customer, support, intercom, fin, resolution, operations |
| vector_top3 | True | 1.00 | 5.56 | 2.43 | 3 | 1857 | 12, 2, 17 | customer, support, intercom, fin, resolution, operations |

### evaluator_optimizer

Query: What is the evaluator-optimizer workflow pattern useful for?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 1.00 | 7.27 | -0.97 | 6 | 3142 | 68, 69, 70, 52, 58, 59 | evaluator, optimizer, feedback, iterative, quality, loop |
| hybrid_reading_top1 | True | 1.00 | 7.27 | 6.08 | 3 | 2430 | 68, 69, 70 | evaluator, optimizer, feedback, iterative, quality, loop |
| hybrid_semantic_top1 | True | 1.00 | 7.27 | -2.11 | 5 | 2408 | 69, 68, 52, 58, 59 | evaluator, optimizer, feedback, iterative, quality, loop |
| hybrid_top3 | True | 0.67 | 6.26 | -0.87 | 3 | 2124 | 69, 53, 71 | evaluator, optimizer, feedback, quality |
| rerank_vector10_top3 | True | 1.00 | 7.27 | 6.08 | 3 | 2430 | 68, 69, 70 | evaluator, optimizer, feedback, iterative, quality, loop |
| semantic_near_top1 | True | 1.00 | 7.27 | -2.11 | 5 | 2408 | 69, 68, 52, 58, 59 | evaluator, optimizer, feedback, iterative, quality, loop |
| semantic_near_top3 | True | 1.00 | 7.27 | -1.81 | 9 | 5540 | 69, 53, 71, 55, 59, 58, 68, 52, 70 | evaluator, optimizer, feedback, iterative, quality, loop |
| vector_reading_top1 | True | 1.00 | 7.27 | 6.08 | 3 | 2430 | 68, 69, 70 | evaluator, optimizer, feedback, iterative, quality, loop |
| vector_reading_top3 | True | 1.00 | 7.27 | -0.85 | 8 | 5054 | 68, 69, 70, 52, 53, 54, 71, 72 | evaluator, optimizer, feedback, iterative, quality, loop |
| vector_top3 | True | 0.67 | 6.26 | -0.87 | 3 | 2124 | 69, 53, 71 | evaluator, optimizer, feedback, quality |

### observability

Query: Why do agent systems need observability and audit trails?

| strategy | section_hit | term_recall | ce_max | ce_mean | chunks | chars | seqs | terms_found |
|---|---:|---:|---:|---:|---:|---:|---|---|
| combined_reading_semantic_top1 | True | 0.83 | 0.83 | -6.40 | 7 | 3708 | 22, 23, 24, 72, 0, 35, 34 | observable, explain, trace, decision, black |
| hybrid_reading_top1 | True | 0.83 | 0.23 | -4.35 | 3 | 2182 | 22, 23, 24 | observable, explain, trace, decision, black |
| hybrid_semantic_top1 | True | 0.83 | 0.83 | -6.35 | 6 | 2841 | 23, 72, 0, 35, 24, 34 | observable, explain, trace, decision, black |
| hybrid_top3 | True | 1.00 | 0.83 | -0.36 | 3 | 2277 | 23, 77, 34 | observable, explain, audit, trace, decision, black |
| rerank_vector10_top3 | True | 1.00 | 0.83 | 0.02 | 3 | 2590 | 34, 23, 53 | observable, explain, audit, trace, decision, black |
| semantic_near_top1 | True | 0.83 | 0.83 | -6.35 | 6 | 2841 | 23, 72, 0, 35, 24, 34 | observable, explain, trace, decision, black |
| semantic_near_top3 | True | 1.00 | 0.83 | -6.07 | 11 | 6720 | 23, 77, 34, 72, 0, 35, 24, 33, 78, 83, 82 | observable, explain, audit, trace, decision, black |
| vector_reading_top1 | True | 0.83 | 0.23 | -4.35 | 3 | 2182 | 22, 23, 24 | observable, explain, trace, decision, black |
| vector_reading_top3 | True | 1.00 | 0.83 | -5.18 | 9 | 6079 | 22, 23, 24, 76, 77, 78, 33, 34, 35 | observable, explain, audit, trace, decision, black |
| vector_top3 | True | 1.00 | 0.83 | -0.36 | 3 | 2277 | 23, 77, 34 | observable, explain, audit, trace, decision, black |

## Interpretation

Use this table defensively. A graph-expanded strategy wins only if it improves evidence coverage or relevance at an acceptable context cost. If it adds many chunks with lower `ce_mean`, it is exploration/visualization value, not proof of better answer quality.

For article claims, the honest framing is: semantic proximity edges make vector similarity traversable and explainable as a graph. They can improve recall/context assembly, but they do not automatically improve top-k ranking; reranking or budgeting is still needed to control noise.