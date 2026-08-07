# Hybrid Search Workflows

Combining full-text search (BM25) with vector search for better results.

!!! tip "Built-in helper"
    For passage graphs created with [Document Chunking](document-chunking.md),
    use `DocumentIngestor.hybrid_search()` (RRF fusion, ownership/generation
    filters). Manual recipes below still apply to ad-hoc indexes.

!!! warning "Recipes below pass the raw query to `text_search`"
    That method takes [FTS5 syntax](fulltext.md#query-syntax), so a user-typed
    question raises `DatabaseError` on its punctuation. `db.hybrid_search()`
    handles this for you by retrying as literal terms; in your own recipe,
    quote the tokens (`' '.join(f'"{t}"' for t in re.findall(r'\w+', query))`)
    before calling `text_search`.

## Why Hybrid Search?

| Search Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **Full-Text (BM25)** | Exact keyword matching, fast | No semantic understanding |
| **Vector** | Semantic similarity, conceptual | Misses exact keyword matches |
| **Hybrid** | Best of both worlds | More complex, slightly slower |

## Basic Hybrid Search

### Reciprocal Rank Fusion (RRF)

```python
def hybrid_search_rrf(db, query, query_vec, k=10, rank_k=60):
    """
    Combine BM25 and vector results using RRF.
    rank_k: constant for RRF formula (typically 60)
    """
    # Get keyword results
    text_results = db.text_search(query, k=50)
    text_ids = {r.node.id: 1/(rank_k + i) for i, r in enumerate(text_results)}

    # Get vector results
    vec_results = db.semantic_search(query_vec, k=50, index='articles_vec')
    vec_ids = {r.node.id: 1/(rank_k + i) for i, r in enumerate(vec_results)}

    # Combine scores
    all_ids = set(text_ids.keys()) | set(vec_ids.keys())
    scores = {id: text_ids.get(id, 0) + vec_ids.get(id, 0) for id in all_ids}

    # Get top k
    top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]

    # Fetch nodes
    results = []
    for node_id in top_ids:
        node = db.get_node(node_id)
        results.append({
            'node': node,
            'score': scores[node_id],
            'text_rank': text_ids.get(node_id),
            'vector_rank': vec_ids.get(node_id)
        })

    return results
```

### Weighted Combination

```python
def hybrid_search_weighted(db, query, query_vec, text_weight=0.3, vec_weight=0.7, k=10):
    """Combine with configurable weights."""
    # Normalize scores to [0, 1]
    text_results = db.text_search(query, k=50)
    max_text_score = max((r.score for r in text_results), default=1)
    text_scores = {r.node.id: (r.score / max_text_score) * text_weight
                   for r in text_results}

    vec_results = db.semantic_search(query_vec, k=50, index='articles_vec')
    max_vec_score = max((r.score for r in vec_results), default=1)
    vec_scores = {r.node.id: (r.score / max_vec_score) * vec_weight
                  for r in vec_results}

    # Combine
    all_ids = set(text_scores.keys()) | set(vec_scores.keys())
    combined = {id: text_scores.get(id, 0) + vec_scores.get(id, 0)
                for id in all_ids}

    # Return top k
    top_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:k]
    return [{'node': db.get_node(id), 'score': combined[id]} for id in top_ids]
```

## Advanced Hybrid Patterns

### Cascade Search

```python
def cascade_search(db, query, query_vec, k=10):
    """Use fast text search first, then rerank with vectors."""
    # Stage 1: Fast keyword search to get candidates
    candidates = db.text_search(query, k=100)
    candidate_ids = [r.node.id for r in candidates]

    # Stage 2: Rerank with vectors
    results = []
    for node_id in candidate_ids:
        # Get vector for this node (must be stored)
        vec_result = db.execute("""
            SELECT vector FROM vector_entries
            WHERE node_id = $id AND index_name = 'articles_vec'
        """, {'id': node_id})

        if vec_result:
            node_vec = vec_result[0]['vector']
            # Calculate similarity
            score = cosine_similarity(query_vec, node_vec)
            results.append({'node_id': node_id, 'score': score})

    # Return top k
    results.sort(key=lambda x: x['score'], reverse=True)
    return [db.get_node(r['node_id']) for r in results[:k]]
```

### Filtered Hybrid

```python
def filtered_hybrid_search(db, query, query_vec, filters, k=10):
    """Apply filters before hybrid search."""
    # Pre-filter with properties
    base_results = db.text_search(
        query,
        k=100,
        labels=['Article'],
        properties=filters  # e.g., {'category': 'tech', 'published': True}
    )

    # Get filtered IDs
    filtered_ids = {r.node.id for r in base_results}

    # Vector search with same filters
    vec_results = db.semantic_search(
        query_vec,
        k=100,
        index='articles_vec',
        labels=['Article'],
        properties=filters
    )

    # Combine only filtered results
    # ... RRF or weighted combination ...
```

## Reranking Strategies

### Cross-Encoder Reranking

```python
def rerank_with_cross_encoder(db, query, initial_results, k=10):
    """Use cross-encoder for final reranking."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Prepare pairs
    pairs = [(query, r['node'].properties['content']) for r in initial_results]

    # Score
    scores = model.predict(pairs)

    # Rerank
    for r, score in zip(initial_results, scores):
        r['rerank_score'] = score

    initial_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    return initial_results[:k]
```

### Metadata Boosting

```python
def boost_by_metadata(results, boost_rules):
    """Boost scores based on metadata."""
    for r in results:
        boost = 1.0
        node = r['node']

        # Boost recent content
        if 'created_at' in node.properties:
            age_days = (now() - node.properties['created_at']).days
            if age_days < 7:
                boost *= 1.2

        # Boost by popularity
        if 'views' in node.properties:
            if node.properties['views'] > 1000:
                boost *= 1.1

        # Boost by author reputation
        if 'author_verified' in node.properties:
            boost *= 1.15

        r['score'] *= boost

    return sorted(results, key=lambda x: x['score'], reverse=True)
```

## Complete Example

```python
from grafito import GrafitoDatabase
from sentence_transformers import SentenceTransformer

class HybridSearcher:
    def __init__(self, db_path):
        self.db = GrafitoDatabase(db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def search(self, query, k=10, method='rrf'):
        # Encode query
        query_vec = self.model.encode(query).tolist()

        # Get results from both methods
        text_results = self.db.text_search(query, k=50)
        vec_results = self.db.semantic_search(
            query_vec, k=50, index='articles_vec'
        )

        if method == 'rrf':
            return self._rrf_merge(text_results, vec_results, k)
        elif method == 'weighted':
            return self._weighted_merge(text_results, vec_results, k)

    def _rrf_merge(self, text_results, vec_results, k, rank_k=60):
        """Reciprocal Rank Fusion."""
        scores = {}

        for i, r in enumerate(text_results):
            scores[r.node.id] = scores.get(r.node.id, 0) + 1/(rank_k + i)

        for i, r in enumerate(vec_results):
            scores[r.node.id] = scores.get(r.node.id, 0) + 1/(rank_k + i)

        top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]

        return [{
            'node': self.db.get_node(id),
            'rrf_score': scores[id]
        } for id in top_ids]

    def _weighted_merge(self, text_results, vec_results, k):
        """Weighted score combination."""
        text_weight, vec_weight = 0.3, 0.7

        # Normalize
        max_text = max((r.score for r in text_results), default=1)
        max_vec = max((r.score for r in vec_results), default=1)

        scores = {}
        for r in text_results:
            scores[r.node.id] = (r.score / max_text) * text_weight
        for r in vec_results:
            scores[r.node.id] = scores.get(r.node.id, 0) + (r.score / max_vec) * vec_weight

        top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]

        return [{
            'node': self.db.get_node(id),
            'score': scores[id]
        } for id in top_ids]

# Usage
searcher = HybridSearcher('mydb.db')
results = searcher.search('python graph database', k=10, method='rrf')

for r in results:
    print(f"{r['node'].properties['title']}: {r.get('rrf_score', r.get('score')):.3f}")
```

## Evaluation

### Measuring Quality

```python
def evaluate_search(db, test_queries, ground_truth):
    """
    Evaluate search quality.
    test_queries: list of query strings
    ground_truth: dict mapping query to list of relevant doc IDs
    """
    from sklearn.metrics import ndcg_score

    results = {}
    for query in test_queries:
        # Get search results
        search_results = hybrid_search(db, query, model.encode(query))
        retrieved_ids = [r['node'].id for r in search_results]

        # Calculate metrics
        relevant = ground_truth[query]
        precision = len(set(retrieved_ids) & set(relevant)) / len(retrieved_ids)
        recall = len(set(retrieved_ids) & set(relevant)) / len(relevant)

        results[query] = {
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall)
        }

    return results
```

## Best Practices

1. **Tune Weights**: Adjust text/vector weights based on your data
2. **Candidate Pool**: Use larger k for initial retrieval (50-100)
3. **Reranking**: Apply cross-encoders or metadata boosting for final ranking
4. **Caching**: Cache query embeddings for repeated queries
5. **A/B Testing**: Compare hybrid vs single-method search

## When to Use Hybrid

| Scenario | Recommendation |
|----------|---------------|
| Technical documentation | ✅ Hybrid (keywords matter) |
| Semantic Q&A | ✅ Hybrid (concepts + keywords) |
| E-commerce search | ✅ Hybrid (product names + intent) |
| Pure semantic similarity | ❌ Vector only |
| Log analysis | ❌ Text only |
