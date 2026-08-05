# Vector Search in Cypher

Perform semantic search using Cypher procedures.

## Vector Search Procedure

```cypher
CALL db.vector.search(
    index_name,      // Name of vector index
    query_vector,    // Vector to search for
    k,               // Number of results
    options          // Optional configuration
)
YIELD node, score
```

## Basic Vector Search

```python
from grafito.cypher import format_vector_literal

# Prepare query vector
query_vec = model.encode("python graph database").tolist()
vector_literal = format_vector_literal(query_vec, precision=8)

# Run vector search
cypher = f"""
    CALL db.vector.search('articles_vec', {vector_literal}, 10)
    YIELD node, score
    RETURN node.title, node.content, score
    ORDER BY score DESC
"""

results = db.execute(cypher)
```

## With Label Filters

```cypher
// Search only within specific labels
CALL db.vector.search(
    'articles_vec',
    [0.123, -0.456, ...],
    10,
    {labels: ['Article', 'Tutorial']}
)
YIELD node, score
RETURN node.title, score
```

## With Property Filters

```cypher
// Search with property constraints
CALL db.vector.search(
    'articles_vec',
    [0.123, -0.456, ...],
    10,
    {
        labels: ['Article'],
        properties: {published: true, language: 'en'}
    }
)
YIELD node, score
RETURN node.title, score
```

## With Reranking

```cypher
// Enable reranking with stored vectors
CALL db.vector.search(
    'articles_vec',
    [0.123, -0.456, ...],
    10,
    {rerank: true, candidate_multiplier: 3}
)
YIELD node, score
RETURN node.title, score
```

## Custom Reranker

```python
# Register custom reranker
def my_reranker(query_vector, candidates):
    # candidates: [{"id": int, "vector": [...], "score": float, "node": Node}, ...]
    re_ranked = []
    for c in candidates:
        # Custom scoring logic
        boost = 1.0
        if 'featured' in c['node'].labels:
            boost = 1.2
        re_ranked.append({
            'id': c['id'],
            'score': c['score'] * boost
        })
    return re_ranked

db.register_reranker('featured_boost', my_reranker)
```

```cypher
// Use in Cypher
CALL db.vector.search(
    'articles_vec',
    [0.123, -0.456, ...],
    10,
    {reranker: 'featured_boost'}
)
YIELD node, score
RETURN node.title, score
```

## Aliasing YIELD Outputs

`YIELD` accepts `AS` to rename an output. The alias is a real pattern variable,
so it can be matched on directly:

```cypher
CALL db.vector.search('articles_vec', $query_vec, 10) YIELD node AS article, score AS relevance
MATCH (article)<-[:AUTHORED]-(author:Person)
RETURN article.title, author.name, relevance
ORDER BY relevance DESC
```

Without `AS`, the outputs keep their procedure names (`node`, `score`) — which
collide if you call the procedure twice in one query. Aliasing is what makes the
two-endpoint pattern below possible.

## Semantic Predicates

`db.vector.search` answers "which nodes are closest?". Sometimes you already
have a node from a pattern match and need the opposite question: "how close is
*this* node?". Two functions score a node against a query:

```cypher
VECTOR_SCORE(node, query)                 // the raw similarity score, or NULL
SIMILAR(node, query)                      // score >= default threshold (0.5)
SIMILAR(node, query, 0.75)                // score >= 0.75
SIMILAR(node, query, {index: 'articles_vec', min_score: 0.75})
```

`query` is either a vector literal/parameter or a string, which is embedded with
the index's embedding function. Each distinct query string is embedded **once
per execution**, not once per candidate row.

Options:

| Option | Meaning | Default |
| --- | --- | --- |
| `index` | Vector index to score against | `'default'` |
| `min_score` | Threshold for `SIMILAR` | `0.5` |

Semantics worth knowing:

- A node with **no embedding** in the index scores `NULL` under `VECTOR_SCORE`
  and `false` under `SIMILAR` — an unindexed node is dropped by the filter
  rather than propagating unknown through it.
- A `NULL` node (from `OPTIONAL MATCH`) yields `NULL` for both.
- Scores follow the index metric and are always "higher is better": cosine in
  `[-1, 1]`, inner product unbounded, and negated squared distance (`<= 0`) for
  `l2`. Because the default threshold is calibrated for cosine, `SIMILAR` on an
  `l2` index **requires** an explicit `min_score` rather than silently matching
  nothing.

!!! warning "These are predicates, not seeks"
    `SIMILAR` and `VECTOR_SCORE` score the node they are handed; they never
    consult the ANN index for neighbours. Filtering a bare `MATCH (n)` with them
    scans and scores **every node in the database**:

    ```cypher
    // Slow: full scan, one score per node
    MATCH (n) WHERE SIMILAR(n, 'transformers') RETURN n
    ```

    Seed the pattern with `db.vector.search` — which does use the index — and
    use `SIMILAR` only to constrain an already-bounded set:

    ```cypher
    // Fast: ANN picks 20 candidates, SIMILAR filters the far end
    CALL db.vector.search('articles_vec', 'transformers', 20) YIELD node AS a
    MATCH (a)-[:CITES]->(b)
    WHERE SIMILAR(b, 'attention mechanism', 0.7)
    RETURN a.title, b.title
    ```

## Deep Graph Search

Because `YIELD` aliases bind as pattern variables, both ends of a path can be
seeded semantically — finding paths whose endpoints are *similar to* two
concepts, rather than searching for one concept and then the other:

```cypher
CALL db.vector.search('articles_vec', 'chatgpt', 10) YIELD node AS a
CALL db.vector.search('articles_vec', 'anthropic', 10) YIELD node AS b
MATCH p=(a)-[*1..3]->(b)
RETURN p
LIMIT 10
```

The second `CALL` runs per row from the first, so the pattern explores every
pairing of the two candidate sets. Keep both `k` values modest — the path search
is quadratic in them.

To score how strongly each endpoint matched, keep the scores:

```cypher
CALL db.vector.search('papers_vec', $topic_a, 10) YIELD node AS a, score AS score_a
CALL db.vector.search('papers_vec', $topic_b, 10) YIELD node AS b, score AS score_b
MATCH p=(a)-[:CITES*1..3]->(b)
RETURN a.title, b.title, length(p) AS hops, score_a + score_b AS strength
ORDER BY strength DESC, hops ASC
LIMIT 20
```

## Combining with Graph Patterns

```python
# Hybrid: vector search + graph traversal
query_vec = model.encode("machine learning tutorials").tolist()
vector_literal = format_vector_literal(query_vec)

cypher = f"""
    // Stage 1: Vector search for candidates
    CALL db.vector.search('articles_vec', {vector_literal}, 20)
    YIELD node, score
    
    // Stage 2: Expand to authors
    MATCH (node)<-[:AUTHORED]-(author:Person)
    
    // Stage 3: Get author's other content
    OPTIONAL MATCH (author)-[:AUTHORED]->(other:Article)
    WHERE other <> node
    
    RETURN 
        node.title as matched_article,
        score,
        author.name as author,
        collect(DISTINCT other.title)[0..3] as other_works
    ORDER BY score DESC
    LIMIT 10
"""

results = db.execute(cypher)
```

## With Path Expansion

```cypher
// Vector search + related content
CALL db.vector.search('articles_vec', $query_vec, 5)
YIELD node, score

// Find related articles through shared tags
MATCH (node)-[:TAGGED]->(tag)<-[:TAGGED]-(related)
WHERE related <> node

RETURN 
    node.title as main_result,
    score,
    collect(DISTINCT related.title)[0..5] as related_articles
ORDER BY score DESC
```

## Aggregating Vector Results

```cypher
// Cluster search results by category
CALL db.vector.search('articles_vec', $query_vec, 50)
YIELD node, score

WITH node.category as category, 
     collect({node: node, score: score}) as items,
     avg(score) as avg_score

RETURN 
    category,
    count(*) as count,
    avg_score,
    items[0..3] as top_items
ORDER BY avg_score DESC
```

## Dynamic Vector Queries

```python
# Build vector queries dynamically
def vector_search_with_context(db, query, user_id, k=10):
    # Get user's interests
    interests = db.execute("""
        MATCH (u:User {id: $user_id})-[:INTERESTED_IN]->(topic)
        RETURN collect(topic.name) as interests
    """, {'user_id': user_id})
    
    # Build enriched query
    enriched_query = query + ' ' + ' '.join(interests[0]['interests'])
    query_vec = model.encode(enriched_query).tolist()
    
    # Search with personalization boost
    return db.execute("""
        CALL db.vector.search('articles_vec', $vec, $k)
        YIELD node, score
        
        // Boost if matches user interests
        OPTIONAL MATCH (node)-[:ABOUT]->(topic)
        WHERE topic.name IN $interests
        
        WITH node, score, count(topic) as interest_matches
        ORDER BY score + (interest_matches * 0.1) DESC
        
        RETURN node.title, node.summary, score
        LIMIT $k
    """, {'vec': query_vec, 'k': k, 'interests': interests[0]['interests']})
```

## Error Handling

### Unknown Index

```
DatabaseError: Vector index 'unknown_idx' not found
```

Solution: Check index name with `SHOW INDEXES` or `db.list_vector_indexes()`.

### Dimension Mismatch

```
DatabaseError: Query vector dimension 768 does not match index dimension 384
```

Solution: Ensure query vector has same dimension as index.

### Unknown Reranker

```
CypherExecutionError: Unknown reranker 'invalid_name'
```

Solution: Register reranker first with `db.register_reranker()`.

## Best Practices

### Use Precision Parameter

```python
# Format vector for Cypher with appropriate precision
vector_literal = format_vector_literal(query_vec, precision=6)
# Higher precision = more accurate but longer query string
```

### Limit Initial Results

```cypher
// Get more candidates than needed for reranking
CALL db.vector.search('idx', $vec, 50)  // Get 50
YIELD node, score
// ... filter/process ...
RETURN ...
LIMIT 10  // Return top 10
```

### Combine with Property Indexes

```python
# Create property index for common filters
db.create_node_index('Article', 'published')
db.create_node_index('Article', 'language')

# Vector search will use these for pre-filtering
```

### Cache Query Embeddings

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_query_embedding(query):
    return model.encode(query).tolist()

# Reuse embeddings for repeated queries
vec = get_query_embedding("python graphql")
```
