# Semantic Search Overview

GrafitoDB combines the structural power of **property graph databases** with the semantic understanding of **vector embeddings**, enabling a new class of intelligent graph applications.

## Why Combine Semantic Search with Knowledge Graphs?

Traditional knowledge graphs excel at **structural reasoning** (finding paths, relationships, patterns), while semantic search excels at **understanding meaning**. Together, they create a powerful synergy:

### 1. Semantic Discovery with Structural Navigation

Find nodes by meaning, then traverse their relationships:

```python
# Find documents about "machine learning" semantically
results = db.semantic_search("machine learning techniques", k=5)

# Then navigate to related entities
for result in results:
    doc_node = result["node"]
    # Find authors of these documents
    authors = db.get_neighbors(doc_node.id, direction="outgoing", rel_type="AUTHORED_BY")
    # Find cited papers
    citations = db.get_neighbors(doc_node.id, direction="outgoing", rel_type="CITES")
```

### 2. Context-Aware Retrieval

Use graph structure to inform semantic search:

```python
# Find papers semantically similar to "neural networks"
papers = db.semantic_search("neural networks", k=10, filter_labels=["Paper"])

# For each paper, get its citation network
for paper_result in papers:
    paper = paper_result["node"]

    # Get papers this paper cites (outgoing edges)
    references = db.get_neighbors(paper.id, direction="outgoing", rel_type="CITES")

    # Get papers that cite this paper (incoming edges)
    cited_by = db.get_neighbors(paper.id, direction="incoming", rel_type="CITES")

    # Find common authors
    authors = db.get_neighbors(paper.id, rel_type="AUTHORED_BY")
```

### 3. Multi-Hop Semantic Queries

Combine semantic similarity with graph traversal:

```cypher
// Find papers semantically similar to a query
CALL db.vector.search('papers_vec', $query_vector, 5)
YIELD node AS paper, score

// Then find co-authors of those papers
MATCH (paper)-[:AUTHORED_BY]->(author)-[:AUTHORED_BY]->(other_paper)
WHERE other_paper <> paper
RETURN paper.title, author.name, collect(other_paper.title) AS coauthor_papers
```

### 4. Hybrid Ranking

Combine semantic similarity with graph metrics (PageRank, centrality, citation count):

```python
results = db.semantic_search("deep learning", k=20)

for result in results:
    node = result["node"]
    semantic_score = result["score"]

    # Calculate graph-based importance
    citation_count = len(db.get_neighbors(node.id, direction="incoming", rel_type="CITES"))

    # Hybrid score
    hybrid_score = 0.7 * semantic_score + 0.3 * (citation_count / 100)

    result["hybrid_score"] = hybrid_score
```

### 5. Question Answering with Graph Context

Build RAG systems with rich relationship context:

```python
# User question: "Who are the leading researchers in reinforcement learning?"
query_vector = embedder(["reinforcement learning research"])[0]

# Find relevant papers semantically
papers = db.semantic_search(query_vector, k=10, filter_labels=["Paper"])

# Get authors and their collaboration networks
for paper in papers:
    authors = db.get_neighbors(paper["node"].id, rel_type="AUTHORED_BY")
    for author in authors:
        # Get author's other papers
        other_papers = db.get_neighbors(author.id, direction="incoming", rel_type="AUTHORED_BY")
        # Get collaboration network
        collaborators = db.execute("""
            MATCH (a:Author {id: $author_id})-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]-(coauthor:Author)
            WHERE coauthor <> a
            RETURN coauthor, count(p) AS num_collaborations
            ORDER BY num_collaborations DESC
        """, {"author_id": author.id})
```

## Architecture

### How It Works

GrafitoDB's semantic search implementation consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                       GrafitoDB Database                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐     ┌───────────┐  │
│  │   Nodes      │      │  Embeddings  │     │  Vector   │  │
│  │  (SQLite)    │◄────►│   (SQLite)   │────►│  Index    │  │
│  │              │      │              │     │ (Memory)  │  │
│  │ id | props   │      │ node_id |    │     │           │  │
│  │ 1  | {...}   │      │ vector       │     │  FAISS/   │  │
│  │ 2  | {...}   │      │              │     │  HNSW/    │  │
│  └──────────────┘      └──────────────┘     │  Annoy    │  │
│                                              └───────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Embedding Function                        │  │
│  │  (OpenAI / Cohere / HuggingFace / Ollama / etc.)    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Node Creation**: Create nodes with properties (text, metadata)
2. **Embedding Generation**: Convert text properties to vectors using embedding functions
3. **Index Insertion**: Store vectors in specialized vector indexes for fast similarity search
4. **Query**: Convert query text to vector, search index for nearest neighbors
5. **Retrieval**: Return nodes with their properties and similarity scores
6. **Graph Traversal**: Navigate relationships from retrieved nodes

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **Multiple Embedding Providers** | OpenAI, Cohere, HuggingFace, Ollama, and more |
| **Multiple ANN Backends** | FAISS, HNSWlib, Annoy, LEANN, USearch, Voyager |
| **Similarity Metrics** | Cosine similarity, L2 distance, and inner product |
| **Property Filtering** | Combine semantic search with graph structure filters |
| **Reranking** | Improve precision with exact reranking |
| **Cypher Integration** | Native `CALL db.vector.search()` procedure |
| **Persistent Storage** | Save and load vector indexes |

## Real-World Applications

| Domain | Use Case |
|--------|----------|
| **Academic Research** | Semantic paper discovery + citation networks |
| **E-commerce** | Product similarity + purchase patterns and user behavior graphs |
| **Healthcare** | Symptom matching + patient history and treatment pathways |
| **Enterprise Knowledge Management** | Document similarity + organizational hierarchies |
| **Recommendation Systems** | Content similarity + social graphs and interaction patterns |
| **Fraud Detection** | Anomaly detection + transaction networks |
| **Chatbots/Assistants** | Semantic understanding + knowledge graphs for context |

## Next Steps

- Learn about [Vector Search](vector.md) - Core functionality
- Explore [ANN Backends](ann-backends.md) - Choose the right backend
- Understand [Hybrid Search](hybrid.md) - Combine text and vector search
- Set up [Embeddings](../embeddings/overview.md) - Configure embedding providers
