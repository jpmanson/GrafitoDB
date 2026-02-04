# Semantic Search Examples

Practical examples of using semantic search with GrafitoDB in real-world scenarios.

## Academic Paper Search

Build a semantic search system for academic papers with citation networks.

```python
from grafito import GrafitoDatabase
from grafito.embedding_functions import SentenceTransformerEmbeddingFunction
from grafito.indexers import HNSWlibIndexer

# Initialize
db = GrafitoDatabase(':memory:')
embedder = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    normalize_embeddings=True
)

# Create vector index
indexer = HNSWlibIndexer(
    options={"metric": "cosine"},
    embedding_function=embedder
)
db.create_vector_index("papers_vec", indexer=indexer)

# Sample papers
papers = [
    {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture, the Transformer...",
        "year": 2017,
        "authors": ["Vaswani", "Shazeer", "Parmar"]
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce a new language representation model called BERT...",
        "year": 2018,
        "authors": ["Devlin", "Chang", "Lee", "Toutanova"]
    },
]

# Create paper nodes and embeddings
paper_nodes = {}
for paper in papers:
    node = db.create_node(
        labels=["Paper"],
        properties={
            "title": paper["title"],
            "abstract": paper["abstract"],
            "year": paper["year"]
        }
    )
    paper_nodes[paper["title"]] = node
    
    # Generate embedding from title + abstract
    text = f"{paper['title']}. {paper['abstract']}"
    vector = embedder([text])[0]
    db.upsert_embedding(node.id, vector, index="papers_vec")

# Create author nodes and relationships
for paper in papers:
    paper_node = paper_nodes[paper["title"]]
    for author_name in paper["authors"]:
        author = db.create_node(
            labels=["Author"],
            properties={"name": author_name}
        )
        db.create_relationship(paper_node.id, author.id, "AUTHORED_BY")

# Search semantically
results = db.semantic_search(
    "transformer architecture for NLP",
    k=10,
    index="papers_vec"
)

# Navigate citation network from results
for result in results:
    paper = result["node"]
    print(f"\nPaper: {paper.properties['title']} (score: {result['score']:.3f})")
    
    # Get authors
    authors = db.get_neighbors(paper.id, direction="outgoing", rel_type="AUTHORED_BY")
    author_names = [a.properties['name'] for a in authors]
    print(f"Authors: {', '.join(author_names)}")
    
    # Get citations (if we had them)
    # citations = db.get_neighbors(paper.id, direction="outgoing", rel_type="CITES")
```

## E-commerce Product Search

Semantic product search with related product recommendations.

```python
# Products with descriptions
products = [
    {
        "name": "UltraBook Pro 15",
        "description": "High-performance laptop with 16GB RAM, 512GB SSD, Intel i7 processor",
        "category": "Electronics",
        "price": 1299
    },
    {
        "name": "BaristaMaster Coffee Maker",
        "description": "Automatic drip coffee maker with programmable timer and thermal carafe",
        "category": "Kitchen",
        "price": 199
    },
    {
        "name": "ErgoChair Plus",
        "description": "Ergonomic office chair with lumbar support and adjustable armrests",
        "category": "Furniture",
        "price": 449
    },
]

# Create product nodes and embeddings
for product in products:
    node = db.create_node(
        labels=["Product"],
        properties=product
    )
    vector = embedder([product["description"]])[0]
    db.upsert_embedding(node.id, vector, index="products_vec")

# Natural language search
query = "machine for brewing coffee automatically"
results = db.semantic_search(
    query,
    k=5,
    index="products_vec",
    labels=["Product"]
)

for result in results:
    product = result["node"]
    print(f"{product.properties['name']}: ${product.properties['price']}")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Description: {product.properties['description'][:80]}...")
```

## RAG System with Graph Context

Build a Retrieval-Augmented Generation system with rich graph context.

```python
def rag_query(db, user_question: str, embedder, llm_complete, k: int = 5):
    """
    Answer question using graph-enhanced RAG.
    
    Args:
        db: GrafitoDB instance
        user_question: The user's question
        embedder: Embedding function
        llm_complete: Function to call LLM with prompt
        k: Number of documents to retrieve
    """
    # 1. Semantic search for relevant documents
    query_vector = embedder([user_question])[0]
    results = db.semantic_search(query_vector, k=k, index="docs_vec")
    
    # 2. Gather graph context
    context_parts = []
    for result in results:
        doc = result["node"]
        
        # Document content
        context_parts.append(f"Document: {doc.properties['text']}")
        
        # Related entities from graph
        related = db.get_neighbors(doc.id, rel_type="MENTIONS")
        if related:
            entities = [n.properties.get('name') for n in related]
            context_parts.append(f"Related entities: {', '.join(entities)}")
        
        # Source metadata
        sources = db.get_neighbors(doc.id, rel_type="FROM_SOURCE")
        if sources:
            source_names = [s.properties.get('name') for s in sources]
            context_parts.append(f"Sources: {', '.join(source_names)}")
    
    # 3. Build prompt with rich context
    context = "\n\n".join(context_parts)
    prompt = f"""Context from knowledge graph:
{context}

Question: {user_question}

Answer:"""
    
    # 4. Send to LLM
    return llm_complete(prompt)

# Usage
# response = rag_query(db, "What are the main benefits of graph databases?", embedder, call_llm)
```

## Healthcare: Symptom-Disease Matching

Match patient symptoms to diseases using semantic similarity.

```python
# Create disease nodes with symptom descriptions
diseases = [
    {
        "name": "Influenza",
        "symptoms": "fever, cough, sore throat, body aches, fatigue, chills",
        "severity": "moderate",
        "contagious": True
    },
    {
        "name": "COVID-19",
        "symptoms": "fever, dry cough, fatigue, loss of taste or smell, shortness of breath",
        "severity": "high",
        "contagious": True
    },
    {
        "name": "Common Cold",
        "symptoms": "runny nose, sneezing, mild cough, sore throat, congestion",
        "severity": "mild",
        "contagious": True
    },
]

# Create disease nodes and embeddings
for disease in diseases:
    node = db.create_node(
        labels=["Disease"],
        properties=disease
    )
    vector = embedder([disease["symptoms"]])[0]
    db.upsert_embedding(node.id, vector, index="diseases_vec")
    
    # Create treatment nodes
    treatments = get_treatments_for(disease["name"])  # Your function
    for treatment in treatments:
        treatment_node = db.create_node(
            labels=["Treatment"],
            properties={"name": treatment}
        )
        db.create_relationship(node.id, treatment_node.id, "TREATED_BY")

# Patient presents with symptoms
patient_symptoms = "I have a high temperature, dry cough, and can't taste food"
results = db.semantic_search(patient_symptoms, k=3, index="diseases_vec")

# Get treatment protocols from graph
print("Possible diagnoses:")
for i, result in enumerate(results, 1):
    disease = result["node"]
    print(f"\n{i}. {disease.properties['name']} (score: {result['score']:.3f})")
    print(f"   Severity: {disease.properties['severity']}")
    
    treatments = db.get_neighbors(disease.id, rel_type="TREATED_BY")
    if treatments:
        treatment_names = [t.properties['name'] for t in treatments]
        print(f"   Treatments: {', '.join(treatment_names)}")
```

## Document Management with Semantic + Graph

Enterprise document management combining semantic search with organizational structure.

```python
# Create department and document structure
departments = ["Engineering", "Sales", "HR", "Legal"]
dept_nodes = {}

for dept in departments:
    node = db.create_node(
        labels=["Department"],
        properties={"name": dept}
    )
    dept_nodes[dept] = node

# Create documents linked to departments
documents = [
    {
        "title": "Q4 Engineering Roadmap",
        "content": "Detailed plans for Q4 including microservices migration...",
        "department": "Engineering",
        "confidentiality": "internal"
    },
    {
        "title": "Sales Playbook 2024",
        "content": "Strategies for enterprise sales and customer retention...",
        "department": "Sales",
        "confidentiality": "confidential"
    },
]

for doc in documents:
    node = db.create_node(
        labels=["Document"],
        properties={
            "title": doc["title"],
            "content": doc["content"],
            "confidentiality": doc["confidentiality"]
        }
    )
    
    # Link to department
    db.create_relationship(node.id, dept_nodes[doc["department"]].id, "BELONGS_TO")
    
    # Create embedding
    vector = embedder([doc["content"]])[0]
    db.upsert_embedding(node.id, vector, index="documents_vec")

# Search with department filter
query = "microservices architecture migration"
results = db.semantic_search(
    query,
    k=10,
    index="documents_vec",
    labels=["Document"],
    properties={"confidentiality": "internal"}
)

# Navigate organizational context
for result in results:
    doc = result["node"]
    print(f"\nDocument: {doc.properties['title']}")
    print(f"Score: {result['score']:.3f}")
    
    # Get department
    depts = db.get_neighbors(doc.id, rel_type="BELONGS_TO")
    for dept in depts:
        print(f"Department: {dept.properties['name']}")
        
        # Find related documents in same department
        related = db.get_neighbors(dept.id, direction="incoming", rel_type="BELONGS_TO")
        related_docs = [r for r in related if r.id != doc.id][:3]
        if related_docs:
            print("Related documents in same dept:")
            for r in related_docs:
                print(f"  - {r.properties['title']}")
```

## Social Network Content Discovery

Find relevant content in social networks using semantic search.

```python
# Create users and posts
users = [
    {"username": "alice_dev", "interests": ["python", "machine learning"]},
    {"username": "bob_data", "interests": ["data science", "sql"]},
]

user_nodes = {}
for user in users:
    node = db.create_node(
        labels=["User"],
        properties=user
    )
    user_nodes[user["username"]] = node

# Create posts
posts = [
    {
        "author": "alice_dev",
        "content": "Just discovered this amazing graph database called GrafitoDB!",
        "tags": ["databases", "graphs"]
    },
    {
        "author": "bob_data",
        "content": "Working on a new ML pipeline for recommendation systems...",
        "tags": ["machine learning", "recommendations"]
    },
]

for post in posts:
    node = db.create_node(
        labels=["Post"],
        properties={
            "content": post["content"],
            "tags": post["tags"]
        }
    )
    
    # Link to author
    db.create_relationship(user_nodes[post["author"]].id, node.id, "AUTHORED")
    
    # Create embedding
    vector = embedder([post["content"]])[0]
    db.upsert_embedding(node.id, vector, index="posts_vec")

# Search for content
query = "graph databases for recommendations"
results = db.semantic_search(query, k=10, index="posts_vec")

# Get social context
for result in results:
    post = result["node"]
    print(f"\nPost: {post.properties['content'][:60]}...")
    print(f"Score: {result['score']:.3f}")
    
    # Get author
    authors = db.get_neighbors(post.id, direction="incoming", rel_type="AUTHORED")
    for author in authors:
        print(f"Author: @{author.properties['username']}")
        
        # Get author's other posts
        other_posts = db.get_neighbors(author.id, rel_type="AUTHORED")
        other_posts = [p for p in other_posts if p.id != post.id][:2]
        if other_posts:
            print("Other posts by this user:")
            for p in other_posts:
                print(f"  - {p.properties['content'][:40]}...")
```

## Code Search with Semantic Understanding

Search code repositories semantically.

```python
# Code snippets
snippets = [
    {
        "filename": "database.py",
        "language": "python",
        "code": "def create_node(self, labels, properties):\n    '''Create a new node in the graph'''\n    ...",
        "description": "Create a new node with labels and properties"
    },
    {
        "filename": "query.py",
        "language": "python", 
        "code": "def semantic_search(self, vector, k=10):\n    '''Find similar vectors'''\n    ...",
        "description": "Perform semantic search using vector similarity"
    },
]

for snippet in snippets:
    node = db.create_node(
        labels=["CodeSnippet", snippet["language"].capitalize()],
        properties=snippet
    )
    
    # Embed description + code
    text = f"{snippet['description']}. {snippet['code']}"
    vector = embedder([text])[0]
    db.upsert_embedding(node.id, vector, index="code_vec")

# Search code semantically
query = "how to create nodes in the graph"
results = db.semantic_search(query, k=5, index="code_vec")

for result in results:
    snippet = result["node"]
    print(f"\nFile: {snippet.properties['filename']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Description: {snippet.properties['description']}")
    print(f"Code:\n{snippet.properties['code'][:200]}...")
```

## Performance Tips for Production

When deploying semantic search at scale:

```python
# 1. Use batch operations for indexing
batch_size = 1000
vectors = []
node_ids = []

for i, item in enumerate(items):
    vector = embedder([item["text"]])[0]
    vectors.append(vector)
    node_ids.append(item["id"])
    
    if len(vectors) >= batch_size:
        db.upsert_embeddings(node_ids, vectors, index="my_index")
        vectors = []
        node_ids = []

# Insert remaining
if vectors:
    db.upsert_embeddings(node_ids, vectors, index="my_index")

# 2. Enable persistence for faster restarts
db.create_vector_index(
    name="production_index",
    dim=384,
    backend="faiss",
    method="hnsw",
    options={
        "metric": "cosine",
        "index_path": "/data/vector_index.faiss"
    }
)

# 3. Use reranking for critical applications
results = db.semantic_search(
    query,
    k=10,
    index="production_index",
    rerank=True,
    candidate_multiplier=5  # Fetch 50, return top 10
)

# 4. Cache query embeddings
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(query_text):
    return tuple(embedder([query_text])[0])

# 5. Monitor performance
import time

start = time.time()
results = db.semantic_search(query, k=10, index="my_index")
latency_ms = (time.time() - start) * 1000

print(f"Search latency: {latency_ms:.2f}ms")
```
