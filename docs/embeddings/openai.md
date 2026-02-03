# OpenAI Embeddings

GrafitoDB integrates with OpenAI's embedding API to generate high-quality vector representations of text.

## Overview

OpenAI provides state-of-the-art embedding models that are:
- **High quality**: Trained on diverse internet text
- **Multilingual**: Support for 100+ languages
- **Cost-effective**: Competitive pricing per token
- **Well-documented**: Extensive community support

## Supported Models

| Model | Dimensions | Context Window | Best For |
|-------|-----------|----------------|----------|
| `text-embedding-3-small` | 1536 | 8192 | General use, cost-effective |
| `text-embedding-3-large` | 3072 | 8192 | Maximum quality |
| `text-embedding-ada-002` | 1536 | 8192 | Legacy model |

## Setup

### Installation

```bash
pip install grafito[openai]
# or
pip install httpx
```

### API Key Configuration

```bash
export OPENAI_API_KEY="sk-..."
```

Or provide explicitly:

```python
from grafito.embedding_functions import OpenAIEmbeddingFunction

embed_fn = OpenAIEmbeddingFunction(
    api_key="sk-..."
)
```

## Basic Usage

### Initialize Embedding Function

```python
from grafito import GrafitoDatabase
from grafito.embedding_functions import OpenAIEmbeddingFunction

# Using default model (text-embedding-3-small)
embed_fn = OpenAIEmbeddingFunction()

# Or specify a model
embed_fn = OpenAIEmbeddingFunction(
    model="text-embedding-3-large"
)
```

### Create Vector Index

```python
db = GrafitoDatabase(":memory:")

# Create index with OpenAI embedding function
db.create_vector_index(
    name="documents_vec",
    dim=1536,  # text-embedding-3-small
    embedding_function=embed_fn
)
```

### Generate Embeddings

```python
# Create a document
doc = db.create_node(
    labels=["Document"],
    properties={"title": "Getting Started with Graph Databases"}
)

# Generate embedding from text
db.upsert_embedding(
    node_id=doc.id,
    text="Getting Started with Graph Databases",
    index="documents_vec"
)
```

## Advanced Configuration

### Custom Environment Variable

```python
# Use a custom environment variable for the API key
embed_fn = OpenAIEmbeddingFunction(
    api_key_env_var="MY_OPENAI_KEY"
)
```

Then set:
```bash
export MY_OPENAI_KEY="sk-..."
```

### Custom Base URL

For using a proxy or Azure OpenAI:

```python
embed_fn = OpenAIEmbeddingFunction(
    model="text-embedding-3-small",
    base_url="https://your-proxy.com/v1/embeddings"
)
```

## Complete Example

```python
from grafito import GrafitoDatabase
from grafito.embedding_functions import OpenAIEmbeddingFunction

# Initialize embedding function
embed_fn = OpenAIEmbeddingFunction(
    model="text-embedding-3-small"
)

# Create database and index
db = GrafitoDatabase(":memory:")
db.create_vector_index(
    name="articles_vec",
    dim=1536,
    embedding_function=embed_fn,
    options={"store_embeddings": True}
)

# Add documents with embeddings
articles = [
    "Introduction to Python programming",
    "Graph databases vs relational databases",
    "Machine learning fundamentals",
    "Natural language processing techniques",
    "Data structures and algorithms"
]

for i, text in enumerate(articles):
    node = db.create_node(
        labels=["Article"],
        properties={"id": i, "content": text}
    )
    db.upsert_embedding(
        node_id=node.id,
        text=text,
        index="articles_vec"
    )

# Perform semantic search
results = db.execute("""
    CALL db.vector.search('articles_vec', 'Python coding basics', 3)
    YIELD node, score
    RETURN node.content AS content, score
""")

for result in results:
    print(f"{result['score']:.3f}: {result['content']}")
```

## Distance Metrics

OpenAI embeddings work well with multiple distance metrics:

```python
# Cosine similarity (default)
embed_fn.default_space()  # Returns: "cosine"

# All supported spaces
embed_fn.supported_spaces()  # Returns: ["cosine", "l2", "ip"]
```

## Error Handling

```python
from grafito.embedding_functions import OpenAIEmbeddingFunction

try:
    embed_fn = OpenAIEmbeddingFunction()
except ValueError as e:
    print(f"Missing API key: {e}")

try:
    embeddings = embed_fn(["text to embed"])
except ValueError as e:
    if "API error" in str(e):
        print(f"OpenAI API error: {e}")
```

## Pricing Considerations

OpenAI charges per token:
- text-embedding-3-small: ~$0.02 per 1M tokens
- text-embedding-3-large: ~$0.13 per 1M tokens

Tips to reduce costs:
1. Batch multiple texts in a single call
2. Cache embeddings for frequently used texts
3. Use text-embedding-3-small for most use cases

## Comparison with Other Providers

| Feature | OpenAI | Cohere | Hugging Face |
|---------|--------|--------|--------------|
| Quality | Excellent | Excellent | Good |
| Speed | Fast | Fast | Varies |
| Cost | Moderate | Moderate | Low/Free |
| Multilingual | Yes | Yes | Model-dependent |
| Custom models | No | No | Yes |

## Next Steps

- [Hugging Face](./huggingface.md) - Open-source alternatives
- [Cloud Providers](./cloud-providers.md) - Other cloud options
- [Ollama](./ollama.md) - Local embedding models
