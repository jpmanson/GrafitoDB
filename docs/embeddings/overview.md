# Embeddings Overview

GrafitoDB provides seamless integration with multiple embedding providers, allowing you to convert text into vector representations for semantic search, clustering, and similarity analysis.

## What are Embeddings?

Embeddings are dense vector representations of text (or other data) that capture semantic meaning. Similar texts have vectors that are close together in the embedding space, enabling:

- **Semantic search**: Find content with similar meaning
- **Clustering**: Group similar items automatically
- **Recommendation**: Suggest similar items
- **Classification**: Categorize content based on similarity

## Supported Providers

GrafitoDB includes built-in support for 12+ embedding providers:

### Cloud APIs
| Provider | Description | Best For |
|----------|-------------|----------|
| [OpenAI](./openai.md) | text-embedding-3-small, text-embedding-3-large | High-quality embeddings |
| [Hugging Face](./huggingface.md) | Inference API with thousands of models | Flexibility, open-source models |
| [Google GenAI](./cloud-providers.md#google-genai) | text-embedding-004 | Google Cloud integration |
| [AWS Bedrock](./cloud-providers.md#aws-bedrock) | Titan embeddings | AWS ecosystem |
| [Cohere](./cloud-providers.md#cohere) | Embed models | Enterprise use |
| [Jina](./cloud-providers.md#jina-ai) | Jina embeddings | Specialized models |
| [Mistral](./cloud-providers.md#mistral) | Mistral embeddings | European AI |
| [VoyageAI](./cloud-providers.md#voyage-ai) | Voyage embeddings | Domain-specific |
| [Together AI](./cloud-providers.md#together-ai) | Together embeddings | Open-source models |

### Local/On-Premise
| Provider | Description | Best For |
|----------|-------------|----------|
| [Ollama](./ollama.md) | Local LLM inference | Privacy, offline use |
| [Sentence Transformers](./huggingface.md#sentence-transformers-local) | Local embedding models | Cost-effective |
| [TensorFlow Hub](./cloud-providers.md#tensorflow-hub) | Google ML models | TensorFlow ecosystem |

## Quick Start

### 1. Create an Embedding Function

```python
from grafito import GrafitoDatabase
from grafito.embedding_functions import OpenAIEmbeddingFunction

# Initialize embedding function
embed_fn = OpenAIEmbeddingFunction(
    model="text-embedding-3-small",
    api_key_env_var="OPENAI_API_KEY"
)
```

### 2. Create a Vector Index

```python
db = GrafitoDatabase(":memory:")

# Create vector index with embedding function
db.create_vector_index(
    name="articles_vec",
    dim=1536,  # Match your embedding model
    embedding_function=embed_fn
)
```

### 3. Add Data with Embeddings

```python
# Create nodes
article = db.create_node(
    labels=["Article"],
    properties={"title": "Introduction to Graph Databases"}
)

# Generate embeddings automatically or manually
db.upsert_embedding(
    node_id=article.id,
    text="Introduction to Graph Databases",
    index="articles_vec"
)
```

### 4. Perform Semantic Search

```python
# Search for similar content
results = db.vector_search(
    index="articles_vec",
    query="graph database basics",
    k=5
)

for node, score in results:
    print(f"{node.properties['title']}: {score}")
```

## Embedding Function Base Interface

All embedding functions implement a common interface:

```python
class EmbeddingFunction:
    """Abstract embedding function interface."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for the given input texts."""
        ...

    def name(self) -> str:
        """Return the registry name for this embedding function."""
        ...

    def default_space(self) -> str:
        """Return default distance space (cosine, l2, ip)."""
        ...

    def supported_spaces(self) -> list[str]:
        """Return supported distance spaces."""
        ...

    def dimension(self) -> int | None:
        """Return embedding dimension if known."""
        ...
```

## Distance Spaces

Different embedding models work best with different distance metrics:

| Space | Description | Best For |
|-------|-------------|----------|
| `cosine` | Cosine similarity | Most text embeddings |
| `l2` | Euclidean distance | Image embeddings |
| `ip` | Inner product | OpenAI embeddings |

## Configuration Management

Embedding functions can be serialized for persistence:

```python
# Get configuration
config = embed_fn.get_config()
# {'model': 'text-embedding-3-small'}

# Recreate from configuration
from grafito.embedding_functions import create_embedding_function

embed_fn = create_embedding_function("openai", config)
```

## Choosing a Provider

### For Production Use
- **OpenAI**: Best quality embeddings, reliable API
- **Cohere**: Enterprise features, good multilingual support
- **AWS Bedrock**: If already using AWS infrastructure

### For Cost-Effective Solutions
- **Hugging Face Inference API**: Pay-per-use, many free models
- **Sentence Transformers**: Run locally, no API costs

### For Privacy/Offline
- **Ollama**: Run models locally on your hardware
- **Sentence Transformers**: Local inference

### For Experimentation
- **Ollama**: Easy to try different models
- **Hugging Face**: Largest selection of models

## Environment Variables

Most providers support environment variables for API keys:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Hugging Face
export HF_TOKEN="hf_..."

# Google GenAI
export GOOGLE_API_KEY="..."

# AWS Bedrock (uses boto3 credential chain)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

## Error Handling

```python
from grafito.embedding_functions import OpenAIEmbeddingFunction

try:
    embed_fn = OpenAIEmbeddingFunction()
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    embeddings = embed_fn(["text to embed"])
except Exception as e:
    print(f"API error: {e}")
```

## Next Steps

- [OpenAI Integration](./openai.md) - OpenAI's embedding models
- [Hugging Face](./huggingface.md) - Open-source models
- [Cloud Providers](./cloud-providers.md) - Google, AWS, Cohere, etc.
- [Ollama](./ollama.md) - Local embedding models
