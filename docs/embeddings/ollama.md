# Ollama Embeddings (Local)

GrafitoDB can generate embeddings locally using the Ollama embeddings API.
This is ideal for offline or privacy-sensitive workloads.

## Requirements

- Ollama installed and running on your machine
- An embeddings-capable model pulled locally (for example: `nomic-embed-text`)

## Installation

```bash
pip install httpx
```

## Basic Usage

```python
from grafito import GrafitoDatabase
from grafito.embedding_functions import OllamaEmbeddingFunction

embed_fn = OllamaEmbeddingFunction(
    model="nomic-embed-text"
)

db = GrafitoDatabase(":memory:")
db.create_vector_index(
    name="docs_vec",
    dim=768,
    embedding_function=embed_fn
)

node = db.create_node(labels=["Doc"], properties={"text": "Local embeddings"})
db.upsert_embedding(node_id=node.id, text="Local embeddings", index="docs_vec")
```

## Custom Host / Remote Ollama

By default, Grafito uses `http://localhost:11434` or the `OLLAMA_HOST` environment
variable if set.

```bash
export OLLAMA_HOST="http://localhost:11434"
```

```python
embed_fn = OllamaEmbeddingFunction(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)
```

## Notes

- The embedding dimension depends on the model you choose.
- Ollama generates embeddings one input at a time; batch large datasets carefully.

## Next Steps

- [Overview](./overview.md) - Embedding concepts and workflows
- [Hugging Face](./huggingface.md) - Local Sentence Transformers
- [Cloud Providers](./cloud-providers.md) - Managed embedding APIs
