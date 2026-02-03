# Hugging Face Embeddings

GrafitoDB supports Hugging Face embeddings through two paths:

- **Hugging Face Inference API**: hosted models via HTTP.
- **Sentence Transformers (Local)**: run models locally with `sentence-transformers`.

Both integrate with GrafitoDB's vector indexes in the same way.

## Hugging Face Inference API

### Installation

```bash
pip install httpx
```

### API Token

Grafito will read the API token from any of the following environment variables (in order):

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- `HUGGINGFACEHUB_API_TOKEN`
- `HUGGINGFACE_API_KEY`

```bash
export HF_TOKEN="hf_..."
```

### Basic Usage

```python
from grafito import GrafitoDatabase
from grafito.embedding_functions import HuggingFaceEmbeddingFunction

embed_fn = HuggingFaceEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = GrafitoDatabase(":memory:")
db.create_vector_index(
    name="docs_vec",
    dim=384,
    embedding_function=embed_fn
)

node = db.create_node(labels=["Doc"], properties={"text": "Graph search"})
db.upsert_embedding(node_id=node.id, text="Graph search", index="docs_vec")
```

### Configuration

```python
embed_fn = HuggingFaceEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_key_env_var="MY_HF_TOKEN"
)
```

## Sentence Transformers (Local)

Use local inference with the `sentence_transformers` library.

### Installation

```bash
pip install sentence_transformers
```

### Basic Usage

```python
from grafito import GrafitoDatabase
from grafito.embedding_functions import SentenceTransformerEmbeddingFunction

embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=False
)

db = GrafitoDatabase(":memory:")
db.create_vector_index(
    name="docs_vec",
    dim=384,
    embedding_function=embed_fn
)
```

### Advanced Configuration

`SentenceTransformerEmbeddingFunction` accepts extra keyword arguments passed to
`SentenceTransformer(...)` (only primitive JSON-like types are allowed):

```python
embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
    trust_remote_code=False
)
```

### Notes

- The embedding dimension is derived from the model if available.
- `normalize_embeddings=True` is useful when you rely on cosine similarity.

## Next Steps

- [Cloud Providers](./cloud-providers.md) - Managed embedding APIs
- [Ollama](./ollama.md) - Local embeddings via Ollama
- [Overview](./overview.md) - General embedding concepts
