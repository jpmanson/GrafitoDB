# Cloud Provider Embeddings

GrafitoDB supports multiple managed embedding APIs. Each provider has its own
SDK or HTTP requirements, but all integrate via `EmbeddingFunction`.

## AWS Bedrock

### Installation

```bash
pip install boto3
```

### Basic Usage

```python
from grafito.embedding_functions import AmazonBedrockEmbeddingFunction

embed_fn = AmazonBedrockEmbeddingFunction(
    model_name="amazon.titan-embed-text-v1",
    region_name="us-east-1"
)
```

### Configuration

You can pass AWS credentials directly or rely on the standard AWS credential
chain. Extra client arguments are forwarded to `boto3.Session(...).client(...)`.

```python
embed_fn = AmazonBedrockEmbeddingFunction(
    model_name="amazon.titan-embed-text-v1",
    profile_name="default",
    region_name="us-east-1",
    retries={"max_attempts": 5}
)
```

## Google GenAI

### Installation

```bash
pip install google-genai
```

### API Key

```bash
export GOOGLE_API_KEY="..."
```

### Basic Usage

```python
from grafito.embedding_functions import GoogleGenAIEmbeddingFunction

embed_fn = GoogleGenAIEmbeddingFunction(
    model_name="text-embedding-004"
)
```

### Vertex AI Configuration

```python
embed_fn = GoogleGenAIEmbeddingFunction(
    model_name="text-embedding-004",
    vertexai=True,
    project="my-gcp-project",
    location="us-central1"
)
```

## Cohere

### Installation

```bash
pip install httpx
```

### API Key

```bash
export COHERE_API_KEY="..."
```

### Basic Usage

```python
from grafito.embedding_functions import CohereEmbeddingFunction

embed_fn = CohereEmbeddingFunction(
    model="embed-english-v3.0",
    input_type="search_document"
)
```

### Notes

- `input_type` can be adjusted depending on whether you're embedding documents
  or queries.

## Jina AI

### Installation

```bash
pip install httpx
```

### API Key

```bash
export JINA_API_KEY="..."
```

### Basic Usage

```python
from grafito.embedding_functions import JinaEmbeddingFunction

embed_fn = JinaEmbeddingFunction(
    model_name="jina-embeddings-v2-base-en",
    task="retrieval.passage",
    normalized=True
)
```

### Query Embeddings

Jina supports a separate query embedding path:

```python
query_vecs = embed_fn.embed_query(["graph database performance"])  # type: ignore[attr-defined]
```

## Mistral

### Installation

```bash
pip install mistralai
```

### API Key

```bash
export MISTRAL_API_KEY="..."
```

### Basic Usage

```python
from grafito.embedding_functions import MistralEmbeddingFunction

embed_fn = MistralEmbeddingFunction(model="mistral-embed")
```

## Together AI

### Installation

```bash
pip install httpx
```

### API Key

```bash
export TOGETHER_API_KEY="..."
```

### Basic Usage

```python
from grafito.embedding_functions import TogetherAIEmbeddingFunction

embed_fn = TogetherAIEmbeddingFunction(
    model_name="togethercomputer/m2-bert-80M-8k-retrieval"
)
```

## Voyage AI

### Installation

```bash
pip install voyageai
```

### API Key

```bash
export VOYAGE_API_KEY="..."
```

### Basic Usage

```python
from grafito.embedding_functions import VoyageAIEmbeddingFunction

embed_fn = VoyageAIEmbeddingFunction(
    model_name="voyage-large-2",
    input_type="document",
    truncation=True
)
```

## TensorFlow Hub

TensorFlow Hub models can be used through a local embedding function.

### Installation

```bash
pip install tensorflow_hub
```

### Basic Usage

```python
from grafito.embedding_functions import TensorFlowHubEmbeddingFunction

embed_fn = TensorFlowHubEmbeddingFunction(
    model_url="https://tfhub.dev/google/universal-sentence-encoder/4"
)
```

## Using Any Provider with GrafitoDB

```python
from grafito import GrafitoDatabase

# embed_fn = ...

db = GrafitoDatabase(":memory:")
db.create_vector_index(
    name="content_vec",
    dim=768,
    embedding_function=embed_fn
)
```

## Next Steps

- [Overview](./overview.md) - Embedding concepts and workflows
- [OpenAI](./openai.md) - OpenAI embeddings
- [Hugging Face](./huggingface.md) - HF Inference API and local models
- [Ollama](./ollama.md) - Local embeddings with Ollama
