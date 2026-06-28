"""Semantic search with Hugging Face embeddings and FAISS."""

from __future__ import annotations

import orjson
import os
import time

from grafito import GrafitoDatabase, DatabaseError
from grafito.embedding_functions import SentenceTransformerEmbeddingFunction
from grafito.indexers import FAISSIndexer


def main() -> None:
    start_total = time.perf_counter()
    step_start = time.perf_counter()
    print("Step 1/6: Initializing embedding function.")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    embedder_kwargs = {}
    if hf_token:
        embedder_kwargs["token"] = hf_token
    try:
        embedder = SentenceTransformerEmbeddingFunction(
            model_name="google/embeddinggemma-300m",
            **embedder_kwargs,
        )
    except Exception as exc:
        print(f"Failed to initialize embedding function: {exc}")
        return
    step_elapsed = time.perf_counter() - step_start
    print(f"Step 1/6 completed in {step_elapsed:.2f}s (model load).")

    step_start = time.perf_counter()
    print("Step 2/6: Creating in-memory Grafito database.")
    db = GrafitoDatabase(":memory:")
    docs = [
        ("Intro", "Grafito is a lightweight property graph database."),
        ("Vector", "Vector search uses embeddings and nearest neighbor queries."),
        ("Cypher", "Cypher queries match patterns in a property graph."),
    ]

    print("Step 2/6 completed in {:.2f}s.".format(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    print("Step 3/6: Inserting sample documents.")
    node_ids = []
    for title, text in docs:
        node = db.create_node(labels=["Doc"], properties={"title": title, "text": text})
        node_ids.append(node.id)

    print("Step 3/6 completed in {:.2f}s.".format(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    print("Step 4/6: Creating FAISS vector index.")
    indexer = FAISSIndexer(
        method="flat",
        options={"metric": "l2", "store_embeddings": True},
        embedding_function=embedder,
    )
    try:
        db.create_vector_index("docs_vec", indexer=indexer)
    except DatabaseError as exc:
        print(f"Failed to create FAISS index: {exc}")
        return
    print("Step 4/6 completed in {:.2f}s.".format(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    print("Step 5/6: Generating embeddings.")
    vectors = embedder([text for _, text in docs])
    print("Embeddings generated in {:.2f}s.".format(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    print("Step 5/6: Upserting embeddings into index.")
    for node_id, vector in zip(node_ids, vectors):
        db.upsert_embedding(node_id, vector, index="docs_vec")
    print("Upsert completed in {:.2f}s.".format(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    print("Step 6/6: Running semantic search via API and Cypher.")
    query = "property graph database"
    embed_start = time.perf_counter()
    query_vec = embedder([query])[0]
    print("Query embedding generated in {:.2f}s.".format(time.perf_counter() - embed_start))

    results = db.semantic_search(query_vec, k=2, index="docs_vec")
    print("Query:", query)
    for rank, result in enumerate(results, start=1):
        node = result["node"]
        print(f"{rank}. {node.properties.get('title')} ({result['score']:.4f})")

    cypher_start = time.perf_counter()
    vector_literal = orjson.dumps(query_vec).decode('utf-8')
    cypher = f"""
    CALL db.vector.search('docs_vec', {vector_literal}, 2)
    YIELD node, score
    RETURN node.title AS title, score
    """
    cypher_results = db.execute(cypher)
    print("Cypher results:")
    for row in cypher_results:
        print(f"- {row['title']} ({row['score']:.4f})")
    print("Cypher execution completed in {:.2f}s.".format(time.perf_counter() - cypher_start))
    print("Step 6/6 completed in {:.2f}s.".format(time.perf_counter() - step_start))
    print("Total time: {:.2f}s.".format(time.perf_counter() - start_total))


if __name__ == "__main__":
    main()
