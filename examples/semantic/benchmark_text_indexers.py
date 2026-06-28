"""
Benchmark script for comparing SQLiteFTSIndex and BM25SIndex performance.
"""

import time
import random
import string
import statistics
import sqlite3
import os
from tempfile import TemporaryDirectory
from typing import List, Tuple, Type

from grafito.text_index.sqlite_fts import SQLiteFTSIndex
try:
    from grafito.text_index.bm25s import BM25SIndex
    HAS_BM25S = True
except ImportError:
    HAS_BM25S = False
    print("Warning: bm25s not installed, skipping BM25SIndex benchmark.")

def generate_text(num_words: int) -> str:
    """Generate a random sentence with num_words words."""
    words = []
    for _ in range(num_words):
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
    return ' '.join(words)

def generate_dataset(num_docs: int, avg_words_per_doc: int) -> List[Tuple[int, str]]:
    """Generate a synthetic dataset of (id, text) tuples."""
    print(f"Generating {num_docs} documents...")
    dataset = []
    for i in range(num_docs):
        text = generate_text(avg_words_per_doc)
        dataset.append((i, text))
    return dataset

def benchmark_indexer(name: str, indexer, dataset: List[Tuple[int, str]], queries: List[str]):
    """Benchmark a single indexer."""
    print(f"\n--- Benchmarking {name} ---")
    
    ids, contents = zip(*dataset)
    ids = list(ids)
    contents = list(contents)
    
    # Benchmark Indexing
    start_time = time.time()
    indexer.add(ids, contents)
    end_time = time.time()
    indexing_time = end_time - start_time
    print(f"Indexing Time: {indexing_time:.4f} seconds ({len(dataset)} docs)")
    
    # Benchmark Search
    query_times = []
    results_counts = []
    
    for query in queries:
        start_q = time.time()
        results = indexer.search(query, k=10)
        end_q = time.time()
        query_times.append(end_q - start_q)
        results_counts.append(len(results))
        
    avg_query_time = statistics.mean(query_times)
    total_search_time = sum(query_times)
    
    print(f"Total Search Time: {total_search_time:.4f} seconds ({len(queries)} queries)")
    print(f"Avg Query Time: {avg_query_time:.6f} seconds")
    print(f"Avg Results Found: {statistics.mean(results_counts):.1f}")
    
    return {
        "indexing_time": indexing_time,
        "avg_query_time": avg_query_time
    }

def main():
    NUM_DOCS = 100000
    AVG_WORDS = 20
    NUM_QUERIES = 500
    
    # Generate data
    dataset = generate_dataset(NUM_DOCS, AVG_WORDS)
    
    # Generate queries (pick random words from dataset)
    all_text = " ".join([d[1] for d in dataset])
    all_words = all_text.split()
    queries = [random.choice(all_words) for _ in range(NUM_QUERIES)]
    
    results = {}
    
    # 1. SQLiteFTSIndex
    with TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn = sqlite3.connect(db_path)
        # Enable FTS5 if available (usually enabled by default in Python's sqlite3)
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(label_type, entity_id, content, entity_type)")
        
        sqlite_indexer = SQLiteFTSIndex(conn, name="benchmark_fts")
        results["SQLiteFTS"] = benchmark_indexer("SQLiteFTSIndex", sqlite_indexer, dataset, queries)
        conn.close()

    # 2. BM25SIndex
    if HAS_BM25S:
        bm25_indexer = BM25SIndex()
        results["BM25S"] = benchmark_indexer("BM25SIndex", bm25_indexer, dataset, queries)

    # Comparison
    print("\n\n=== Final Comparison ===")
    print(f"{'Indexer':<15} | {'Indexing (s)':<15} | {'Avg Search (s)':<15}")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"{name:<15} | {metrics['indexing_time']:<15.4f} | {metrics['avg_query_time']:<15.6f}")

if __name__ == "__main__":
    main()
