"""Exact brute-force vector index backend."""

from __future__ import annotations

import orjson
import math
from typing import Any

from .base import VectorIndex


class BruteForceIndex(VectorIndex):
    """Exact k-NN backend using linear scan."""

    def __init__(self, dim: int, method: str = "flat", options: dict[str, Any] | None = None) -> None:
        super().__init__(dim=dim, method=method, options=options)
        self.metric = (self.options.get("metric") or "cosine").lower()
        if self.metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self._vectors: dict[int, list[float]] = {}

    def add(self, ids: list[int], vectors: list[list[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        for idx, vector in zip(ids, vectors):
            self._validate_vector(vector)
            self._vectors[int(idx)] = list(vector)

    def remove(self, ids: list[int]) -> None:
        for idx in ids:
            self._vectors.pop(int(idx), None)

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        self._validate_vector(vector)
        scores = []
        for idx, candidate in self._vectors.items():
            score = self._score(vector, candidate)
            scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    def search_ids(self, vector: list[float], ids: list[int], k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        self._validate_vector(vector)
        scores = []
        for idx in ids:
            candidate = self._vectors.get(int(idx))
            if candidate is None:
                continue
            score = self._score(vector, candidate)
            scores.append((int(idx), score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    def save(self, path: str) -> None:
        payload = {
            "dim": self.dim,
            "method": self.method,
            "options": self.options,
            "metric": self.metric,
            "vectors": {str(idx): vector for idx, vector in self._vectors.items()},
        }
        serialized = orjson.dumps(payload)
        with open(path, "wb") as handle:
            handle.write(serialized)

    def load(self, path: str) -> None:
        with open(path, "rb") as handle:
            payload = orjson.loads(handle.read())
        self.dim = int(payload["dim"])
        self.method = payload.get("method", self.method)
        self.options = payload.get("options", {})
        self.metric = (payload.get("metric") or "cosine").lower()
        vectors = payload.get("vectors", {})
        self._vectors = {int(idx): list(vec) for idx, vec in vectors.items()}

    def supports_exact(self) -> bool:
        return True

    def supports_remove(self) -> bool:
        return True

    def get_vector(self, idx: int) -> list[float] | None:
        return self._vectors.get(int(idx))

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}")

    def _score(self, left: list[float], right: list[float]) -> float:
        if self.metric == "l2":
            return -self._l2_distance(left, right)
        return self._cosine_similarity(left, right)

    def _l2_distance(self, left: list[float], right: list[float]) -> float:
        total = 0.0
        for a, b in zip(left, right):
            diff = a - b
            total += diff * diff
        return math.sqrt(total)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for a, b in zip(left, right):
            dot += a * b
            left_norm += a * a
            right_norm += b * b
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / math.sqrt(left_norm * right_norm)
