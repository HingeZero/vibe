from pathlib import Path

import numpy as np
from numba import njit

from ..hingezero_cpp20.module import HingeZeroCpp20


@njit(cache=True)
def _routed_scores(bank, centroids, offsets, postings, cue, probes, k):
    clusters = centroids.shape[0]
    dims = centroids.shape[1]
    similarities = np.empty(clusters, dtype=np.float64)
    for c in range(clusters):
        score = 0.0
        for j in range(dims):
            score += float(centroids[c, j]) * float(cue[j])
        similarities[c] = score
    order = np.argsort(similarities)
    chosen = min(clusters, probes)
    count = 0
    for pos in range(clusters - 1, clusters - chosen - 1, -1):
        c = order[pos]
        count += offsets[c + 1] - offsets[c]
    while count < k and chosen < clusters:
        chosen += 1
        c = order[clusters - chosen]
        count += offsets[c + 1] - offsets[c]
    ids = np.empty(count, dtype=np.int32)
    scores = np.empty(count, dtype=np.float64)
    cursor = 0
    for pos in range(clusters - 1, clusters - chosen - 1, -1):
        c = order[pos]
        for ptr in range(offsets[c], offsets[c + 1]):
            row = postings[ptr]
            result = 0.0
            for j in range(dims):
                result += float(bank[row, j]) * float(cue[j])
            ids[cursor] = row
            scores[cursor] = result
            cursor += 1
    return ids, scores


class HingeZeroIndexed(HingeZeroCpp20):
    def __init__(self, metric, field_size=256, centroid_count=128):
        if metric not in ("ip", "normalized"):
            raise ValueError("Indexed pilot supports inner product only")
        super().__init__(metric, field_size)
        self.centroid_count = int(centroid_count)
        if not 2 <= self.centroid_count <= 4096:
            raise ValueError("Invalid centroid count")
        self.probes = 16
        self.routed_count = 0
        self._bank = None

    def append_memories(self, values, sources=None, *, _from_fit=False):
        if not _from_fit:
            raise RuntimeError("Indexed pilot requires rebuilding after append")
        return super().append_memories(values, sources=sources, _from_fit=True)

    def fit(self, X):
        super().fit(X)
        self._bank = np.memmap(
            self.root / "original-f32.bin", dtype="<f4", mode="r",
            shape=(self.count, self.dimension),
        )
        rng = np.random.default_rng(20261027)
        c = min(self.centroid_count, self.count)
        sample_rows = rng.choice(self.count, size=min(8192, self.count), replace=False)
        sample = np.asarray(self._bank[sample_rows], dtype=np.float32)
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        sample = sample / np.maximum(norms, 1e-20)
        centers = sample[rng.choice(len(sample), size=c, replace=False)].copy()
        for _ in range(4):
            labels = np.argmax(sample @ centers.T, axis=1)
            updated = np.zeros_like(centers)
            np.add.at(updated, labels, sample)
            sizes = np.bincount(labels, minlength=c)
            valid = sizes > 0
            new_norms = np.linalg.norm(updated[valid], axis=1, keepdims=True)
            centers[valid] = updated[valid] / np.maximum(new_norms, 1e-20)
        labels = np.empty(self.count, dtype=np.int32)
        for begin in range(0, self.count, 8192):
            end = min(begin + 8192, self.count)
            batch = np.asarray(self._bank[begin:end])
            magnitude = np.linalg.norm(batch, axis=1, keepdims=True)
            normalized = batch / np.maximum(magnitude, 1e-20)
            labels[begin:end] = np.argmax(normalized @ centers.T, axis=1).astype(np.int32)
        self.centroids = np.ascontiguousarray(centers, dtype=np.float32)
        counts = np.bincount(labels, minlength=c)
        self.offsets = np.empty(c + 1, dtype=np.int64)
        self.offsets[0] = 0
        np.cumsum(counts, out=self.offsets[1:])
        self.postings = np.ascontiguousarray(np.argsort(labels, kind="stable"), dtype=np.int32)
        np.savez(self.root / "indexed_router.npz",
                 centroids=self.centroids, offsets=self.offsets, postings=self.postings)
        self.probes = min(16, c)

    def load_memory(self, folder):
        super().load_memory(folder)
        self._bank = np.memmap(
            self.root / "original-f32.bin", dtype="<f4", mode="r",
            shape=(self.count, self.dimension),
        )
        with np.load(Path(folder) / "indexed_router.npz") as data:
            self.centroids = np.ascontiguousarray(data["centroids"], dtype=np.float32)
            self.offsets = np.ascontiguousarray(data["offsets"], dtype=np.int64)
            self.postings = np.ascontiguousarray(data["postings"], dtype=np.int32)
        if len(self.postings) != self.count or self.offsets[-1] != self.count:
            raise ValueError("Corrupt indexed router")
        self.probes = min(16, len(self.centroids))
        return self

    def set_query_arguments(self, probes=16):
        probes = int(probes)
        if not 1 <= probes <= len(self.centroids):
            raise ValueError("Invalid probe count")
        self.probes = probes

    def query(self, vector, n):
        if self._bank is None or not self.handle:
            raise RuntimeError("Index not fitted")
        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError("Invalid query")
        k = int(n)
        if not 1 <= k <= self.count:
            raise ValueError("Invalid neighbour count")
        ids, scores = _routed_scores(
            self._bank, self.centroids, self.offsets, self.postings,
            np.ascontiguousarray(q), self.probes, k,
        )
        if len(ids) < k:
            raise RuntimeError("Insufficient candidates")
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        selected = np.flatnonzero(scores >= cutoff)
        ranked = np.lexsort((ids[selected], -scores[selected]))[:k]
        result = ids[selected[ranked]]
        if len(np.unique(result)) != k:
            raise RuntimeError("Duplicate results")
        for idx in result:
            if not self.witness(int(idx), np.asarray(self._bank[idx])):
                raise RuntimeError("Original-memory Witness failed")
        self.queries += 1
        self.witness_count += k
        self.routed_count += len(ids)
        return result

    def get_additional(self):
        return {
            "engine": "HingeZero experimental indexed router",
            "refinement_steps": 0,
            "probes": self.probes,
            "centroids": len(self.centroids),
            "avg_scored_vectors": self.routed_count / max(1, self.queries),
            "witness_checks": self.witness_count,
            "full_precision_rerank": True,
        }

    def __str__(self):
        return f"HingeZeroIndexed(centroids={self.centroid_count},probes={self.probes})"
