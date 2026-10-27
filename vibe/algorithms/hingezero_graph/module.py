import ctypes as ct
from pathlib import Path
import time

import faiss
import numpy as np

from ..hingezero_cpp20.module import HingeZeroCpp20


class HingeZeroGraph(HingeZeroCpp20):
    def __init__(self, metric, field_size=256, m=24, ef_construction=120):
        if metric != "ip":
            raise ValueError("Graph pilot supports inner product")
        super().__init__(metric, field_size)
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = 480
        self.rerank_budget = 480
        self.graph = None
        self.graph_queries = 0
        self.graph_witnesses = 0
        self.total_ms = 0.0

    def _prepare_ranker(self):
        self.lib.hz20_rank_subset.argtypes = [
            ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int,
            ct.c_int, ct.c_void_p
        ]
        self.lib.hz20_rank_subset.restype = ct.c_int

    def fit(self, X):
        super().fit(X)
        faiss.omp_set_num_threads(1)
        self._prepare_ranker()
        bank = np.memmap(self.root / "original-f32.bin", dtype="<f4",
                         mode="r", shape=(self.count, self.dimension))
        self.graph = faiss.IndexHNSWFlat(
            self.dimension, self.m, faiss.METRIC_INNER_PRODUCT
        )
        self.graph.hnsw.efConstruction = self.ef_construction
        self.graph.add(np.ascontiguousarray(bank))
        faiss.write_index(self.graph, str(self.root / "hingezero_graph.index"))

    def load_memory(self, folder):
        super().load_memory(folder)
        faiss.omp_set_num_threads(1)
        self._prepare_ranker()
        self.graph = faiss.read_index(str(Path(folder) / "hingezero_graph.index"))
        if self.graph.ntotal != self.count:
            raise RuntimeError("Graph and memory count disagree")
        return self

    def set_query_arguments(self, ef_search=480, rerank_budget=480):
        self.ef_search = int(ef_search)
        self.rerank_budget = int(rerank_budget)
        if self.ef_search < 100 or self.rerank_budget < 100:
            raise ValueError("Invalid graph search settings")

    def query(self, vector, n):
        if self.graph is None or not self.handle:
            raise RuntimeError("Graph and stored memory required")
        q = np.ascontiguousarray(vector, dtype=np.float32)
        k = int(n)
        if q.shape != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError("Invalid query")
        if k < 1 or k > self.count:
            raise ValueError("Invalid neighbour count")
        if self.rerank_budget >= self.count:
            return super().query(q, k)
        budget = min(self.count, max(k, self.rerank_budget))
        self.graph.hnsw.efSearch = max(budget, self.ef_search)
        start = time.perf_counter()
        _, found = self.graph.search(q.reshape(1, -1), budget)
        candidates = np.ascontiguousarray(found[0], dtype=np.int32)
        if np.any(candidates < 0) or len(np.unique(candidates)) < k:
            raise RuntimeError("Insufficient unique graph candidates")
        result = np.empty(k, dtype=np.int32)
        self._check(self.lib.hz20_rank_subset(
            self.handle, ct.c_void_p(q.ctypes.data),
            ct.c_void_p(candidates.ctypes.data), len(candidates), k,
            ct.c_void_p(result.ctypes.data)
        ))
        if len(np.unique(result)) != k:
            raise RuntimeError("Duplicate result IDs")
        self.total_ms += 1000 * (time.perf_counter() - start)
        self.graph_queries += 1
        self.graph_witnesses += k
        return result

    def get_additional(self):
        return {
            "engine": "HingeZero Graph Lab",
            "ef_search": self.ef_search,
            "rerank_budget": self.rerank_budget,
            "m": self.m,
            "witness_checks": self.graph_witnesses,
            "query_ms_total": self.total_ms,
        }

    def __str__(self):
        return f"HingeZeroGraph(m={self.m},ef={self.ef_search})"
