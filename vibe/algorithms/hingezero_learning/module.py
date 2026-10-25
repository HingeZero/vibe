import ctypes as ct
import numpy as np
import hnswlib
from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20


class HingeZeroLearnedRouting(HingeZeroCpp20):
    def __init__(self, metric, graph_degree=24, build_ef=120):
        if metric != "ip":
            raise ValueError("Learned-routing experiment supports IP only")
        super().__init__(metric, field_size=256)
        self.graph_degree = int(graph_degree)
        self.build_ef = int(build_ef)
        self.references = 64
        self.search_ef = 400
        self.neighbor_labels = None
        self.routing = None

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        x = np.ascontiguousarray(X_learn, dtype=np.float32)
        labels = np.ascontiguousarray(X_learn_neighbors, dtype=np.int32)
        if x.ndim != 2 or x.shape[1] != X_train.shape[1]:
            raise ValueError("Learning queries do not match vector dimensions")
        if labels.shape[0] != len(x) or labels.ndim != 2 or labels.shape[1] < 100:
            raise ValueError("Expected learning labels of 100 neighbours per query")
        if np.any(labels < 0) or np.any(labels >= len(X_train)):
            raise ValueError("Bad stored-memory reference in learning labels")
        super().fit(X_train)
        self.neighbor_labels = labels
        self.routing = hnswlib.Index(space="cosine", dim=self.dimension)
        self.routing.init_index(
            max_elements=len(x), ef_construction=self.build_ef,
            M=self.graph_degree, random_seed=19)
        self.routing.add_items(x, ids=np.arange(len(x), dtype=np.int32), num_threads=1)
        self.routing.set_num_threads(1)
        self.routing.set_ef(self.search_ef)
        self.lib.hz20_query_candidates.argtypes = [
            ct.c_void_p, ct.c_void_p, ct.c_int32, ct.c_void_p, ct.c_int32, ct.c_void_p]
        self.lib.hz20_query_candidates.restype = ct.c_int

    def set_query_arguments(self, references, search_ef):
        self.references = int(references)
        self.search_ef = int(search_ef)
        if not 1 <= self.references <= len(self.neighbor_labels) or self.search_ef < self.references:
            raise ValueError("Bad learned query routing budget")
        self.routing.set_ef(self.search_ef)

    def query(self, vector, n):
        q = np.ascontiguousarray(vector, dtype=np.float32)
        if q.shape != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError("Invalid query")
        n = int(n)
        refs, _ = self.routing.knn_query(
            q[None, :], k=self.references, num_threads=1)
        candidate_ids = np.unique(self.neighbor_labels[refs[0]].ravel())
        if len(candidate_ids) < n:
            raise RuntimeError("Learned query routing supplied fewer than k candidates")
        candidate_ids = np.ascontiguousarray(candidate_ids, dtype=np.int32)
        out = np.empty(n, dtype=np.int32)
        rc = self.lib.hz20_query_candidates(
            self.handle, ct.c_void_p(q.ctypes.data), n,
            ct.c_void_p(candidate_ids.ctypes.data), len(candidate_ids),
            ct.c_void_p(out.ctypes.data))
        self._check(rc)
        return out

    def __str__(self):
        return (f"HingeZeroLearnedRouting(M={self.graph_degree},"
                f"build_ef={self.build_ef},references={self.references},"
                f"search_ef={self.search_ef})")
