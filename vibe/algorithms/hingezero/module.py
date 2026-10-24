"""HingeZero C++ v3 adapter for VIBE: FAISS IVF/PQ + dual-tanh refinement.

The pinned native kernel performs associative routing, witness checks, candidate
expansion and original-metric scoring. This is NOT the 1-bit memory architecture.
"""
import ctypes as C
import os
from pathlib import Path

import faiss
import numpy as np
from ..base.module import BaseANN


class _Parameters(C.Structure):
    _fields_ = [(name, C.c_int32) for name in (
        "metric", "dtype", "mode", "steps", "field_limit", "threads"
    )] + [(name, C.c_double) for name in (
        "alpha", "eps", "lam", "commit_margin", "route_mix", "retention_mix"
    )]


_STAT_NAMES = (
    "witness_checks", "retention_events", "retention_commits", "append_checks",
    "candidate_reads", "seed_reads", "extra_reads", "failed_checks",
    "refinement_route_events", "refinement_shift_sum", "seed_score_regressions",
    "original_metric_evaluations", "output_record_checks",
)
_MODES = {"full": 0, "without_hz": 1, "zero_steps": 2, "no_retention": 3,
          "query_expansion": 4}
_NATIVE_PATH = "/opt/hingezero/libhingezero.so"


def _native():
    path = os.environ.get("HINGEZERO_NATIVE_LIBRARY", _NATIVE_PATH)
    if not Path(path).is_file():
        raise RuntimeError("HingeZero C++ library unavailable at " + path)
    lib = C.CDLL(path)
    lib.hz_error.argtypes = []
    lib.hz_error.restype = C.c_char_p
    lib.hz_abi_version.argtypes = []
    lib.hz_abi_version.restype = C.c_int
    if lib.hz_abi_version() != 3:
        raise RuntimeError("HingeZero native ABI version is not 3")
    lib.hz_create.argtypes = [
        C.c_void_p, C.c_int64, C.c_int32, C.c_void_p,
        C.c_int32, C.c_void_p, C.c_int32, C.c_int32, C.POINTER(_Parameters),
    ]
    lib.hz_create.restype = C.c_void_p
    lib.hz_route.argtypes = [C.c_void_p, C.c_void_p]
    lib.hz_route.restype = C.c_int
    lib.hz_expand.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int32]
    lib.hz_expand.restype = C.c_int
    lib.hz_finish.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
    lib.hz_finish.restype = C.c_int
    lib.hz_destroy.argtypes = [C.c_void_p]
    lib.hz_destroy.restype = None
    return lib


class HingeZeroVIBE(BaseANN):
    """Static float k-NN adapter, single-core build and search."""

    def __init__(self, metric, nlist, pq_m, pq_bits, train_size):
        if metric not in ("euclidean", "ip", "cosine", "normalized"):
            raise ValueError("Unsupported VIBE metric: " + str(metric))
        self.metric = metric
        self.nlist = int(nlist)
        self.pq_m = int(pq_m)
        self.pq_bits = int(pq_bits)
        self.train_size = int(train_size)
        self.seed = 20261008
        self.mode = "full"
        self.alpha, self.eps, self.lam = 4.5, 0.10, 0.02
        self.commit_margin, self.route_mix, self.retention_mix = 0.10, 0.10, 0.20
        self.field_limit, self.threads = 256, 1
        if min(self.nlist, self.pq_m, self.train_size) < 1:
            raise ValueError("Build parameters must be positive")
        if not 1 <= self.pq_bits <= 8:
            raise ValueError("pq_bits must be in [1,8]")
        self.index = None
        self.bank = None
        self.search_parameters = None
        self.lib = None
        self.stats = {name: 0.0 for name in _STAT_NAMES}
        faiss.omp_set_num_threads(1)

    def _transform(self, x):
        """Never normalize the VIBE input corpus in place."""
        z = np.asarray(x, dtype=np.float32, order="C")
        if self.metric != "cosine":
            return z
        z = z.copy(order="C")
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        np.divide(z, norms, out=z, where=norms > 0)
        return z

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or len(X) < 2 or X.shape[1] < 1:
            raise ValueError("X must be a 2D nonempty vector bank")
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError("Only float VIBE datasets are supported")
        if not np.isfinite(X).all():
            raise ValueError("Input bank contains nonfinite values")
        if len(X) > np.iinfo(np.int32).max:
            raise ValueError("Bank size exceeds native int32 output IDs")
        self.bank = np.ascontiguousarray(X, dtype=np.float32)
        n, d = self.bank.shape
        nlist = min(self.nlist, max(1, n // 40))
        pq_m = max(m for m in range(1, min(self.pq_m, d) + 1) if d % m == 0)
        ntrain = min(self.train_size, n)
        if ntrain < max(nlist, 1 << self.pq_bits):
            raise ValueError("Insufficient training data for IVF/PQ")
        faiss.omp_set_num_threads(1)
        fm = faiss.METRIC_L2 if self.metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        coarse = faiss.IndexFlatL2(d) if fm == faiss.METRIC_L2 else faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(coarse, d, nlist, pq_m, self.pq_bits, fm)
        index.cp.seed = self.seed
        index.cp.niter = 12
        index.pq.cp.seed = self.seed
        index.pq.cp.niter = 12
        rng = np.random.default_rng(self.seed)
        sample_ids = np.sort(rng.choice(n, ntrain, replace=False))
        index.train(self._transform(self.bank[sample_ids]))
        for start in range(0, n, 4096):
            index.add(self._transform(self.bank[start:start + 4096]))
        if int(index.ntotal) != n:
            raise RuntimeError("FAISS index does not include entire bank")
        self.index = index
        self.actual_nlist, self.actual_pq_m = nlist, pq_m
        self.lib = _native()

    def set_query_arguments(self, nprobe, candidates, expansion_candidates, steps):
        self.nprobe = int(nprobe)
        self.candidates = int(candidates)
        self.expansion_candidates = int(expansion_candidates)
        self.steps = int(steps)
        if min(self.nprobe, self.candidates) < 1 or min(self.expansion_candidates, self.steps) < 0:
            raise ValueError("Invalid query parameter counts")
        if self.index is not None:
            self.index.nprobe = min(self.nprobe, self.index.nlist)
        self.search_parameters = (self.nprobe, self.candidates,
                                  self.expansion_candidates, self.steps)
        self.stats = {name: 0.0 for name in _STAT_NAMES}

    def _check(self, code):
        if code != 0:
            message = self.lib.hz_error()
            raise RuntimeError("HingeZero C++: " +
                               (message.decode("utf-8", "replace") if message else "unknown error"))

    def query(self, vector, n):
        if self.index is None or self.bank is None or self.lib is None:
            raise RuntimeError("fit must precede query")
        if self.search_parameters is None:
            raise RuntimeError("set_query_arguments must precede query")
        n = int(n)
        if not 1 <= n <= len(self.bank):
            raise ValueError("Invalid requested k")
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.bank.shape[1] or not np.isfinite(query).all():
            raise ValueError("Invalid query vector")
        query = np.ascontiguousarray(query)
        self.index.nprobe = min(self.nprobe, self.index.nlist)
        faiss.omp_set_num_threads(1)
        seed_count = min(len(self.bank), max(n, self.candidates))
        _, seeds = self.index.search(self._transform(query), seed_count)
        seeds = np.ascontiguousarray(seeds, dtype=np.int64)
        params = _Parameters(
            {"euclidean": 0, "ip": 1, "normalized": 1, "cosine": 2}[self.metric],
            0, _MODES[self.mode], self.steps, self.field_limit, 1,
            self.alpha, self.eps, self.lam, self.commit_margin,
            self.route_mix, self.retention_mix,
        )
        handle = self.lib.hz_create(
            C.c_void_p(self.bank.ctypes.data), len(self.bank), self.bank.shape[1],
            C.c_void_p(query.ctypes.data), 1,
            C.c_void_p(seeds.ctypes.data), seeds.shape[1], n, C.byref(params),
        )
        if not handle:
            message = self.lib.hz_error()
            raise RuntimeError("HingeZero create: " +
                               (message.decode("utf-8", "replace") if message else "unknown"))
        try:
            if self.mode != "without_hz" and self.expansion_candidates > 0:
                routed = np.empty_like(query)
                self._check(self.lib.hz_route(handle, C.c_void_p(routed.ctypes.data)))
                requested = min(len(self.bank), seed_count + self.expansion_candidates)
                _, expanded = self.index.search(self._transform(routed), requested)
                expanded = np.ascontiguousarray(expanded, dtype=np.int64)
                self._check(self.lib.hz_expand(
                    handle, C.c_void_p(expanded.ctypes.data),
                    expanded.shape[1], self.expansion_candidates,
                ))
            ids = np.empty((1, n), dtype=np.int32)
            stats = np.zeros(len(_STAT_NAMES), dtype=np.float64)
            self._check(self.lib.hz_finish(
                handle, C.c_void_p(ids.ctypes.data), C.c_void_p(stats.ctypes.data)
            ))
            for key, value in zip(_STAT_NAMES, stats):
                self.stats[key] += float(value)
        finally:
            self.lib.hz_destroy(handle)
        result = ids[0].astype(np.int64, copy=False)
        if (len(np.unique(result)) != n or np.any(result < 0)
                or np.any(result >= len(self.bank))):
            raise RuntimeError("Duplicate or out-of-range IDs returned")
        return result

    def get_additional(self):
        return {
            "candidate_backend": "FAISS IVF/PQ",
            "recurrence": "dual-tanh C++ v3",
            "native_mode": self.mode,
            "index_build_threads": 1,
            "search_threads": 1,
            "witness_checks": int(self.stats["witness_checks"]),
            "failed_checks": int(self.stats["failed_checks"]),
            "seed_score_regressions": int(self.stats["seed_score_regressions"]),
        }

    def __str__(self):
        q = self.search_parameters
        qdesc = "unconfigured" if q is None else (
            f"nprobe={q[0]},candidates={q[1]},extra={q[2]},steps={q[3]}"
        )
        return (f"HingeZeroC++v3(metric={self.metric},"
                f"nlist={getattr(self, 'actual_nlist', self.nlist)},"
                f"pq_m={getattr(self, 'actual_pq_m', self.pq_m)},"
                f"bits={self.pq_bits},train={self.train_size},mode={self.mode},{qdesc})")
