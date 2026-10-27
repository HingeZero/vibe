import ctypes as ct
import os
import tempfile
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import faiss
import h5py
import numpy as np

from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20


def build_lists(neighbors, assignment, n, groups, size):
    result = []
    for i in range(groups):
        rows = np.flatnonzero(assignment == i)
        freqs = np.bincount(neighbors[rows].ravel(), minlength=n)
        ids = np.flatnonzero(freqs)
        if len(ids) > size:
            ids = ids[np.argpartition(freqs[ids], -size)[-size:]]
        result.append(np.ascontiguousarray(ids[np.lexsort((ids, -freqs[ids]))], dtype=np.int32))
    return result


def run(mem, corpus_index, corpus_ids, query_index, postings, q, truth,
        probes, list_size, n):
    hits = 0
    total = 0
    start = time.perf_counter()
    for cue, gt in zip(q, truth):
        v = np.ascontiguousarray(cue.reshape(1, -1), dtype=np.float32)
        _, local = corpus_index.search(v, 100)
        norm = v.copy()
        faiss.normalize_L2(norm)
        _, cells = query_index.search(norm, probes)
        extras = np.concatenate([postings[c][:list_size] for c in cells[0]])
        candidates = np.unique(np.concatenate([corpus_ids[local[0]], extras])).astype(np.int32)
        out = np.empty(100, dtype=np.int32)
        rc = mem.lib.hz20_rank_subset(
            mem.handle, ct.c_void_p(v.ctypes.data),
            ct.c_void_p(candidates.ctypes.data), len(candidates),
            100, ct.c_void_p(out.ctypes.data)
        )
        if rc:
            raise RuntimeError(mem._err())
        hits += len(set(out.tolist()) & set(gt.tolist()))
        total += len(candidates)
    duration = time.perf_counter() - start
    print("HYBRID_LEARN", "POOL", n, "PROBES", probes,
          "LIST_SIZE", list_size, "RECALL", round(hits / (len(q) * 100), 6),
          "QPS", round(len(q) / duration, 2),
          "CANDIDATES", round(total / len(q), 1), flush=True)


def main():
    faiss.omp_set_num_threads(1)
    with h5py.File("data/yi-128-ip.hdf5", "r") as f:
        bank = np.asarray(f["train"], dtype=np.float32)
        learn = np.asarray(f["learn"], dtype=np.float32)
        labels = np.asarray(f["learn_neighbors"][:, :100], dtype=np.int32)
        query = np.asarray(f["test"][:200], dtype=np.float32)
        truth = np.asarray(f["neighbors"][:200, :100], dtype=np.int32)
    n,d = bank.shape
    freq = np.bincount(labels.ravel(), minlength=n)
    ordered = np.argsort(-freq, kind="stable").astype(np.int32)
    sample_rng = np.random.default_rng(20261027)
    sample = np.ascontiguousarray(learn[sample_rng.choice(len(learn), 24000, replace=False)])
    faiss.normalize_L2(sample)
    kmeans = faiss.Kmeans(d, 512, niter=9, nredo=1, seed=20261027)
    kmeans.train(sample)
    centers = np.ascontiguousarray(kmeans.centroids, dtype=np.float32)
    faiss.normalize_L2(centers)
    route_index = faiss.IndexFlatIP(d)
    route_index.add(centers)
    normalized_learn = np.ascontiguousarray(learn.copy())
    faiss.normalize_L2(normalized_learn)
    _, assigned = route_index.search(normalized_learn, 1)
    postings = build_lists(labels, assigned[:,0], n, 512, 1024)
    with tempfile.TemporaryDirectory() as folder:
        mem = HingeZeroCpp20("ip", storage_path=folder)
        mem.fit(bank)
        mem.lib.hz20_rank_subset.argtypes = [
            ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int,
            ct.c_int, ct.c_void_p
        ]
        mem.lib.hz20_rank_subset.restype = ct.c_int
        for pool_size in (8192, 16384, 32768):
            pool_ids = np.ascontiguousarray(ordered[:pool_size])
            pool = faiss.IndexFlatIP(d)
            pool.add(np.ascontiguousarray(bank[pool_ids]))
            for probes in (1, 2, 4):
                for list_size in (128, 256, 512, 1024):
                    run(mem,pool,pool_ids,route_index,postings,query,truth,
                        probes,list_size,pool_size)
    print("HYBRID_LEARN_FINISHED")


if __name__ == "__main__":
    main()
