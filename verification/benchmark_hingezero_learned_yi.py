import ctypes as ct
import os
import tempfile
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import faiss
import h5py
import numpy as np

from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20


def native_rank(model, cue, candidates, k):
    ids = np.ascontiguousarray(candidates, dtype=np.int32)
    query = np.ascontiguousarray(cue, dtype=np.float32)
    output = np.empty(k, dtype=np.int32)
    ret = model.lib.hz20_rank_subset(
        model.handle, ct.c_void_p(query.ctypes.data),
        ct.c_void_p(ids.ctypes.data), len(ids), k,
        ct.c_void_p(output.ctypes.data)
    )
    if ret:
        raise RuntimeError(model._err())
    return output


def measure(model, queries, exact, k, label, get_candidates):
    hits, selected = 0, 0
    for q in queries[:3]:
        native_rank(model, q, get_candidates(q), k)
    start = time.perf_counter()
    for q, truth in zip(queries, exact):
        candidates = get_candidates(q)
        output = native_rank(model, q, candidates, k)
        hits += len(set(output.tolist()) & set(truth.tolist()))
        selected += len(candidates)
    elapsed = time.perf_counter() - start
    print(
        "ASSOCIATIVE_VIBE", label,
        "RECALL_ID=" + str(round(hits / (len(queries) * k), 6)),
        "QPS=" + str(round(len(queries) / elapsed, 2)),
        "CANDIDATES=" + str(round(selected / len(queries), 1)),
        flush=True,
    )


def main():
    faiss.omp_set_num_threads(1)
    with h5py.File("data/yi-128-ip.hdf5", "r") as data:
        print("SPLITS", {key: data[key].shape for key in data.keys()}, flush=True)
        bank = np.asarray(data["train"], dtype=np.float32)
        cues = np.ascontiguousarray(data["test"][:200], dtype=np.float32)
        correct = np.asarray(data["neighbors"][:len(cues), :100], dtype=np.int32)
        learn = np.ascontiguousarray(data["learn"][:], dtype=np.float32)
        labels = np.asarray(data["learn_neighbors"][:], dtype=np.int32)
    n, d = bank.shape
    k = correct.shape[1]
    if labels.shape[0] != len(learn) or labels.shape[1] < k:
        raise RuntimeError("Learning rows differ")
    labels = labels[:, :k]
    if np.any(labels < 0) or np.any(labels >= n):
        raise RuntimeError("Invalid learning neighbour IDs")
    frequency = np.bincount(labels.ravel(), minlength=n)
    popularity = np.argsort(-frequency, kind="stable").astype(np.int32)
    print("POPULAR_HITS", [(b, round(frequency[popularity[:b]].sum() / frequency.sum(), 6)) for b in (128, 256, 512, 1024, 2048, 4096)], flush=True)
    with tempfile.TemporaryDirectory() as folder:
        memory = HingeZeroCpp20("ip", storage_path=folder)
        memory.fit(bank)
        memory.lib.hz20_rank_subset.argtypes = [
            ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int,
            ct.c_int, ct.c_void_p
        ]
        memory.lib.hz20_rank_subset.restype = ct.c_int
        for budget in (128, 256, 512, 1024, 2048, 4096, 8192):
            shortlist = np.ascontiguousarray(popularity[:budget])
            measure(memory, cues, correct, k, "GLOBAL_" + str(budget),
                    lambda q, ids=shortlist: ids)
        faiss.normalize_L2(learn)
        graph = faiss.IndexHNSWFlat(d, 24, faiss.METRIC_INNER_PRODUCT)
        graph.hnsw.efConstruction = 120
        start = time.perf_counter()
        graph.add(learn)
        graph.hnsw.efSearch = 192
        print("LEARN_GRAPH_BUILD_S", round(time.perf_counter() - start, 2), flush=True)
        def local_candidates(cue, neighbors, global_count):
            q = np.ascontiguousarray(cue.reshape(1, -1), dtype=np.float32).copy()
            faiss.normalize_L2(q)
            _, matches = graph.search(q, neighbors)
            ids = labels[matches[0]].ravel()
            if global_count:
                ids = np.concatenate((ids, popularity[:global_count]))
            return np.ascontiguousarray(np.unique(ids), dtype=np.int32)
        for learned in (1, 2, 4, 8, 16, 32, 64):
            for global_count in (0, 256):
                measure(memory, cues, correct, k,
                        "LEARN_" + str(learned) + "_GLOBAL_" + str(global_count),
                        lambda q, l=learned, g=global_count: local_candidates(q, l, g))
    print("ASSOCIATIVE_VIBE_FINISHED")


if __name__ == "__main__":
    main()
