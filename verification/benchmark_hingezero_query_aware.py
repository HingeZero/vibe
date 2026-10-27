import ctypes as ct
import os
import tempfile
import time

for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[env] = "1"

import faiss
import h5py
import numpy as np

from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20


def scored_ids(memory, q, candidates, k):
    ids = np.ascontiguousarray(np.unique(candidates), dtype=np.int32)
    if len(ids) < k:
        raise ValueError("Insufficient candidates")
    out = np.empty(k, dtype=np.int32)
    cue = np.ascontiguousarray(q, dtype=np.float32)
    rc = memory.lib.hz20_rank_subset(
        memory.handle, ct.c_void_p(cue.ctypes.data),
        ct.c_void_p(ids.ctypes.data), int(len(ids)),
        int(k), ct.c_void_p(out.ctypes.data)
    )
    if rc:
        raise RuntimeError(memory._err())
    return out


def build_lists(labels, assignments, n, k_clusters, maximum):
    all_lists = []
    for centroid in range(k_clusters):
        rows = np.flatnonzero(np.any(assignments == centroid, axis=1))
        if not len(rows):
            all_lists.append(np.empty(0, dtype=np.int32))
            continue
        freq = np.bincount(labels[rows].ravel(), minlength=n)
        positive = np.flatnonzero(freq)
        if len(positive) > maximum:
            selection = np.argpartition(freq[positive], -maximum)[-maximum:]
            positive = positive[selection]
        order = np.lexsort((positive, -freq[positive]))
        all_lists.append(np.ascontiguousarray(positive[order], dtype=np.int32))
    return all_lists


def get_candidates(postings, index, q, probe_count, per_list):
    query = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32).copy()
    faiss.normalize_L2(query)
    _, cluster_ids = index.search(query, probe_count)
    return np.concatenate([postings[i][:per_list] for i in cluster_ids[0]])


def benchmark(memory, q, gt, index, lists, probes, budget):
    n = len(q)
    k = gt.shape[1]
    total = 0
    picked = 0
    start = time.perf_counter()
    for cue, truth in zip(q, gt):
        candidates = get_candidates(lists, index, cue, probes, budget)
        try:
            answer = scored_ids(memory, cue, candidates, k)
        except ValueError:
            return None
        total += len(set(answer.tolist()) & set(truth.tolist()))
        picked += len(np.unique(candidates))
    elapsed = time.perf_counter() - start
    return total / (n * k), n / elapsed, picked / n


def main():
    faiss.omp_set_num_threads(1)
    with h5py.File("data/yi-128-ip.hdf5", "r") as f:
        bank = np.asarray(f["train"], dtype=np.float32)
        learn = np.asarray(f["learn"], dtype=np.float32)
        neighbor_ids = np.asarray(f["learn_neighbors"][:, :100], dtype=np.int32)
        query = np.asarray(f["test"][:200], dtype=np.float32)
        truth = np.asarray(f["neighbors"][:200, :100], dtype=np.int32)
    if neighbor_ids.shape != (len(learn), 100):
        raise RuntimeError("Incorrect learning split")
    n, d = bank.shape
    seed = 20261027
    for k_clusters in (128, 512):
        begin = time.perf_counter()
        sample_rng = np.random.default_rng(seed)
        sample = np.ascontiguousarray(
            learn[sample_rng.choice(len(learn), min(24000, len(learn)), replace=False)]
        )
        faiss.normalize_L2(sample)
        km = faiss.Kmeans(d, k_clusters, niter=9, nredo=1, seed=seed,
                         verbose=False, gpu=False)
        km.train(sample)
        centers = np.ascontiguousarray(km.centroids, dtype=np.float32)
        faiss.normalize_L2(centers)
        lookup = faiss.IndexFlatIP(d)
        lookup.add(centers)
        learn_copy = np.ascontiguousarray(learn, dtype=np.float32)
        faiss.normalize_L2(learn_copy)
        _, assigned = lookup.search(learn_copy, 4)
        print("CLUSTER_BUILD", k_clusters, "SECONDS", round(time.perf_counter() - begin, 2), flush=True)
        with tempfile.TemporaryDirectory() as folder:
            memory = HingeZeroCpp20("ip", storage_path=folder)
            memory.fit(bank)
            memory.lib.hz20_rank_subset.argtypes = [
                ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int,
                ct.c_int, ct.c_void_p
            ]
            memory.lib.hz20_rank_subset.restype = ct.c_int
            for replication in (1, 2, 4):
                begin = time.perf_counter()
                lists = build_lists(neighbor_ids, assigned[:, :replication],
                                   n, k_clusters, 4096)
                lens = np.array([len(x) for x in lists])
                print("PLACEMENT", k_clusters, replication, "AVG_LIST", round(float(lens.mean()), 1),
                      "BUILD_S", round(time.perf_counter() - begin, 2), flush=True)
                for probes in (1, 2, 4):
                    for budget in (128, 256, 512, 1024, 2048):
                        measured = benchmark(memory, query, truth, lookup, lists,
                                             probes, budget)
                        if measured is not None:
                            recall, qps, candidates = measured
                            print("MIPS_ROUTE", "CENTERS", k_clusters, "COPIES", replication,
                                  "PROBES", probes, "PER_LIST", budget,
                                  "RECALL", round(recall, 5), "QPS", round(qps, 2),
                                  "CANDIDATES", round(candidates, 1), flush=True)
    print("MIPS_ROUTING_COMPLETE")


if __name__ == "__main__":
    main()
