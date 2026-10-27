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


def main():
    faiss.omp_set_num_threads(1)
    with h5py.File("data/yi-128-ip.hdf5", "r") as f:
        bank = np.asarray(f["train"], dtype=np.float32)
        learned = np.asarray(f["learn_neighbors"][:, :100], dtype=np.int32)
        queries = np.ascontiguousarray(f["test"][:200], dtype=np.float32)
        truths = np.asarray(f["neighbors"][:200, :100], dtype=np.int32)
    n, d = bank.shape
    freq = np.bincount(learned.ravel(), minlength=n)
    order = np.argsort(-freq, kind="stable").astype(np.int32)
    active = int(np.count_nonzero(freq))
    print("ACTIVE_LEARN_MEMORY", active, "TOTAL", n, "FRACTION", round(active/n, 5), flush=True)
    with tempfile.TemporaryDirectory() as folder:
        mem = HingeZeroCpp20("ip", storage_path=folder)
        mem.fit(bank)
        mem.lib.hz20_rank_subset.argtypes = [
            ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int,
            ct.c_int, ct.c_void_p
        ]
        mem.lib.hz20_rank_subset.restype = ct.c_int
        for budget in (1024, 2048, 4096, 8192, 16384, 32768, 65536, active):
            budget = min(budget, active)
            chosen = order[:budget]
            corpus = np.ascontiguousarray(bank[chosen], dtype=np.float32)
            index = faiss.IndexFlatIP(d)
            index.add(corpus)
            true_coverage = float(np.isin(truths, chosen).mean())
            index.search(queries[:5], 100)
            begin = time.perf_counter()
            pred = []
            for q in queries:
                _, local_ids = index.search(q[None, :], 100)
                ids = np.ascontiguousarray(chosen[local_ids[0]], dtype=np.int32)
                out = np.empty(100, dtype=np.int32)
                rc = mem.lib.hz20_rank_subset(
                    mem.handle, ct.c_void_p(q.ctypes.data),
                    ct.c_void_p(ids.ctypes.data), 100, 100,
                    ct.c_void_p(out.ctypes.data)
                )
                if rc:
                    raise RuntimeError(mem._err())
                pred.append(out)
            duration = time.perf_counter() - begin
            recall = sum(len(set(x.tolist()) & set(t.tolist())) for x, t in zip(pred, truths)) / (len(queries) * 100)
            print("LEARN_SUBSET", "SIZE", budget, "COVERAGE", round(true_coverage, 6),
                  "RECALL", round(recall, 6), "QPS", round(len(queries)/duration, 2), flush=True)
    print("LEARN_SUBSET_FINISHED")


if __name__ == "__main__":
    main()
