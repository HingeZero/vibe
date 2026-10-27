import os
import tempfile
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import h5py
import numpy as np

from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20
from vibe.algorithms.hingezero_graph.module import HingeZeroGraph


def measure(engine, queries, truths, k, name, ef=None):
    for q in queries[:5]:
        engine.query(q, k)
    start = time.perf_counter()
    hits = 0
    for q, truth in zip(queries, truths):
        ids = engine.query(q, k)
        if len(np.unique(ids)) != k:
            raise RuntimeError("Invalid result count")
        hits += len(set(ids.tolist()) & set(truth.tolist()))
    elapsed = time.perf_counter() - start
    print(
        "MATCHED_VIBE", name,
        "EF=" + str(ef if ef is not None else 0),
        "RECALL_ID=" + str(round(hits / (len(queries) * k), 7)),
        "QPS=" + str(round(len(queries) / elapsed, 2)),
        "QUERIES=" + str(len(queries)),
        "SECONDS=" + str(round(elapsed, 4)),
        flush=True,
    )


def main():
    filename = "data/yi-128-ip.hdf5"
    with h5py.File(filename, "r") as f:
        bank = np.asarray(f["train"], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:200], dtype=np.float32)
        truths = np.asarray(f["neighbors"][:len(queries), :100], dtype=np.int32)
    n, d = bank.shape
    k = truths.shape[1]
    print("DATASET", filename, "N", n, "D", d, "K", k, flush=True)
    with tempfile.TemporaryDirectory() as folder:
        graph = HingeZeroGraph("ip", m=24, ef_construction=120)
        graph.storage_path = folder
        start = time.perf_counter()
        graph.fit(bank)
        print("INDEX_BUILD_S", round(time.perf_counter() - start, 2), flush=True)
        old = HingeZeroCpp20("ip")
        old.load_memory(folder)
        old.set_query_arguments(steps=0, budget=208)
        measure(old, queries, truths, k, "HingeZero-Cpp20-208")
        for ef in (240, 480, 960, 1920, 3840):
            graph.set_query_arguments(ef_search=ef, rerank_budget=ef)
            measure(graph, queries, truths, k, "HingeZero-Graph", ef)
    print("MATCHED_VIBE_RUN_FINISHED")


if __name__ == "__main__":
    main()
