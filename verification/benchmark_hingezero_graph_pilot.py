import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import time
import faiss
import numpy as np

faiss.omp_set_num_threads(1)
rng = np.random.default_rng(20261027)
n, d, k = 40000, 128, 100
bank = rng.normal(size=(n, d)).astype(np.float32)
q_near = bank[rng.choice(n, 40, replace=False)] + .35 * rng.normal(size=(40, d)).astype(np.float32)
q_ood = rng.normal(size=(40, d)).astype(np.float32)
query_sets = [("near", np.ascontiguousarray(q_near)), ("ood", np.ascontiguousarray(q_ood))]
index = faiss.IndexHNSWFlat(d, 24, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 120
start = time.perf_counter()
index.add(bank)
print("BUILD_SECONDS", round(time.perf_counter() - start, 2), flush=True)
truth = {}
for name, queries in query_sets:
    exact = []
    for q in queries:
        scores = bank.astype(np.float64) @ q.astype(np.float64)
        exact.append(set(np.argpartition(scores, -k)[-k:].tolist()))
    truth[name] = exact
for ef in [120, 240, 480, 960, 1920, 3840]:
    index.hnsw.efSearch = ef
    for name, queries in query_sets:
        index.search(queries[:2], k)
        start = time.perf_counter()
        scores, ids = index.search(queries, k)
        elapsed = time.perf_counter() - start
        hits = sum(len(set(rows) & exact) for rows, exact in zip(ids, truth[name]))
        if np.any(ids < 0):
            raise ValueError("Missing neighbours")
        print("METHOD=HNSW", "EF=" + str(ef), "SET=" + name,
              "RECALL=" + str(round(hits / (len(queries) * k), 5)),
              "QPS=" + str(round(len(queries) / elapsed, 2)), flush=True)
