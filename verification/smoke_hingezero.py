"""Small local correctness smoke test; NOT a VIBE leaderboard result."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vibe.algorithms.hingezero.module import HingeZeroVIBE


def exact_neighbors(X, q, metric, k):
    xf, qf = X.astype(np.float64), q.astype(np.float64)
    if metric == "euclidean":
        scores = -np.sum((xf - qf) ** 2, axis=1)
    elif metric == "cosine":
        norms = np.linalg.norm(xf, axis=1) * np.linalg.norm(qf)
        scores = np.divide(xf @ qf, norms, out=np.zeros(len(xf)), where=norms > 0)
    else:
        scores = xf @ qf
    return np.lexsort((np.arange(len(X)), -scores))[:k].astype(np.int64)


def main():
    rng = np.random.default_rng(20261024)
    N, D, K = 2048, 32, 100
    results = {}
    for metric in ("euclidean", "cosine", "ip", "normalized"):
        X = rng.normal(size=(N, D)).astype(np.float32)
        if metric == "normalized":
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        before = X.copy()
        algo = HingeZeroVIBE(metric, 16, 8, 4, N)
        algo.fit(X)
        assert np.array_equal(X, before), "fit changed VIBE corpus"
        algo.set_query_arguments(16, N, 0, 8)
        q = rng.normal(size=D).astype(np.float32)
        if metric == "normalized":
            q /= np.linalg.norm(q)
        ids = algo.query(q, K)
        expected = exact_neighbors(X, q, metric, K)
        np.testing.assert_array_equal(ids, expected)
        assert algo.get_additional()["failed_checks"] == 0
        assert algo.get_additional()["seed_score_regressions"] == 0
        results[metric] = {"exact_top_100": True, "unique_ids": True,
                           "bank_unchanged": True, "witness_failures": 0}
        algo.mode = "full"
        algo.set_query_arguments(8, 256, 256, 8)
        full = algo.query(q, K)
        algo.mode = "zero_steps"
        algo.set_query_arguments(8, 256, 256, 0)
        zero_steps = algo.query(q, K)
        results[metric]["full_recall_local"] = round(len(set(full) & set(expected)) / K, 4)
        results[metric]["zero_steps_recall_local"] = round(len(set(zero_steps) & set(expected)) / K, 4)
    print(json.dumps({"type": "local_smoke_not_VIBE_result", "status": "PASS",
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
