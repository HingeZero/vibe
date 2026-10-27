import os
import tempfile
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from vibe.algorithms.hingezero_graph.module import HingeZeroGraph


def main():
    rng = np.random.default_rng(20261027)
    n, d, k = 12000, 128, 100
    bank = rng.normal(size=(n, d)).astype(np.float32)
    cues = np.ascontiguousarray(
        bank[rng.choice(n, 40, replace=False)]
        + 0.35 * rng.normal(size=(40, d)).astype(np.float32)
    )
    truth = []
    bank64 = bank.astype(np.float64)
    for cue in cues:
        scores = bank64 @ cue.astype(np.float64)
        truth.append(set(np.argpartition(scores, -k)[-k:].tolist()))
    with tempfile.TemporaryDirectory() as folder:
        index = HingeZeroGraph("ip", m=24, ef_construction=100)
        index.storage_path = folder
        index.fit(bank)
        for ef in (120, 240, 480, 960):
            index.set_query_arguments(ef_search=ef, rerank_budget=ef)
            start = time.perf_counter()
            answers = [index.query(cue, k) for cue in cues]
            elapsed = time.perf_counter() - start
            hits = sum(len(set(row.tolist()) & gt) for row, gt in zip(answers, truth))
            for row in answers:
                assert len(set(row.tolist())) == k
                assert index.witness(int(row[0]), bank[int(row[0])])
            print(f"FULL_PIPELINE EF={ef} RECALL={hits/(len(cues)*k):.6f} QPS={len(cues)/elapsed:.2f}", flush=True)
        index.set_query_arguments(ef_search=960, rerank_budget=n)
        for cue, expected in zip(cues[:3], truth[:3]):
            found = index.query(cue, k)
            assert set(found.tolist()) == expected
        recovered = HingeZeroGraph("ip", m=24, ef_construction=100)
        recovered.load_memory(folder)
        recovered.set_query_arguments(ef_search=480, rerank_budget=480)
        index.set_query_arguments(ef_search=480, rerank_budget=480)
        assert np.array_equal(recovered.query(cues[0], k), index.query(cues[0], k))
        assert not recovered.witness(0, bank[0] + 1)
    print("HINGEZERO_GRAPH_PIPELINE_PASS")


if __name__ == "__main__":
    main()
