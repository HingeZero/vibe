import os
import tempfile
import time

import numpy as np

from vibe.algorithms.hingezero_indexed.module import HingeZeroIndexed


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    rng = np.random.default_rng(20261027)
    bank = rng.normal(size=(4096, 128)).astype(np.float32)
    cues = rng.normal(size=(128, 128)).astype(np.float32)
    k = 100
    with tempfile.TemporaryDirectory() as path:
        index = HingeZeroIndexed("ip", centroid_count=32, field_size=64)
        index.storage_path = path
        index.fit(bank)
        index.set_query_arguments(probes=32)
        for cue in cues[:12]:
            found = index.query(cue, k)
            scores = bank.astype(np.float64) @ cue.astype(np.float64)
            exact = np.lexsort((np.arange(len(bank)), -scores))[:k]
            assert np.array_equal(found, exact), "Full-probe exact retrieval failed"
        index.set_query_arguments(probes=4)
        begin = time.perf_counter()
        hits = 0
        for cue in cues:
            found = index.query(cue, k)
            scores = bank.astype(np.float64) @ cue.astype(np.float64)
            exact = np.lexsort((np.arange(len(bank)), -scores))[:k]
            hits += len(set(found.tolist()).intersection(set(exact.tolist())))
            assert len(set(found.tolist())) == k
        seconds = time.perf_counter() - begin
        print(f"SYNTHETIC_ROUTED_QPS {len(cues)/seconds:.2f}")
        print(f"SYNTHETIC_ROUTED_RECALL100 {hits/(len(cues)*k):.6f}")
        assert not index.witness(0, bank[0] + 1)
        index2 = HingeZeroIndexed("ip", centroid_count=32, field_size=64)
        index2.load_memory(path)
        index2.set_query_arguments(probes=32)
        for cue in cues[:4]:
            found = index2.query(cue, k)
            scores = bank.astype(np.float64) @ cue.astype(np.float64)
            exact = np.lexsort((np.arange(len(bank)), -scores))[:k]
            assert np.array_equal(found, exact)
    print("HINGEZERO_INDEXED_SMOKE_PASS")


if __name__ == "__main__":
    main()
