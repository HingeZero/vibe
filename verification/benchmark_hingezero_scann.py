import ctypes as ct
import os
import tempfile
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import h5py
import numpy as np
import scann

from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20


def rank_native(mem, cue, found, k):
    candidates = np.ascontiguousarray(found, dtype=np.int32)
    out = np.empty(k, dtype=np.int32)
    q = np.ascontiguousarray(cue, dtype=np.float32)
    status = mem.lib.hz20_rank_subset(
        mem.handle, ct.c_void_p(q.ctypes.data),
        ct.c_void_p(candidates.ctypes.data), len(candidates),
        k, ct.c_void_p(out.ctypes.data)
    )
    if status:
        raise RuntimeError(mem._err())
    return out


def main():
    with h5py.File("data/yi-128-ip.hdf5", "r") as f:
        bank = np.asarray(f["train"], dtype=np.float32)
        cues = np.asarray(f["test"][:200], dtype=np.float32)
        truth = np.asarray(f["neighbors"][:200, :100], dtype=np.int32)
    with tempfile.TemporaryDirectory() as folder:
        mem = HingeZeroCpp20("ip", storage_path=folder)
        mem.fit(bank)
        mem.lib.hz20_rank_subset.argtypes = [
            ct.c_void_p, ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_int, ct.c_void_p
        ]
        mem.lib.hz20_rank_subset.restype = ct.c_int
        for leaf_count in (512, 1024):
            begin = time.perf_counter()
            engine = (
                scann.scann_ops_pybind.builder(bank, 100, "dot_product")
                .tree(leaf_count, 1, training_sample_size=len(bank),
                      spherical=True, quantize_centroids=True)
                .score_ah(2, anisotropic_quantization_threshold=0.1)
                .reorder(1)
                .build()
            )
            print("SCANN_BUILD", leaf_count, round(time.perf_counter() - begin, 2), flush=True)
            for probes in (2, 4, 8, 16, 32, 64, 128):
                for shortlist in (100, 200, 400, 800):
                    try:
                        engine.search(cues[0], shortlist, shortlist, probes)
                        begin = time.perf_counter()
                        hits = 0
                        for q, gt in zip(cues, truth):
                            ids, _ = engine.search(q, shortlist, shortlist, probes)
                            if len(ids) != shortlist:
                                raise RuntimeError("Short candidate results")
                            out = rank_native(mem, q, ids, 100)
                            hits += len(set(out.tolist()) & set(gt.tolist()))
                        elapsed = time.perf_counter() - begin
                        print("SCANN_HZ", "LEAVES", leaf_count, "PROBES", probes,
                              "SHORTLIST", shortlist, "RECALL_ID", round(hits / 20000, 6),
                              "QPS", round(len(cues) / elapsed, 2), flush=True)
                    except Exception as exc:
                        print("CONFIG_FAILED", leaf_count, probes, shortlist, str(exc), flush=True)
    print("SCANN_HZ_COMPLETE")


if __name__ == "__main__":
    main()
