import os
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import faiss
import h5py
import numpy as np


def main():
    faiss.omp_set_num_threads(1)
    with h5py.File("data/yi-128-ip.hdf5", "r") as f:
        bank = np.asarray(f["train"], dtype=np.float32)
        learn_ids = np.asarray(f["learn_neighbors"][:, :100], dtype=np.int32)
        q = np.asarray(f["test"][:200], dtype=np.float32)
        gt = np.asarray(f["neighbors"][:200, :100], dtype=np.int32)
    n,d = bank.shape
    freq = np.bincount(learn_ids.ravel(), minlength=n)
    ids = np.flatnonzero(freq)
    train_order = ids[np.lexsort((ids, -freq[ids]))]
    norms = np.einsum("ij,ij->i", bank, bank)
    norm_order = np.argsort(-norms, kind="stable").astype(np.int32)
    print("TRAIN_ACTIVE", len(train_order),flush=True)
    for frequency_budget in (32768,49152,65536,82797):
        base = train_order[:frequency_budget]
        for norm_budget in (0,2048,8192,16384,32768,65536):
            combined = np.unique(
                np.concatenate((base, norm_order[:norm_budget]))
            ).astype(np.int32)
            memory = np.ascontiguousarray(bank[combined])
            idx = faiss.IndexFlatIP(d)
            idx.add(memory)
            _, loc = idx.search(np.ascontiguousarray(q[:5]),100)
            start=time.perf_counter()
            found=np.empty((len(q),100),np.int32)
            for i,cue in enumerate(q):
                _,local =idx.search(np.ascontiguousarray(cue[None, :]),100)
                found[i]=combined[local[0]]
            duration=time.perf_counter()-start
            recall=np.mean([len(set(a.tolist()) & set(b.tolist()))/100 for a,b in zip(found,gt)])
            print("NORM_RESCUE", "FREQ", frequency_budget, "NORM", norm_budget,
                  "TOTAL",len(combined),"RECALL",round(float(recall),6),
                  "QPS",round(len(q)/duration,2),flush=True)
    print("NORM_RESCUE_FINISHED")


if __name__ == "__main__":
    main()
