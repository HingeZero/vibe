"""Research-only, out-of-sample measurement of exact MIPS prefix pruning."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import h5py
import numpy as np
import time

def quantile_summary(values):
    a=np.asarray(values,dtype=np.float64)
    return f"mean={a.mean():.0f} median={np.median(a):.0f} min={a.min():.0f} max={a.max():.0f}"

def main():
    fn=Path("data/yi-128-ip.hdf5")
    if not fn.exists():
        raise FileNotFoundError(fn)
    t=time.perf_counter()
    with h5py.File(fn,"r") as f:
        print("DATASET KEYS",list(f.keys()),flush=True)
        bank=np.ascontiguousarray(f["train"][:],dtype=np.float32)
        query=np.ascontiguousarray(f["test"][:min(40,len(f["test"]))],dtype=np.float32)
    n,d=bank.shape
    print(f"DATA n={n} d={d} tested_queries={len(query)} build_s={time.perf_counter()-t:.1f}",flush=True)
    norm=np.linalg.norm(bank,axis=1)
    print("RECORD_NORM "+quantile_summary(norm),flush=True)
    print("QUERY_NORM "+quantile_summary(np.linalg.norm(query,axis=1)),flush=True)
    mu=bank.mean(axis=0).astype(np.float32)
    centered=bank-mu
    t=time.perf_counter()
    cov=(centered.T@centered)/float(n)
    values,basis=np.linalg.eigh(cov)
    order=np.argsort(values)[::-1]
    values=np.maximum(values[order],0)
    basis=np.ascontiguousarray(basis[:,order],dtype=np.float32)
    explained=np.cumsum(values)/max(np.sum(values),1e-20)
    for k in [4,8,16,32,48,64,96]:
        print(f"PCA_ENERGY p={k} total={explained[k-1]:.5f}",flush=True)
    print(f"PCA_FIT_S {time.perf_counter()-t:.1f}",flush=True)
    t=time.perf_counter()
    transformed=np.ascontiguousarray(centered@basis,dtype=np.float32)
    rot_query=np.ascontiguousarray(query@basis,dtype=np.float32)
    print(f"PCA_ROTATION_S {time.perf_counter()-t:.1f}",flush=True)
    tail=np.zeros((n,d+1),dtype=np.float32)
    # A suffix norm for a split p in [0,d].
    for j in range(d-1,-1,-1):
        tail[:,j]=tail[:,j+1]+transformed[:,j]*transformed[:,j]
    prefix_lengths=[8,16,24,32,48,64,96,112]
    survivors={p:[] for p in prefix_lengths}
    score_gaps=[]
    total=time.perf_counter()
    for qi,q in enumerate(rot_query):
        truth=(transformed@q).astype(np.float64)
        kth=float(np.partition(truth,-100)[-100])
        sorted_scores=np.partition(truth,[-102,-100,-99])
        gap=float(sorted_scores[-100]-sorted_scores[-101])
        score_gaps.append(gap)
        qtail=np.zeros(d+1,dtype=np.float64)
        for j in range(d-1,-1,-1):
            qtail[j]=qtail[j+1]+float(q[j])**2
        for p in prefix_lengths:
            partial=transformed[:,:p]@q[:p]
            bound=partial.astype(np.float64)+np.sqrt(tail[:,p].astype(np.float64)*qtail[p])
            # Widen to absorb float32 PCA projection and dot-product errors
            safe_epsilon=0.0001*(1+max(abs(kth),np.max(np.abs(truth))))
            candidate_count=int(np.count_nonzero(bound+safe_epsilon>=kth))
            survivors[p].append(candidate_count)
        if qi in [0,9,19,39]: print(f"PROGRESS query={qi+1} elapsed_s={time.perf_counter()-total:.1f}",flush=True)
    print("TOP100_SCORE_GAP "+quantile_summary(score_gaps),flush=True)
    for p in prefix_lengths:
        counts=np.array(survivors[p])
        print(f"PCA_CERTIFY p={p} "+quantile_summary(counts)+f" fraction={np.mean(counts)/n:.5f}",flush=True)
    print(f"TOTAL_RESEARCH_S {time.perf_counter()-t:.1f}",flush=True)

if __name__=="__main__":
    main()
