"""Analyze learned-neighbor popularity on VIBE's OOD yi-128-ip, without using test labels for index construction."""
import h5py
import numpy as np
import time
from pathlib import Path

def main():
    file=Path("data/yi-128-ip.hdf5")
    with h5py.File(file,"r") as f:
        print("HOT KEYS",list(f.keys()),flush=True)
        print("HOT SHAPES",[(k, f[k].shape) for k in f.keys() if isinstance(f[k],h5py.Dataset)],flush=True)
        if not all(k in f for k in ["train","test","learn","learn_neighbors"]):
            print("HOT no OOD learn data; cannot use supervised hotspot model",flush=True)
            return
        n=len(f["train"])
        learn=f["learn_neighbors"][:]
        print(f"HOT learn_neighbours={learn.shape} n={n}",flush=True)
        if learn.ndim!=2 or len(learn)==0: return
        counts=np.bincount(learn.ravel().astype(np.int64),minlength=n)
        order=np.lexsort((np.arange(n,dtype=np.int32),-counts))
        available=int((counts>0).sum())
        print(f"HOT unique_learn_labels={available} of {n}, maximum_frequency={int(counts[order[0]])}",flush=True)
        cutoffs=[100,200,500,1000,2000,4000,8000,16000,32000,64000,128000]
        for k in cutoffs:
            k=min(k,n)
            print(f"HOT TRAIN_TOP_{k} coverage={counts[order[:k]].sum()/len(learn.ravel()):.6f}",flush=True)
        if "neighbors" in f:
            gold=np.asarray(f["neighbors"][:],dtype=np.int32)
            if len(gold) != len(f["test"]):
                print("HOT gold size mismatch",flush=True)
                return
            print(f"HOT TEST_GOLD shape={gold.shape}",flush=True)
            for k in cutoffs:
                k=min(k,n)
                flag=np.zeros(n,dtype=np.uint8)
                flag[order[:k]]=1
                matched=flag[gold]
                exact_recall=float(matched.mean())
                perquery=matched.mean(axis=1)
                print(f"HOT TEST_TOP_{k} oracle_upper_recall={exact_recall:.6f} query_min={perquery.min():.6f} queries_perfect={(perquery==1).sum()}/{len(gold)}",flush=True)
        else:
            print("HOT no ground truth available",flush=True)
        bank=np.ascontiguousarray(f["train"][:],dtype=np.float32)
        queries=np.ascontiguousarray(f["test"][:min(150,len(f["test"]))],dtype=np.float32)
    for k in [500,2000,8000,32000]:
        hot=np.ascontiguousarray(bank[order[:k]],dtype=np.float32)
        t=time.perf_counter()
        for q in queries[:10]:
            scores=hot@q
            _=np.argpartition(scores,-100)[-100:]
        start=time.perf_counter()
        for q in queries:
            scores=hot@q
            _=np.argpartition(scores,-100)[-100:]
        elapsed=time.perf_counter()-start
        print(f"HOT PY_NUMPY_SINGLE_BLAS n={k} qps={len(queries)/elapsed:.1f} elapsed={elapsed:.4f} (not VIBE-native benchmark)",flush=True)
if __name__=="__main__":main()
