"""Diagnostic: use VIBE's permitted learn split only; never read test neighbours."""
import h5py
import numpy as np
from pathlib import Path

def main():
    with h5py.File("data/yi-128-ip.hdf5","r") as f:
        nbr=np.asarray(f["learn_neighbors"],dtype=np.int32)
        n_corpus=len(f["train"])
        n_learn=len(f["learn"])
    print(f"OOD_LABELS learn_queries={n_learn} shape={nbr.shape} corpus={n_corpus}",flush=True)
    assert nbr.shape==(n_learn,100)
    assert np.all(nbr>=0) and np.all(nbr<n_corpus)
    rng=np.random.default_rng(1159)
    perm=rng.permutation(n_learn)
    split=int(n_learn*0.70)
    train=nbr[perm[:split]]
    validation=nbr[perm[split:]]
    counts=np.bincount(train.ravel(),minlength=n_corpus)
    ranked=np.argsort(counts,kind="stable")[::-1]
    print(f"OOD_LABELS training_positive_unique={np.count_nonzero(counts)}",flush=True)
    for take in [100,256,512,1024,2048,4096,8192,16384,32768]:
        selected=np.zeros(n_corpus,dtype=np.bool_)
        selected[ranked[:take]]=True
        per_query=selected[validation].sum(axis=1)
        print(f"OOD_HOT_SIZE={take} recall_mean={per_query.mean()/100:.6f} min_recall={per_query.min()/100:.4f} p10_recall={np.percentile(per_query,10)/100:.4f} fully_hit_rate={np.mean(per_query==100):.6f}",flush=True)
    print(f"OOD_LABELS train_top_id_frac={counts.max() / len(train):.4f} distinct_val_labels={len(np.unique(validation))}",flush=True)

if __name__=="__main__":main()
