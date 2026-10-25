"""Diagnose whether low-dimensional PCA supports certified exact pruning.

Training only: never use benchmark test queries or neighbor ground truth
to choose parameters. No algorithm modifications or benchmark substitutions.
"""
import h5py
import numpy as np
from pathlib import Path

def run():
    path=Path("data/yi-128-ip.hdf5")
    assert path.exists(), "VIBE dataset isn't cached"
    with h5py.File(path,"r") as f:
        x=np.asarray(f["train"],dtype=np.float64)
    n,d=x.shape
    rng=np.random.default_rng(20261025)
    ids=rng.choice(n,size=min(4096,n),replace=False)
    subset=x[ids]
    mean=np.mean(subset,axis=0)
    _,singular,vt=np.linalg.svd(subset-mean,full_matrices=False)
    var=singular**2
    print("PCA-CERTIFIED TRAIN-ONLY DIAGNOSTIC: N",n,"D",d)
    print("PCA explained variance ratios",[(i,round(float(sum(var[:i])/sum(var)),5)) for i in [8,16,32,64,96,128]])
    centered=x-mean
    norms=np.sum(centered**2,axis=1)
    trials=12
    # Tests on training vectors, NEVER VIBE's held-out evaluation queries.
    qids=rng.choice(n,size=trials,replace=False)
    qtrain=x[qids]
    exact100=[]
    for q in qtrain:
        score=x@q
        exact100.append(np.partition(score,-100)[-100])
    for rank in [8,16,32,64,96,112]:
        basis=vt[:rank]
        proj=centered@basis.T
        residual=np.sqrt(np.maximum(0,norms-np.sum(proj*proj,axis=1)))
        counts=[]
        for q,threshold in zip(qtrain,exact100):
            qp=basis@q
            qr=np.sqrt(max(0,float(q@q)-float(qp@qp)))
            upper=proj@qp+residual*qr+mean@q+1e-6*(1+abs(threshold))
            counts.append(int(np.count_nonzero(upper>=threshold)))
        print("PCA-CERT",rank,"avg survivor count",round(float(np.mean(counts)),1),
              "survivor fraction",round(float(np.mean(counts))/n,5),
              "max survivor count",max(counts),"min",min(counts))
    print("PCA CERT DIAGNOSTIC PASS")

if __name__=="__main__":run()
