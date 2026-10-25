import os
from pathlib import Path
import numpy as np
from vibe.algorithms.hingezero_learning.module import HingeZeroLearnedRouting

def main():
  rng = np.random.default_rng(19)
  X=np.ascontiguousarray(rng.normal(size=(200,128)),dtype=np.float32)
  queries=X[:16].copy()
  labels=np.argsort(-(queries.astype(np.float64)@X.astype(np.float64).T),axis=1)[:,:100].astype(np.int32)
  alg=HingeZeroLearnedRouting("ip", graph_degree=12, build_ef=64)
  alg.fit_ood(X,queries,labels)
  alg.set_query_arguments(4,32)
  q=queries[0]
  result=alg.query(q,100)
  expected=np.argsort(-(X.astype(np.float64)@q.astype(np.float64)),stable=True)[:100]
  assert set(result.tolist())==set(expected.tolist()),"candidate expansion missed exact known neighbours"
  assert all(alg.witness(i,alg.read_record(i)) for i in result[:3])
  print("HINGEZERO LEARNED ROUTING SMALL SMOKE PASS")
if __name__=="__main__": main()
