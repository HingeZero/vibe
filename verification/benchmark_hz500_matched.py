import ctypes as ct
import os
import tempfile
import time
from pathlib import Path
import h5py
import numpy as np

def open_native(file):
    obj=ct.CDLL(file)
    obj.hz20_open.argtypes=[ct.c_char_p,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int]
    obj.hz20_open.restype=ct.c_void_p
    obj.hz20_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int32,ct.c_int,ct.c_void_p,ct.c_void_p]
    obj.hz20_query.restype=ct.c_int
    if hasattr(obj,'hz20_set_budget'):
        obj.hz20_set_budget.argtypes=[ct.c_void_p,ct.c_int]
        obj.hz20_set_budget.restype=ct.c_int
    obj.hz20_close.argtypes=[ct.c_void_p]
    obj.hz20_close.restype=None
    return obj

def run(lib,path,n,d,queries,k=100,budget=None):
    t=time.perf_counter()
    handle=lib.hz20_open(os.fsencode(path),n,d,1,256)
    build=time.perf_counter()-t
    assert handle
    if budget is not None:
        assert lib.hz20_set_budget(handle,int(budget))==0
    answers=np.empty((len(queries),k),np.int32)
    stats=np.zeros(3,np.float64)
    try:
        for i in range(4):
            out=np.empty(k,np.int32)
            assert lib.hz20_query(handle,ct.c_void_p(queries[0].ctypes.data),
                                  k,0,ct.c_void_p(out.ctypes.data),
                                  ct.c_void_p(stats.ctypes.data))==0
        t=time.perf_counter()
        evaluations=0
        for i,q in enumerate(queries):
            rc=lib.hz20_query(handle,ct.c_void_p(q.ctypes.data),
                        k,0,ct.c_void_p(answers[i].ctypes.data),
                        ct.c_void_p(stats.ctypes.data))
            assert rc==0
            evaluations+=stats[2]
        dur=time.perf_counter()-t
    finally:
        lib.hz20_close(handle)
    return len(queries)/dur, build, answers, evaluations/len(queries)

def main():
    file=Path("data/yi-128-ip.hdf5")
    if not file.is_file():raise FileNotFoundError("Expected VIBE data for matched comparison")
    with h5py.File(file,"r") as f:
        bank=np.asarray(f["train"],dtype="<f4")
        queries=np.ascontiguousarray(f["test"][:],dtype=np.float32)
    n,d=bank.shape
    qs=queries[:min(len(queries),300)]
    old=open_native(os.environ["HZ500_REFERENCE_LIB"])
    new=open_native(os.environ["HINGEZERO_CPP20_LIB"])
    with tempfile.TemporaryDirectory() as tmp:
        mem=Path(tmp)/"original-f32.bin"
        bank.tofile(mem)
        results=[]
        for name,lib in [("ref",old),("new",new),("new",new),("ref",old)]:
            qps,build,ids,candidates=run(lib,mem,n,d,qs,budget=208 if name=='new' else None)
            results.append((name,qps,build,ids,candidates))
            print(f"MATCHED {name} QPS {qps:.2f} build_s {build:.2f} fullscore_avg {candidates:.1f}")
        exact=results[0][3]
        for name,_,_,ids,_ in results:
            assert np.array_equal(ids,exact),f"{name} changed top100 ranked IDs"
        baseline=np.mean([v[1] for v in results if v[0]=="ref"])
        experimental=np.mean([v[1] for v in results if v[0]=="new"])
        print(f"MATCHED HINGEZERO QPS reference={baseline:.2f} experimental={experimental:.2f} ratio={experimental/baseline:.3f} queries={len(qs)}")
        print("MATCHED TOP100 EXACT PARITY PASS")
if __name__=="__main__":main()
