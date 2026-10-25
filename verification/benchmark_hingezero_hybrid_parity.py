"""Compare pre-hybrid and hybrid C++20 nearest-neighbour engines on the SAME worker.

Uses VIBE's corpus and queries; no extra VIBE algorithm config or changed
benchmark answers. The hash map is only for explicit exact identity calls.
"""
import ctypes as ct
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import h5py

def make_native(path):
    lib=ct.CDLL(path)
    lib.hz20_open.argtypes=[ct.c_char_p,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int]
    lib.hz20_open.restype=ct.c_void_p
    lib.hz20_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int32,ct.c_int,ct.c_void_p,ct.c_void_p]
    lib.hz20_query.restype=ct.c_int
    lib.hz20_close.argtypes=[ct.c_void_p]
    return lib

def probe(lib,filename,n,d,cues,k=100):
    t0=time.perf_counter()
    ptr=lib.hz20_open(os.fsencode(filename),n,d,1,256)
    build=time.perf_counter()-t0
    assert ptr,"native memory opening failed"
    outs=np.empty((len(cues),k),dtype=np.int32)
    stats=np.zeros(3,np.float64)
    for j in range(4):
        dest=np.empty(k,np.int32)
        assert lib.hz20_query(ptr,ct.c_void_p(cues[0].ctypes.data),k,0,
               ct.c_void_p(dest.ctypes.data),ct.c_void_p(stats.ctypes.data))==0
    t=time.perf_counter()
    for j,cue in enumerate(cues):
        assert lib.hz20_query(ptr,ct.c_void_p(cue.ctypes.data),k,0,
            ct.c_void_p(outs[j].ctypes.data),ct.c_void_p(stats.ctypes.data))==0
    elapsed=time.perf_counter()-t
    lib.hz20_close(ptr)
    return build,len(cues)/elapsed,outs

def main():
    path=Path("data/yi-128-ip.hdf5")
    if not path.exists():raise FileNotFoundError("VIBE dataset missing")
    with h5py.File(path,"r") as f:
        memory=np.asarray(f["train"],dtype="<f4")
        queries=np.ascontiguousarray(f["test"][:],dtype=np.float32)
    n,d=memory.shape
    tests=queries[:min(256,len(queries))]
    a=make_native(os.environ["HZ20_BEFORE_HYBRID_LIB"])
    b=make_native(os.environ["HINGEZERO_CPP20_LIB"])
    with tempfile.TemporaryDirectory() as temp:
        bank=Path(temp)/"memory-f32.bin"
        memory.tofile(bank)
        # Interleaved ordering reduces cached-benchmark ordering effects.
        cases=[]
        for i in range(2):
            for name,lib in ([("prehybrid",a),("hybrid",b)] if i==0
                              else [("hybrid",b),("prehybrid",a)]):
                build,qps,ids=probe(lib,bank,n,d,tests)
                cases.append((name,build,qps,ids))
        reference=cases[0][3]
        for name,build,qps,ids in cases:
            assert np.array_equal(reference,ids),name+" changed top-100 result IDs"
            print(f"SAME-WORKER {name} QPS {qps:.2f} build_s {build:.3f}")
        pre=[x[2] for x in cases if x[0]=="prehybrid"]
        hybr=[x[2] for x in cases if x[0]=="hybrid"]
        print("SAME-WORKER PREHYBRID MEAN QPS",round(float(np.mean(pre)),2))
        print("SAME-WORKER HYBRID MEAN QPS",round(float(np.mean(hybr)),2))
        print("SAME-WORKER HYBRID/PREHYBRID RATIO",round(float(np.mean(hybr)/np.mean(pre)),3))
        print("N",n,"DIM",d,"K",100,"QUERIES",len(tests))
        print("EXACT TOP-100 PARITY PASS")
if __name__=="__main__":main()
