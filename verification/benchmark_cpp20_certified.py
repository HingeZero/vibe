"""Same-host C++20 HingeZero exact SIMD versus certified bound-pruning.

One algorithm intended for submission (zero refinement), and a historical
reference library for validation only. No approximate candidate omission.
"""
import ctypes as ct
import os
from pathlib import Path
import tempfile
import time
import numpy as np

def native(path):
    lib=ct.CDLL(str(path))
    lib.hz20_open.argtypes=[ct.c_char_p,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int]
    lib.hz20_open.restype=ct.c_void_p
    lib.hz20_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int32,ct.c_int,ct.c_void_p,ct.c_void_p]
    lib.hz20_query.restype=ct.c_int
    lib.hz20_close.argtypes=[ct.c_void_p]
    return lib

def evaluate(lib,bank_path,records,queries,d,k):
    t=time.perf_counter()
    handle=lib.hz20_open(os.fsencode(bank_path),len(records),d,1,256)
    build=time.perf_counter()-t
    assert handle, "failed to open C++20 bank"
    out=np.empty((len(queries),k),np.int32)
    stat=np.empty(3,np.float64)
    for i in range(3):
        tmp=np.empty(k,np.int32)
        assert lib.hz20_query(handle,ct.c_void_p(queries[0].ctypes.data),
                 k,0,ct.c_void_p(tmp.ctypes.data),ct.c_void_p(stat.ctypes.data))==0
    exact_count=0
    start=time.perf_counter()
    for j,q in enumerate(queries):
        assert lib.hz20_query(handle,ct.c_void_p(q.ctypes.data),
                k,0,ct.c_void_p(out[j].ctypes.data),ct.c_void_p(stat.ctypes.data))==0
        exact_count+=int(stat[2])
    duration=time.perf_counter()-start
    lib.hz20_close(handle)
    return out,len(queries)/duration,build,exact_count/len(queries)

def run():
    rng=np.random.default_rng(20261025)
    results=[]
    for n,qn in [(65536,256),(187843,128)]:
        d,k=128,100
        train=rng.normal(size=(n,d)).astype("<f4")
        query=rng.normal(size=(qn,d)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"bank.bin"
            train.tofile(path)
            values={}
            for name,key in [("simd-exhaustive","HZ20_EXHAUSTIVE_LIB"),
                              ("certified-int16","HINGEZERO_CPP20_LIB")]:
                values[name]=evaluate(native(os.environ[key]),path,train,query,d,k)
            a=values["simd-exhaustive"]
            b=values["certified-int16"]
            assert np.array_equal(a[0],b[0]),"Certified pruning altered exact top-100 IDs"
            print(f"SAME-MACHINE HINGEZERO N={n} D=128 Q={qn} K=100")
            print(f"Exhaustive SIMD: {a[1]:.2f} QPS, build {a[2]:.3f}s")
            print(f"Certified int16: {b[1]:.2f} QPS, build {b[2]:.3f}s")
            print(f"Pruned average exact comparisons: {b[3]:.1f} of {n} records; {100*(1-b[3]/n):.2f}% skipped")
            print(f"Measured same-worker speedup: {b[1]/a[1]:.2f}x")
            print("EXACT TOP-100 ID PARITY PASS")
    print("CERTIFIED C++20 ZERO-STEP BENCHMARK PASS")

if __name__=="__main__":
    run()
