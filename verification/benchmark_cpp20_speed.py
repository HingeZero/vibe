"""Same runner/hardware C++20 scalar baseline vs AVX2 inner-product speed.

This diagnostic is not an additional VIBE submission or index configuration.
Run using the identical stored float32 bank and queries; verify exact IDs.
"""
import ctypes as ct
import os
from pathlib import Path
import statistics
import tempfile
import time
import numpy as np


def bind(path):
    lib=ct.CDLL(str(path))
    lib.hz20_abi.restype=ct.c_int
    lib.hz20_error.restype=ct.c_char_p
    lib.hz20_open.argtypes=[ct.c_char_p,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int]
    lib.hz20_open.restype=ct.c_void_p
    lib.hz20_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int32,ct.c_int,ct.c_void_p,ct.c_void_p]
    lib.hz20_query.restype=ct.c_int
    lib.hz20_close.argtypes=[ct.c_void_p]
    assert lib.hz20_abi()==1
    return lib

def run():
    rng=np.random.default_rng(20261025)
    n,d,k,qn=65536,128,100,192
    memory=rng.normal(size=(n,d)).astype("<f4")
    queries=rng.normal(size=(qn,d)).astype(np.float32)
    runs=[]
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/"vibe-float32.bin"
        memory.tofile(path)
        for name,library_path in [
            ("scalar-reference",os.environ["HZ20_SCALAR_LIB"]),
            ("runtime-avx2",os.environ["HINGEZERO_CPP20_LIB"])
        ]:
            lib=bind(library_path)
            handle=lib.hz20_open(os.fsencode(path),n,d,1,256)
            if not handle:raise RuntimeError(name+" failed to open")
            outs=np.empty((qn,k),dtype=np.int32)
            timing=np.empty(3,dtype=np.float64)
            # Equal warm cache + initialisation on identical memory.
            for i in range(3):
                out=outs[0].copy()
                err=lib.hz20_query(handle,ct.c_void_p(queries[0].ctypes.data),k,0,
                        ct.c_void_p(out.ctypes.data),ct.c_void_p(timing.ctypes.data))
                if err:raise RuntimeError(name+" warmup native query failed")
            start=time.perf_counter()
            for i,q in enumerate(queries):
                err=lib.hz20_query(handle,ct.c_void_p(q.ctypes.data),k,0,
                        ct.c_void_p(outs[i].ctypes.data),ct.c_void_p(timing.ctypes.data))
                if err:raise RuntimeError(name+" measured native query failed")
            duration=time.perf_counter()-start
            lib.hz20_close(handle)
            runs.append((name,qn/duration,outs.copy()))
        assert np.array_equal(runs[0][2],runs[1][2]),"SIMD changed top-100 query IDs"
        print("SAME-HOST 0-STEP HINGEZERO C++20 SPEED TEST")
        print("Records:",n,"dimension:",d,"queries:",qn,"k:",k)
        print(f"Scalar C++20 QPS: {runs[0][1]:.2f}")
        print(f"Vectorised C++20 QPS: {runs[1][1]:.2f}")
        print(f"Same-machine speedup: {runs[1][1]/runs[0][1]:.2f}x")
        print("Top-100 output exact parity: PASS")

if __name__=="__main__":
    run()
