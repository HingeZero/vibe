"""Same-run comparison: proven old AVX2 C++20 vs candidate AVX-512 C++20.

No additional VIBE submission; both use HingeZero with steps=0.
"""
import ctypes as ct
import os
from pathlib import Path
import tempfile
import time
import numpy as np

def bind(path):
    x=ct.CDLL(str(path))
    x.hz20_open.argtypes=[ct.c_char_p,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int]
    x.hz20_open.restype=ct.c_void_p
    x.hz20_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int32,ct.c_int,ct.c_void_p,ct.c_void_p]
    x.hz20_query.restype=ct.c_int
    x.hz20_close.argtypes=[ct.c_void_p]
    return x

def run():
    rng=np.random.default_rng(20261025)
    count,d,queries,k=65536,128,256,100
    bank=rng.normal(size=(count,d)).astype("<f4")
    cues=rng.normal(size=(queries,d)).astype(np.float32)
    with tempfile.TemporaryDirectory() as root:
        f=Path(root)/"original-f32.bin"
        bank.tofile(f)
        outputs={}
        for label,libpath in [
            ("prior-avx2",os.environ["HZ20_PRIOR_LIB"]),
            ("new-avx512-dispatch",os.environ["HINGEZERO_CPP20_LIB"])
        ]:
            lib=bind(libpath)
            h=lib.hz20_open(os.fsencode(f),count,d,1,256)
            assert h,label+" bank failed to open"
            out=np.empty((queries,k),np.int32)
            stats=np.zeros(3,np.float64)
            for j in range(4):
                v=np.empty(k,np.int32)
                assert lib.hz20_query(h,ct.c_void_p(cues[0].ctypes.data),k,0,
                          ct.c_void_p(v.ctypes.data),ct.c_void_p(stats.ctypes.data))==0
            start=time.perf_counter()
            for j,cue in enumerate(cues):
                assert lib.hz20_query(h,ct.c_void_p(cue.ctypes.data),k,0,
                        ct.c_void_p(out[j].ctypes.data),ct.c_void_p(stats.ctypes.data))==0
            qps=queries/(time.perf_counter()-start)
            if label=="new-avx512-dispatch":
                lib.hz20_simd_level.restype=ct.c_int
                print("Candidate selected SIMD width:",lib.hz20_simd_level())
            outputs[label]=(qps,out.copy())
            lib.hz20_close(h)
        a,b=outputs["prior-avx2"],outputs["new-avx512-dispatch"]
        print(f"SAME-RUN PRIOR AVX2 QPS {a[0]:.2f}")
        print(f"SAME-RUN NEW AVX512 QPS {b[0]:.2f}")
        print(f"SAME-RUN SPEED RATIO {b[0]/a[0]:.3f}")
        print("EXACT TOP100 IDS MATCH",np.array_equal(a[1],b[1]))
        assert np.array_equal(a[1],b[1]),"New CPU path changed native top-100 results"
        print("HINGEZERO ZERO-STEP EXACT SIMD PARITY PASSED")

if __name__=="__main__":
    run()
