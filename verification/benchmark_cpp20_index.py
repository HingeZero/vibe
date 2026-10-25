"""Same runner test: prior exhaustive C++20 SIMD vs new C++20 indexed engine.
Both receive identical raw float32 records and queries, HingeZero steps=0.
"""
import ctypes as ct
import os
from pathlib import Path
import tempfile
import time
import numpy as np

def open_lib(filename, indexed):
    lib=ct.CDLL(filename)
    lib.hz20_open.argtypes=[ct.c_char_p,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int]
    lib.hz20_open.restype=ct.c_void_p
    sig=[ct.c_void_p,ct.c_void_p,ct.c_int32,ct.c_int]
    if indexed:sig.append(ct.c_int)
    lib.hz20_query.argtypes=sig+[ct.c_void_p,ct.c_void_p]
    lib.hz20_query.restype=ct.c_int
    lib.hz20_close.argtypes=[ct.c_void_p]
    return lib

def run():
    rng=np.random.default_rng(20261025)
    n,d,k,qn=65536,128,100,256
    records=rng.normal(size=(n,d)).astype("<f4")
    queries=rng.normal(size=(qn,d)).astype(np.float32)
    times={}
    outputs={}
    with tempfile.TemporaryDirectory() as root:
        path=Path(root)/"original-f32.bin"
        records.tofile(path)
        for label,env,idx,budget in [
            ("preindex-exhaustive","HZ20_PREINDEX_LIB",False,128),
            ("ivf-all-128","HINGEZERO_CPP20_LIB",True,128),
            ("ivf-64","HINGEZERO_CPP20_LIB",True,64),
            ("ivf-32","HINGEZERO_CPP20_LIB",True,32)
        ]:
            lib=open_lib(os.environ[env],idx)
            t0=time.perf_counter()
            h=lib.hz20_open(os.fsencode(path),n,d,1,256)
            build=time.perf_counter()-t0
            assert h,label+" failed to open"
            ids=np.empty((qn,k),dtype=np.int32)
            t=np.zeros(3,dtype=np.float64)
            for p in range(3):
                tmp=np.empty(k,np.int32)
                params=[h,ct.c_void_p(queries[0].ctypes.data),k,0]
                if idx:params.append(budget)
                params.extend([ct.c_void_p(tmp.ctypes.data),ct.c_void_p(t.ctypes.data)])
                assert lib.hz20_query(*params)==0
            start=time.perf_counter()
            for j,q in enumerate(queries):
                params=[h,ct.c_void_p(q.ctypes.data),k,0]
                if idx:params.append(budget)
                params.extend([ct.c_void_p(ids[j].ctypes.data),ct.c_void_p(t.ctypes.data)])
                assert lib.hz20_query(*params)==0
            elapsed=time.perf_counter()-start
            lib.hz20_close(h)
            outputs[label]=ids
            times[label]=(qn/elapsed,build)
        exact=outputs["preindex-exhaustive"]
        print("C++20 INDEX SAME-WORKER: n=65536 d=128 queries=256 k=100")
        for label in times:
            recall=float(np.mean([len(set(a)&set(b))/k for a,b in zip(outputs[label],exact)]))
            identical=np.array_equal(outputs[label],exact)
            print(f"{label}: {times[label][0]:.2f} QPS, {recall:.5f} recall@100 vs exact, exact_ID_parity={identical}, build_s={times[label][1]:.2f}")
        assert np.array_equal(exact,outputs["ivf-all-128"]), "Index changed the exact all-bank answer"
        print("INDEXED FULL-PROBE EXACT-PARITY PASS")

if __name__=="__main__":run()
