import ctypes as ct
import os
from pathlib import Path
import statistics
import tempfile
import time
import h5py
import numpy as np

def load(path):
    obj=ct.CDLL(path)
    obj.hz20_open.argtypes=[ct.c_char_p,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int]
    obj.hz20_open.restype=ct.c_void_p
    obj.hz20_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int32,ct.c_int,ct.c_void_p,ct.c_void_p]
    obj.hz20_query.restype=ct.c_int
    obj.hz20_close.argtypes=[ct.c_void_p]
    if hasattr(obj,"hz20_set_budget"):
        obj.hz20_set_budget.argtypes=[ct.c_void_p,ct.c_int]
        obj.hz20_set_budget.restype=ct.c_int
    return obj

def main():
    with h5py.File("data/yi-128-ip.hdf5","r") as f:
        bank=np.ascontiguousarray(f["train"][:],dtype="<f4")
        queries=np.ascontiguousarray(f["test"][:min(500,len(f["test"]))],dtype=np.float32)
    n,d=bank.shape
    q8=load("/tmp/libhz20_q8.so")
    q16=load(os.environ["HINGEZERO_CPP20_LIB"])
    ref=load("/tmp/hz500_reference.so")
    with tempfile.TemporaryDirectory() as tmp:
        mem=Path(tmp)/"original-f32.bin"
        bank.tofile(mem)
        def start(lib,budget):
            h=lib.hz20_open(os.fsencode(mem),n,d,1,256)
            assert h
            if budget:
                assert lib.hz20_set_budget(h,budget)==0
            return h
        hs=[start(q8,256),start(q16,256),start(ref,None)]
        libs=[q8,q16,ref]
        counts={name:[] for name in ("q8","q16")}
        truths=None
        try:
            for name in ("q8","q16","ref","q16","q8"):
                li={"q8":0,"q16":1,"ref":2}[name]
                answers=np.empty((len(queries),100),dtype=np.int32)
                stats=np.zeros(3,dtype=np.float64)
                t=time.perf_counter()
                for i,q in enumerate(queries):
                    assert libs[li].hz20_query(
                        hs[li],ct.c_void_p(q.ctypes.data),
                        100,0,ct.c_void_p(answers[i].ctypes.data),
                        ct.c_void_p(stats.ctypes.data))==0
                qps=len(queries)/(time.perf_counter()-t)
                if name=="ref":
                    truths=answers.copy()
                else:
                    counts[name].append(qps)
                if name in ("q8","q16"):
                    if "stored" not in locals():stored={}
                    if name in stored:
                        assert np.array_equal(answers,stored[name])
                    stored[name]=answers
                print(f"HEAD2HEAD name={name} qps={qps:.2f} candidates={stats[2]:.1f}",flush=True)
            for name in ("q8","q16"):
                assert np.array_equal(stored[name],truths),f"top100 ranking diverged {name}"
                print(f"HEAD2HEAD_RESULT {name} median_qps={statistics.median(counts[name]):.2f} exact_parity=PASS",flush=True)
            print("HEAD2HEAD_RATIO split_over_q8="+f"{statistics.median(counts['q16'])/statistics.median(counts['q8']):.3f}",flush=True)
        finally:
            for lib,h in zip(libs,hs):lib.hz20_close(h)
if __name__=="__main__":main()
