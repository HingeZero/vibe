"""Separately measure C ABI exact-key and direct ID lookup on VIBE's actual bank.

This is NOT the VIBE nearest-neighbor QPS result. Timings include querying
in-memory native mapped bank but exclude index construction, measured separately.
"""
import ctypes as ct
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import h5py
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20

def run():
    data=Path("data/yi-128-ip.hdf5")
    if not data.exists():
        raise FileNotFoundError("VIBE dataset is required for exact-retrieval benchmark")
    with h5py.File(data,"r") as f:
        bank=np.asarray(f["train"],dtype=np.float32)
    rng=np.random.default_rng(20101986)
    with tempfile.TemporaryDirectory() as root:
        hz=HingeZeroCpp20("ip",storage_path=root)
        begin=time.perf_counter()
        hz.fit(bank)
        build=time.perf_counter()-begin
        assert hz.steps==0
        native=hz.lib
        ptr=hz.handle
        trials=50000
        ids=rng.integers(0,len(bank),size=trials)
        samples=np.ascontiguousarray(bank[ids])
        output=ct.c_int32(-1)
        # Warm caches and JIT-less C++ native dispatch
        for i in range(256):
            rc=native.hz20_exact_lookup(ptr,ct.c_void_p(samples[i].ctypes.data),ct.byref(output))
            assert rc==1
        start=time.perf_counter()
        for i in range(trials):
            rc=native.hz20_exact_lookup(ptr,ct.c_void_p(samples[i].ctypes.data),ct.byref(output))
            assert rc==1 and 0<=output.value<len(bank)
        seconds=time.perf_counter()-start
        # Verify every result separately from the timed lookup throughput loop.
        batched=hz.lookup_exact_batch(samples)
        assert len(batched)==trials
        assert np.array_equal(bank[batched].view(np.uint8),samples.view(np.uint8))
        qps=trials/seconds
        start=time.perf_counter()
        batched=hz.lookup_exact_batch(samples)
        batch_qps=trials/(time.perf_counter()-start)
        assert np.array_equal(bank[batched].view(np.uint8),samples.view(np.uint8))
        read=np.empty(bank.shape[1],dtype=np.float32)
        start=time.perf_counter()
        for id in ids:
            rc=native.hz20_get_by_id(ptr,int(id),ct.c_void_p(read.ctypes.data))
            assert rc==1
        id_qps=trials/(time.perf_counter()-start)
        print("HINGEZERO HYBRID DIRECT LOOKUP REPORT")
        print(json.dumps({"dataset":"yi-128-ip","records":len(bank),
              "dimensions":bank.shape[1],"queries":trials,
              "hash_build_including_bank_fit_seconds":round(build,3),
              "single_call_ctypes_exact_key_qps":round(qps,2),
              "native_batched_exact_key_qps":round(batch_qps,2),
              "single_call_ctypes_direct_id_qps":round(id_qps,2),
              "note":"Native exact-key and ID API QPS; NOT VIBE top-100 nearest-neighbor QPS",
              "zero_refinement":hz.steps==0,
              "record_identity_verified":True},indent=2))
        assert qps>0 and id_qps>0
        print("HINGEZERO HYBRID DIRECT LOOKUP BENCHMARK PASS")

if __name__=="__main__":run()
