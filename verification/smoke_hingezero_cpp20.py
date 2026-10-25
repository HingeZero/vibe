"""C++20 source parity, append-only preservation, native witness, VIBE metrics."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20


def run():
    origexe = os.environ.get("HZ20_ORIGINAL_EXE")
    assert origexe and Path(origexe).is_file(), "Original C++20 executable must be built"
    with tempfile.TemporaryDirectory() as directory:
        original_bank=Path(directory)/"native-bipolar.bin"
        subprocess.run([origexe,"create",str(original_bank),"8192"],check=True,
                       capture_output=True,text=True)
        prefix_hash=hashlib.sha256(original_bank.read_bytes()).hexdigest()
        subprocess.run([origexe,"append",str(original_bank),"8192","256"],check=True,
                       capture_output=True,text=True)
        assert hashlib.sha256(original_bank.read_bytes()[:8192*16]).hexdigest()==prefix_hash
        chk=json.loads(subprocess.check_output(
            [origexe,"check",str(original_bank),"8448","417"]))
        assert chk["accepted"]
        # The official native SOURCE pattern generates exactly 16 bytes per row,
        # each bit is a bipolar signed dimension, scaled by 1/sqrt(128).
        raw=np.fromfile(original_bank,dtype=np.uint8).reshape(-1,16)
        bits=np.unpackbits(raw,axis=1,bitorder="little").astype(np.float32)
        vectors=(2*bits-1)*np.float32(0.08838834764831845)
        obj=HingeZeroCpp20("ip",storage_path=str(Path(directory)/"float-adaptation"))
        obj.fit(vectors)
        cue=vectors[417].copy()
        adapted=obj.refined_vector(cue,8)
        reference=json.loads(subprocess.check_output(
             [origexe,"parity",str(original_bank),"8448","417","exact"]))["refined"]
        max_error=float(np.max(np.abs(adapted - np.asarray(reference,dtype=np.float32))))
        assert max_error < 0.00005, f"C++20 original/refined parity failed: {max_error}"
        record=obj.read_record(417)
        assert obj.witness(417,record)
        bad=record.copy();bad[3]=0
        assert not obj.witness(417,bad)
        obj.set_query_arguments(0)
        exact=obj.query(cue,100)
        # Exact original-metric full scan with no HZ against source bipolar bank.
        expected=np.lexsort((np.arange(len(vectors)),
                      -(vectors.astype(np.float64)@cue.astype(np.float64))))[:100]
        assert np.array_equal(exact,expected)
        obj.set_query_arguments(8)
        refined_ids=obj.query(cue,100)
        assert len(refined_ids)==len(set(refined_ids))==100
        print("C++20 SOURCE PARITY PASS",{"max_refinement_error":max_error,
             "pattern_records":len(vectors),"original_binary_prefix_sha256":prefix_hash})
    gen=np.random.default_rng(20261025)
    for metric in ("ip","normalized","euclidean","cosine"):
        with tempfile.TemporaryDirectory() as root:
            bank=gen.normal(size=(1024,128)).astype(np.float32)
            if metric=="normalized":
                bank/=np.linalg.norm(bank,axis=1,keepdims=True)
            a=HingeZeroCpp20(metric,mode="raw",storage_path=root)
            a.fit(bank)
            original=Path(root)/"original-f32.bin"
            prefix_hash=hashlib.sha256(original.read_bytes()).hexdigest()
            new=gen.normal(size=(64,128)).astype(np.float32)
            if metric=="normalized":
                new/=np.linalg.norm(new,axis=1,keepdims=True)
            a.append_memories(new,sources=[f"new:{i}" for i in range(64)])
            combined=np.concatenate([bank,new],axis=0)
            assert hashlib.sha256(original.read_bytes()[:1024*128*4]).hexdigest()==prefix_hash
            assert a.count==1088 and a.source_map[1024]=="new:0"
            b=HingeZeroCpp20(metric,mode="raw").load_memory(root)
            assert b.witness(1024,new[0])
            altered=new[0].copy();altered[0]+=0.10
            assert not b.witness(1024,altered)
            q=gen.normal(size=128).astype(np.float32)
            if metric=="normalized":q/=np.linalg.norm(q)
            b.set_query_arguments(0)
            ids=b.query(q,100)
            v=combined.astype(np.float64);q64=q.astype(np.float64)
            if metric=="euclidean":
                scores=-np.sum((v-q64)**2,axis=1)
            elif metric=="cosine":
                den=np.linalg.norm(v,axis=1)*np.linalg.norm(q64)
                scores=(v@q64)/den
            else:
                scores=v@q64
            ground=np.lexsort((np.arange(len(v)),-scores))[:100]
            assert np.array_equal(ids,ground),f"native exact metric {metric} disagrees with ground truth"
            hz=HingeZeroCpp20(metric,mode="refined").load_memory(root)
            hz.set_query_arguments(8)
            refined=hz.query(q,100)
            assert len(refined)==len(set(refined))==100
            assert hz.witness(int(refined[0]),hz.read_record(int(refined[0])))
            print(metric,"C++20 SMOKE PASS")
    print("ALL HINGEZERO C++20 NATIVE TESTS PASSED")


if __name__=="__main__":
    run()
