"""Verify submitted zero-iteration C++20 HingeZero, no alternate benchmark mode."""
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
    executable = Path(os.environ["HZ20_ORIGINAL_EXE"])
    assert executable.is_file(), "Original C++20 source wasn't compiled"
    with tempfile.TemporaryDirectory() as tmp:
        packed=Path(tmp)/"binary-bank.bin"
        subprocess.run([str(executable),"create",str(packed),"512"],check=True,capture_output=True)
        digest=hashlib.sha256(packed.read_bytes()).hexdigest()
        subprocess.run([str(executable),"append",str(packed),"512","64"],check=True,capture_output=True)
        assert hashlib.sha256(packed.read_bytes()[:512*16]).hexdigest()==digest
        rec=json.loads(subprocess.check_output([str(executable),"check",str(packed),"576","417"]))
        assert rec["accepted"], "Original tagged C++20 witness rejected a stored record"

    rng=np.random.default_rng(20261025)
    for metric in ("ip","normalized","euclidean","cosine"):
        with tempfile.TemporaryDirectory() as tmp:
            X=rng.normal(size=(1024,128)).astype(np.float32)
            if metric=="normalized": X/=np.linalg.norm(X,axis=1,keepdims=True)
            hz=HingeZeroCpp20(metric,storage_path=tmp)
            hz.fit(X)
            before=hashlib.sha256((Path(tmp)/"original-f32.bin").read_bytes()).hexdigest()
            extra=rng.normal(size=(64,128)).astype(np.float32)
            if metric=="normalized": extra/=np.linalg.norm(extra,axis=1,keepdims=True)
            added=hz.append_memories(extra,sources=[f"source:{i}" for i in range(64)])
            assert added.tolist()==list(range(1024,1088))
            prefix=(Path(tmp)/"original-f32.bin").read_bytes()[:1024*128*4]
            assert hashlib.sha256(prefix).hexdigest()==before, "Pre-existing memories modified"
            bank=np.vstack([X,extra]).astype(np.float64)
            reopened=HingeZeroCpp20(metric).load_memory(tmp)
            assert reopened.source_map[1024]=="source:0"
            assert reopened.witness(1024,extra[0])
            corrupted=extra[0].copy();corrupted[0]+=0.25
            assert not reopened.witness(1024,corrupted)
            cue=rng.normal(size=128).astype(np.float32)
            if metric=="normalized":cue/=np.linalg.norm(cue)
            reopened.set_query_arguments(0,208)
            assert np.array_equal(reopened.refined_vector(cue,0),cue), "zero-step modified initial cue"
            try:
                reopened.set_query_arguments(8)
                raise AssertionError("Invalid nonzero step configuration accepted")
            except ValueError:
                pass
            ids=reopened.query(cue,100)
            c=cue.astype(np.float64)
            if metric=="euclidean":
                scores=-np.sum((bank-c)**2,axis=1)
            elif metric=="cosine":
                scores=(bank@c)/(np.linalg.norm(bank,axis=1)*np.linalg.norm(c))
            else:
                scores=bank@c
            expected=np.lexsort((np.arange(len(bank)),-scores))[:100]
            assert np.array_equal(ids,expected),metric+" native zero-step search misranked"
            assert len(ids)==len(set(ids))==100
            assert reopened.witness(int(ids[0]),reopened.read_record(int(ids[0])))
            assert reopened.get_additional()["steps"]==0
            print(metric,"HINGEZERO_CPP20_ZERO_STEPS_PASS")
    print("SINGLE HINGEZERO CPP20 SUBMISSION SMOKE PASS")

if __name__=="__main__":
    run()
