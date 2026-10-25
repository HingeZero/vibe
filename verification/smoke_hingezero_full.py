"""Correctness and persistence/append regression test for standalone HingeZero."""
import hashlib
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from vibe.algorithms.hingezero_full.module import HingeZeroFullMemory, HingeZeroPackedControl

rng=np.random.default_rng(7789)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    for metric in ("ip", "cosine", "euclidean", "normalized"):
        with tempfile.TemporaryDirectory() as folder:
            a=rng.normal(size=(512,32)).astype(np.float32)
            if metric=="normalized":
                a/=np.linalg.norm(a,axis=1,keepdims=True)
            hz=HingeZeroFullMemory(metric,storage_path=folder)
            hz.fit(a)
            original=hz.bank.raw_file
            old=original.read_bytes()
            old_hash=digest(original)
            result=hz.append_memories(a[:32],sources=["source-"+str(i) for i in range(32)])
            assert list(result)==list(range(512,544))
            assert original.read_bytes()[:len(old)]==old
            assert old_hash==hashlib.sha256(original.read_bytes()[:len(old)]).hexdigest()
            assert hz.bank.source(512)=="source-0"
            assert hz.bank.count==544
            assert hz.bank.verify(0,a[0])
            altered=a[0].copy();altered[0]+=0.01
            assert not hz.bank.verify(0,altered)
            restored=HingeZeroFullMemory(metric).load_memory(folder)
            assert restored.bank.count==544
            assert restored.bank.verify(512,a[0])
            assert restored.bank.source(512)=="source-0"
            restored.set_query_arguments(544,0,8)
            cue=rng.normal(size=32).astype(np.float32)
            if metric=="normalized":
                cue/=np.linalg.norm(cue)
            got=restored.query(cue,100)
            records=np.concatenate((a,a[:32]),axis=0).astype(np.float64)
            x=cue.astype(np.float64)
            if metric=="euclidean":
                scores=-np.sum((records-x)**2,axis=1)
            elif metric=="cosine":
                scores=(records@x)/(np.linalg.norm(records,axis=1)*np.linalg.norm(x))
            else:
                scores=records@x
            truth=np.lexsort((np.arange(len(records)),-scores))[:100]
            assert np.array_equal(got,truth),f"{metric}: all-bank rerank not exact"
            assert restored.get_additional()["witness_failures"]==0
            restored.set_query_arguments(256,256,8)
            part=restored.query(cue,100)
            assert len(part)==len(set(part))==100
            assert np.all((part>=0)&(part<544))
            record=restored.memory_recall(cue,min_similarity=float("inf"))
            assert record["status"]=="no_match"
            ctrl=HingeZeroPackedControl(metric)
            ctrl.load_memory(folder)
            ctrl.set_query_arguments(256,256,8)
            compare=ctrl.query(cue,100)
            assert len(compare)==100
            from vibe.algorithms.hingezero_full.module import WitnessedState, RetainingSession
            state=WitnessedState(0,a[0])
            session=RetainingSession(restored,state)
            step=session.step(disturbance=np.full(32,4,dtype=np.float32))
            assert step["rolled_back"] and step["record_id"]==0
            print(metric,"PASS",{"packed_bytes":restored.bank.sign_file.stat().st_size,
                                 "raw_bytes":restored.bank.raw_file.stat().st_size})
    print("HINGEZERO FULL MEMORY SMOKE PASS")


if __name__=="__main__":
    run()
