"""Unit tests: HingeZero expected O(1) exact recall + unchanged similarity ranking."""
import hashlib
import os
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20

def main():
    rng=np.random.default_rng(992814)
    for metric in ["ip","normalized","euclidean","cosine"]:
        with tempfile.TemporaryDirectory() as root:
            memory=rng.normal(size=(4096,128)).astype(np.float32)
            if metric=="normalized":
                memory/=np.linalg.norm(memory,axis=1,keepdims=True)
            # force duplicates to prove deterministic oldest-ID lookup
            memory[192]=memory[17]
            hz=HingeZeroCpp20(metric,storage_path=root)
            hz.fit(memory)
            for i in [0,17,511,2048,4095]:
                hit=hz.lookup_exact(memory[i])
                assert hit is not None and hit["verified"]
                assert hit["id"]==i, (metric,i,hit["id"])
                assert np.array_equal(hit["vector"].view(np.uint8),
                                      memory[i].view(np.uint8))
                assert np.array_equal(hz.lookup_by_id(i).view(np.uint8),
                                      memory[i].view(np.uint8))
            assert hz.lookup_exact(memory[192])["id"]==17
            assert np.array_equal(hz.lookup_by_id(192),memory[17])
            missing=memory[17].copy()
            missing[0]=np.nextafter(missing[0],np.float32(np.inf))
            assert hz.lookup_exact(missing) is None
            for j in [-1,4096]:
                try: hz.lookup_by_id(j)
                except IndexError: pass
                else: raise AssertionError("Out-of-range ID accepted")
            prefix=hashlib.sha256((Path(root)/"original-f32.bin").read_bytes()).hexdigest()
            add=np.array([[2.3]*128,[1.7]*128],dtype=np.float32)
            ids=hz.append_memories(add,sources=["new-a","new-b"])
            assert ids.tolist()==[4096,4097]
            assert hashlib.sha256((Path(root)/"original-f32.bin").read_bytes()[:4096*128*4]).hexdigest()==prefix
            assert hz.lookup_exact(add[0])["id"]==4096
            assert hz.lookup_exact(add[1])["source"]=="new-b"
            old=HingeZeroCpp20(metric).load_memory(root)
            assert old.lookup_exact(memory[17])["id"]==17
            assert old.lookup_exact(add[0])["id"]==4096
            # Nearest neighbor still performs true k-NN, not identity bypass.
            cue=rng.normal(size=128).astype(np.float32)
            if metric=="normalized": cue/=np.linalg.norm(cue)
            ids=old.query(cue,100)
            bank=np.vstack([memory,add]).astype(np.float64)
            q=cue.astype(np.float64)
            if metric=="euclidean": score=-np.sum((bank-q)**2,axis=1)
            elif metric=="cosine":
                den=np.linalg.norm(bank,axis=1)*np.linalg.norm(q)
                score=(bank@q)/den
            else: score=bank@q
            expected=np.lexsort((np.arange(len(bank)),-score))[:100]
            assert np.array_equal(ids,expected),metric+" nearest-neighbor query was altered"
            assert old.get_additional()["steps"]==0
            print(metric, "HYBRID_EXACT_AND_TOPK_PASS")
    print("HINGEZERO_HYBRID_O1_SMOKE_PASS")

if __name__=="__main__":main()
