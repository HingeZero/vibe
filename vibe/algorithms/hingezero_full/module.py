"""HingeZero standalone packed-memory backend for VIBE.

No FAISS: sign-packed binary scanning supplies candidate identities, HingeZero's
dual-tanh associative field optionally routes a second search, and original
float32 records determine the final benchmark metric. The on-disk bank supports
verified read-only reload and incremental append. No fabricated accuracy claims.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..base.module import BaseANN


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    length = float(np.linalg.norm(x))
    return np.ascontiguousarray(x / length if length > 1e-12 else np.zeros_like(x))


def hinge_phi(h, alpha=4.5):
    return np.tanh(h) + float(alpha) * np.tanh(2.0 * h)


class DiskPackedMemory:
    """Append-only durable raw float vectors plus 1-bit sign signatures.

    Signatures are a retrieval index, not lossless compression of raw floats.
    We store exact raw vectors on disk to preserve correct VIBE metrics.
    Metadata is plain text and NOT encrypted or access-controlled.
    """
    def __init__(self, root, dimension=None, source_tag="vibe", reopen=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.json"
        self.source_tag = str(source_tag)
        if reopen:
            info = json.loads(self.manifest.read_text(encoding="utf-8"))
            if info.get("version") != 1:
                raise ValueError("Unsupported memory manifest")
            self.dimension = int(info["dimension"])
            self.count = int(info["count"])
            self.source_tag = info["source_tag"]
        else:
            if self.manifest.exists():
                raise FileExistsError("Existing memory bank cannot be overwritten")
            if not dimension or int(dimension) < 1:
                raise ValueError("Invalid vector dimension")
            self.dimension = int(dimension)
            self.count = 0
            self._write_manifest()
        self.byte_width = (self.dimension + 7) // 8
        self.raw_file = self.root / "original-f32.bin"
        self.sign_file = self.root / "sign-1bit.bin"
        self.norm_file = self.root / "norm-f32.bin"
        self.meta_file = self.root / "provenance.jsonl"
        self.sources = {}
        self.evidence = {}
        if self.meta_file.exists():
            for line in self.meta_file.read_text(encoding="utf-8").splitlines():
                obj = json.loads(line)
                self.sources[int(obj["id"])] = obj["source"]
                self.evidence[int(obj["id"])] = float(obj.get("evidence", 0))
        self._mount()

    def _write_manifest(self):
        temp = self.manifest.with_suffix(".json.tmp")
        temp.write_text(json.dumps({
            "version": 1, "dimension": self.dimension,
            "count": self.count, "source_tag": self.source_tag,
            "index": "one-bit sign scan", "raw_precision": "float32"
        }, sort_keys=True), encoding="utf-8")
        os.replace(temp, self.manifest)

    def _mount(self):
        self.raw = self.packed = self.norms = None
        if not self.count:
            return
        expected = [
            (self.root / "original-f32.bin", self.count * self.dimension * 4),
            (self.root / "sign-1bit.bin", self.count * self.byte_width),
            (self.root / "norm-f32.bin", self.count * 4),
        ]
        for path, nbytes in expected:
            if not path.is_file() or path.stat().st_size != nbytes:
                raise IOError("Stored memory size mismatch: " + str(path))
        self.raw = np.memmap(self.raw_file, mode="r", dtype="<f4",
                             shape=(self.count, self.dimension))
        self.packed = np.memmap(self.sign_file, mode="r", dtype=np.uint8,
                                shape=(self.count, self.byte_width))
        self.norms = np.memmap(self.norm_file, mode="r", dtype="<f4",
                               shape=(self.count,))

    def append(self, rows, sources=None, evidence=None):
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.dimension or not len(values):
            raise ValueError("Invalid append shape")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite memories prohibited")
        if self.count + len(values) > np.iinfo(np.int32).max:
            raise ValueError("VIBE int32 neighbour-ID limit exceeded")
        if sources is not None and len(sources) != len(values):
            raise ValueError("Source metadata length mismatch")
        if evidence is not None and (len(evidence) != len(values) or
                                     not np.isfinite(np.asarray(evidence, dtype=float)).all()):
            raise ValueError("Evidence metadata invalid")
        previous = self.count
        self.raw = self.packed = self.norms = None  # release old readonly views
        with open(self.raw_file, "ab") as f:
            f.write(np.ascontiguousarray(values, dtype="<f4").tobytes())
            f.flush()
            os.fsync(f.fileno())
        bits = np.packbits(values >= 0, axis=1, bitorder="little")
        with open(self.sign_file, "ab") as f:
            f.write(bits.tobytes())
            f.flush()
            os.fsync(f.fileno())
        lengths = np.linalg.norm(values.astype(np.float64), axis=1).astype("<f4")
        with open(self.norm_file, "ab") as f:
            f.write(lengths.tobytes())
            f.flush()
            os.fsync(f.fileno())
        if sources is not None or evidence is not None:
            with open(self.meta_file, "a", encoding="utf-8") as f:
                for i in range(len(values)):
                    identifier = previous + i
                    item = {
                        "id": identifier,
                        "source": str(sources[i]) if sources is not None else
                                  f"{self.source_tag}:{identifier}",
                        "evidence": float(evidence[i]) if evidence is not None else 0.0,
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    self.sources[identifier] = item["source"]
                    self.evidence[identifier] = item["evidence"]
        self.count += len(values)
        self._write_manifest()
        self._mount()
        return np.arange(previous, self.count, dtype=np.int64)

    def source(self, identifier):
        identifier = int(identifier)
        return self.sources.get(identifier, f"{self.source_tag}:{identifier}")

    def get(self, identifiers):
        ids = np.asarray(identifiers, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.count):
            raise IndexError("Unknown memory record")
        return np.asarray(self.raw[ids], dtype=np.float32).copy()

    def verify(self, identifier, original):
        """Witness attests exact stored-state membership, not factual correctness."""
        if not 0 <= int(identifier) < self.count:
            return False
        x = np.asarray(original, dtype=np.float32)
        return x.shape == (self.dimension,) and np.array_equal(self.raw[int(identifier)], x)

    def exact_scores(self, identifiers, query, metric):
        values = self.get(identifiers).astype(np.float64)
        q = np.asarray(query, dtype=np.float64)
        dot = values @ q
        if metric == "euclidean":
            return -(np.einsum("ij,ij->i", values, values) +
                     float(q @ q) - 2.0 * dot)
        if metric == "cosine":
            denom = np.linalg.norm(values, axis=1) * np.linalg.norm(q)
            return np.divide(dot, denom, out=np.zeros(len(values)), where=denom > 1e-12)
        return dot  # inner product and normalized data

    @staticmethod
    def _top_ids(scores, ids, count):
        take = min(count, len(scores))
        if take < len(scores):
            indexes = np.argpartition(-scores, take - 1)[:take]
        else:
            indexes = np.arange(len(scores))
        order = np.lexsort((ids[indexes], -scores[indexes]))
        return ids[indexes[order]], scores[indexes[order]]

    def packed_candidates(self, query, count, metric, chunk=65536):
        """One-bit full-bank scan: O(P * ceil(d/8)), bounded work memory."""
        if self.count < 1:
            raise RuntimeError("Memory bank is empty")
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dimension,):
            raise ValueError("Query dimensions mismatch")
        bits = np.packbits((q >= 0)[None, :], axis=1, bitorder="little")[0]
        query_norm = float(np.linalg.norm(q.astype(np.float64)))
        wanted = min(max(int(count), 1), self.count)
        top_ids = np.empty(0, dtype=np.int64)
        top_scores = np.empty(0, dtype=np.float64)
        for start in range(0, self.count, chunk):
            end = min(start + chunk, self.count)
            xor = np.bitwise_xor(self.packed[start:end], bits[None, :])
            hamming = np.bitwise_count(xor).sum(axis=1, dtype=np.int32)
            cosine_est = 1.0 - 2.0 * hamming.astype(np.float64) / self.dimension
            if self.dimension % 8:
                # Pad bits were set to zero in both stored and query signatures
                pass
            if metric == "cosine":
                coarse = cosine_est
            else:
                nrm = np.asarray(self.norms[start:end], dtype=np.float64)
                approx_ip = cosine_est * nrm * query_norm
                coarse = -(nrm*nrm + query_norm*query_norm - 2*approx_ip) \
                    if metric == "euclidean" else approx_ip
            ids = np.arange(start, end, dtype=np.int64)
            local_ids, local_scores = self._top_ids(coarse, ids, wanted)
            all_ids = np.concatenate((top_ids, local_ids))
            all_scores = np.concatenate((top_scores, local_scores))
            top_ids, top_scores = self._top_ids(all_scores, all_ids, wanted)
        return top_ids


class WitnessedState:
    def __init__(self, record_id, vector):
        self.record_id = int(record_id)
        self.vector = np.asarray(vector, np.float32).copy()


class RetainingSession:
    """Optional sequential memory interface; NOT shared between VIBE queries."""
    def __init__(self, backend, state, drift_boundary=0.15, commit_margin=0.10):
        if not backend.verify(state.record_id, state.vector):
            raise ValueError("Unverified initial checkpoint")
        self.backend = backend
        self.checkpoint = state
        self.working = unit(state.vector)
        self.drift_boundary = float(drift_boundary)
        self.commit_margin = float(commit_margin)

    def step(self, cue=None, disturbance=None):
        if cue is not None and disturbance is not None:
            raise ValueError("Provide cue or disturbance, not both")
        committed = rolled_back = False
        if cue is not None:
            record = self.backend.memory_recall(cue, min_similarity=-np.inf,
                                                 min_margin=-np.inf)
            if record["status"] == "found" and record["margin"] >= self.commit_margin:
                identifier = record["id"]
                vector = self.backend.bank.get([identifier])[0]
                if not self.backend.bank.verify(identifier, vector):
                    raise RuntimeError("Witness verification failed")
                self.checkpoint = WitnessedState(identifier, vector)
                committed = True
            self.working = unit(self.checkpoint.vector)
        else:
            if disturbance is not None:
                dx = np.asarray(disturbance, dtype=np.float32)
                if dx.shape != self.working.shape:
                    raise ValueError("Disturbance shape mismatch")
                self.working = unit(self.working + dx)
            drift = 1.0 - float(self.working @ unit(self.checkpoint.vector))
            if drift > self.drift_boundary:
                self.working = unit(self.checkpoint.vector)
                rolled_back = True
        if not self.backend.bank.verify(self.checkpoint.record_id, self.checkpoint.vector):
            raise RuntimeError("Retained checkpoint no longer exists")
        return {"record_id": self.checkpoint.record_id, "committed": committed,
                "rolled_back": rolled_back}


class HingeZeroFullMemory(BaseANN):
    """Complete *VIBE-adapted* memory engine, not a semantic AI or secure vault."""
    def __init__(self, metric, field_limit=256, alpha=4.5, storage_path=None,
                 mode="full"):
        if metric not in ("euclidean", "cosine", "ip", "normalized"):
            raise ValueError("Unsupported metric")
        if not 1 <= int(field_limit) <= 8192:
            raise ValueError("Invalid field limit")
        if mode not in ("full", "packed_control"):
            raise ValueError("Unsupported memory mode")
        self.metric, self.field_limit, self.alpha, self.mode = metric, int(field_limit), float(alpha), mode
        self.storage_path = storage_path
        self._temp = None
        self.bank = None
        self.args = None
        self.stats = {"witness_checks": 0, "witness_failures": 0, "queries": 0,
                      "refinements": 0, "additional_candidates": 0}

    def fit(self, X):
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("Expected 2D memory bank")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("Float memory bank required")
        if self._temp is None and self.storage_path is None:
            self._temp = tempfile.TemporaryDirectory(prefix="hingezero-vibe-")
        root = self.storage_path or self._temp.name
        self.bank = DiskPackedMemory(root, dimension=values.shape[1])
        for start in range(0, len(values), 8192):
            self.bank.append(values[start:start+8192])
        self.args = (512, 512, 8)

    def set_query_arguments(self, candidate_budget, expansion_budget, steps):
        candidate_budget, expansion_budget, steps = map(int,
                                (candidate_budget, expansion_budget, steps))
        if candidate_budget < 1 or expansion_budget < 0 or steps < 0 or steps > 64:
            raise ValueError("Invalid query budget")
        self.args = candidate_budget, expansion_budget, steps

    def append_memories(self, records, sources=None, evidence=None):
        if self.bank is None:
            raise RuntimeError("Call fit or load_memory first")
        return self.bank.append(records, sources=sources, evidence=evidence)

    def load_memory(self, path):
        self.bank = DiskPackedMemory(path, reopen=True)
        self.storage_path = str(path)
        self.args = self.args or (512, 512, 8)
        return self

    def witness(self, record_id, vector):
        self.stats["witness_checks"] += 1
        ok = self.bank.verify(record_id, vector)
        if not ok:
            self.stats["witness_failures"] += 1
        return ok

    def _refine(self, query, candidate_ids, steps):
        records = self.bank.get(candidate_ids)
        normalized = np.stack([unit(v) for v in records])
        seed_scores = self.bank.exact_scores(candidate_ids, query, self.metric)
        order = np.lexsort((candidate_ids, -seed_scores))
        take = order[:min(self.field_limit, len(order))]
        field = normalized[take]
        x = unit(query)
        if np.linalg.norm(x) <= 1e-12:
            weights = seed_scores[take] - np.min(seed_scores[take])
            weights = weights + 1e-9
            x = unit(weights @ field)
        for _ in range(steps):
            h = field @ x
            response = hinge_phi(h, self.alpha)
            feedback = field.T @ response
            feedback /= float(np.sum(np.abs(response))) + 1e-12
            x = unit(0.98 * x + 0.10 * feedback)
        self.stats["refinements"] += steps
        return x

    def _search(self, vector, n):
        if self.bank is None:
            raise RuntimeError("Call fit or load_memory first")
        if self.args is None:
            raise RuntimeError("Set query arguments first")
        original = np.asarray(vector, dtype=np.float32)
        if original.shape != (self.bank.dimension,) or not np.isfinite(original).all():
            raise ValueError("Invalid query")
        n = int(n)
        if n < 1 or n > self.bank.count:
            raise ValueError("Invalid k")
        seeds_count, extra_count, steps = self.args
        seeds = self.bank.packed_candidates(original, max(n, seeds_count), self.metric)
        gathered = seeds
        if extra_count:
            routed = self._refine(original, seeds, steps) if self.mode == "full" else original
            extras = self.bank.packed_candidates(routed, extra_count, self.metric)
            gathered = np.unique(np.concatenate((seeds, extras)))
            self.stats["additional_candidates"] += max(0, len(gathered) - len(seeds))
        elif self.mode == "full" and steps:
            # No external query expansion: still run the original refinement.
            self._refine(original, seeds, steps)
        scores = self.bank.exact_scores(gathered, original, self.metric)
        order = np.lexsort((gathered, -scores))[:n]
        ranked = gathered[order]
        ranked_scores = scores[order]
        # Verify first and last ranked records and optional checkpoint.
        for rid in np.unique(ranked[[0, -1]]):
            if not self.witness(int(rid), self.bank.get([rid])[0]):
                raise RuntimeError("Witness rejected stored record")
        self.stats["queries"] += 1
        return ranked, ranked_scores

    def query(self, vector, n):
        ids, _ = self._search(vector, n)
        return ids.astype(np.int32)

    def memory_recall(self, cue, *, min_margin=0.1, min_similarity=0.35):
        """Single-record recall with explicit abstention; VIBE query must return k."""
        ids, scores = self._search(cue, 2 if self.bank.count > 1 else 1)
        margin = float(scores[0]-scores[1]) if len(scores)>1 else float("inf")
        if float(scores[0]) < min_similarity:
            return {"status": "no_match", "score": float(scores[0]), "margin": margin}
        if margin < min_margin:
            return {"status": "ambiguous", "score": float(scores[0]), "margin": margin}
        rid = int(ids[0])
        record = self.bank.get([rid])[0]
        if not self.witness(rid, record):
            raise RuntimeError("Witness verification failed")
        return {"status": "found", "id": rid, "source": self.bank.source(rid),
                "score": float(scores[0]), "margin": margin,
                "verified": True, "vector": record.copy()}

    def get_additional(self):
        return {"backend": "native_one_bit_full_scan_plus_disk_float32",
                "mode": self.mode, "field_limit": self.field_limit,
                "witness_checks": self.stats["witness_checks"],
                "witness_failures": self.stats["witness_failures"],
                "refinements": self.stats["refinements"],
                "queries": self.stats["queries"]}

    def __str__(self):
        return f"HingeZeroFullMemory({self.metric},{self.mode},{self.field_limit},args={self.args})"


class HingeZeroPackedControl(HingeZeroFullMemory):
    """Matched same-index control with no HingeZero recurrence."""
    def __init__(self, metric, field_limit=256, alpha=4.5):
        super().__init__(metric, field_limit, alpha, mode="packed_control")
