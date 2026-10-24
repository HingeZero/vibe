# HingeZero C++ v3 for VIBE

**Author:** David Duffy / HingeZero. **Status:** proposed; organizer benchmarks not yet run or accepted.

This submission integrates the published native C++ v3 algorithm from
[HingeZero/big-ann-benchmarks at 1fc8c554](https://github.com/HingeZero/big-ann-benchmarks/commit/1fc8c5540f689421d8577ac7915c8b0ebbd81491)
with the VIBE BaseANN API. The C++ source is checked out at that fixed commit
and compiled without modifying the kernel.

## Scope and limitations

The index backbone is FAISS IVF/PQ 1.14.3, not a novel HingeZero index.
The additional component is iterative dual-tanh associative refinement,
routed retrieval of extra candidates, witness checks and exact original-metric
candidate reranking. It is **not** the separate HingeZero packed 1-bit memory
architecture and **not** a demonstration of billion-vector capacity.
The native core uses alpha=4.5, eps=0.10, lam=0.02, steps=8, and
phi(h)=tanh(h)+alpha*tanh(2h).

Only static float32 k-NN is supported: Euclidean, cosine, inner product
and normalized inner product. Original vectors are retained in RAM for
reranking in addition to the IVF/PQ index. This is not disk-only storage.
No query samples or ground-truth neighbors are used for fitting.

VIBE CPU configuration is single-core for both indexing and queries.
The six query configurations vary nprobe (16/32/64) and seed candidates
(256/512); expansion=256, refinement steps=8, fixed training seed 20261008.

## Reproduction

From the VIBE repository root with Apptainer/Singularity installed:

```sh
./install.sh --algorithm hingezero
apptainer exec images/hingezero.sif python3 verification/smoke_hingezero.py
python3 run.py --dataset yi-128-ip --algorithm hingezero --count 100
python3 run.py --dataset llama-128-ip --algorithm hingezero --count 100
```

The dataset files must be present according to VIBE setup. No performance
values from these commands are claimed until runs actually complete.

## Controlled comparison needed

Before attributing benefit to HingeZero, compare against identical FAISS
IVF/PQ, query expansion without recurrence, and the built-in native
zero-step pathway, all on the same dataset, hardware, evaluation metric and
candidate-search budget. Record recall@100, QPS, build time and peak RAM
together with the complete raw results. Witness checks validate retained
records, not ground-truth answer correctness.

Earlier author-reported BigANN OpenAI-100K recall@10 and QPS measurements
are not comparable with VIBE recall@100 or its single-core CPU setup.
