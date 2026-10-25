# HingeZero 5,000 QPS lab — exact 8-bit certified bounds

Target: reach 5,000 QPS on VIBE yi-128-ip at **100% exact
recall@100** without altering the single zero-step C++20 HingeZero
architecture. Target is **not a claimed achieved result**.

Development branch only: original HingeZero memory, append-only
float32 source bank, same HingeZero native `refine(q,0)`, same record
witness, same exact ranking. Quantized approximate score uses 8-bit
signed signatures (128 bytes per vector) instead of 16-bit signatures
(256 bytes per vector), retaining float32 records losslessly on disk.

Cauchy–Schwarz conservative bound and exact reranking remain:
`x·q ≤ x'·q' + ||x−x'|| ||q|| + ||x'|| ||q−q'||`.
AVX2 widens signed int8 to int16 before vector integer dot.
The reduced signature bandwidth may improve QPS at the cost of
looser score bounds and more exact float32 scoring. All candidate
pruning must remain correctness-preserving.

Do not update upstream PR until GitHub VIBE and exact-ID parity
pass, and measured throughput exceeds the current ~288–292 QPS.

## AVX2 signed-int8 optimised candidate kernel

GitHub initial five-run VIBE result: **332.20 QPS, recall@100 1.00000**.
[Original 8-bit benchmark](https://github.com/HingeZero/vibe/actions/runs/40170356490).
The next experiment changes only integer arithmetic: the signed int8
record is offset by +128 inside AVX2, then `_mm256_maddubs_epi16`
and `_mm256_madd_epi16` accumulate the integer inner product.
Query quantization is restricted to [-63,63] to prevent intermediate
int16 saturation; subtracting 128*sum(query_codes) exactly undoes
the unsigned record offset. The query's larger quantization scale
is propagated into the Cauchy bound. 100% top-100 recall remains
a mandatory acceptance test, not a presupposition.


## Results — 25 October 2026

**Target of 5,000 QPS at exact top-100 retrieval has NOT been met.**
All runs are author-operated single-core GitHub Actions on VIBE `yi-128-ip`,
not organizer-certified.

| Implementation | QPS | VIBE recall@100 | Configuration | Evidence |
|---|---:|---:|---|---|
| Existing submitted C++20 signed-16 certified index | 287.76 | 1.00000 | 0 refinement steps, five timed runs | [run](https://github.com/HingeZero/vibe/actions/runs/40169154193) |
| Experimental signed-int8 bounded scoring | 332.20 | 1.00000 | 0 steps, five timed runs | [run](https://github.com/HingeZero/vibe/actions/runs/40170356490) |
| AVX2 `vpmaddubsw` signed-int8 optimisation | 253.00 | 1.00000 | 0 steps, five timed runs | [run](https://github.com/HingeZero/vibe/actions/runs/40170542615) |
| PCA cascade, staged certified quantized bounds | 47.29 | 1.00000 | 0 steps, three timed runs | [run](https://github.com/HingeZero/vibe/actions/runs/40171043399) |
| Approximate native IVF, 4/128 clusters | 7,656.03 | 0.56634 | 0 steps, five timed runs | [run](https://github.com/HingeZero/vibe/actions/runs/40168064952) |
| Approximate native IVF, 64/128 clusters | 578.06 | 0.96435 | 0 steps, five timed runs | [run](https://github.com/HingeZero/vibe/actions/runs/40168064952) |

**Important:** the 8-bit version reached higher QPS in a separate GitHub
run, but this is not a matched-host performance proof of a 15% gain.
A controlled repeat is needed before claiming it is better. Both new
quantization variants require a numerical upper-bound audit before a
mathematical guarantee can be promoted from a successful finite test.

The PCA cascade result highlights a cost trap: per-query processing of
many progressive bounds and scattered survivors outweighed skipping
most high-dimensional exact scores. The basis was trained only on
training-corpus samples and the stored float32 memory/witness pathway
was not bypassed. VIBE recall was 100% in these tests, but that does not
validate mathematical exactness over all possible queries.

**PR #16 remains unchanged:** exactly one registered HingeZero C++20
algorithm, zero refinements, with the previously confirmed full-results
index. The next useful research should profile the branch-level bottleneck
and test alternative memory layouts, especially one-pass cache-resident
coarse bounds and selective compact-signature blocks. Do not claim
5,000 QPS until reproducibly measured at the agreed recall threshold.
