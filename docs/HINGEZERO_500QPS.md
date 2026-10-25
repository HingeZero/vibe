# HingeZero VIBE 500+ QPS experiment

No changes to the submitted branch or to ANN ranking semantics.
This lab uses 8-bit quantised bound signatures (rather than 16-bit),
an AVX2/AVX-512BW/scalar runtime-dispatched inner product, and the
unchanged lossless original-memory scoring and witness verification.

The quantization error terms remain in the score bound. Every 100-ID
answer must equal the preceding exact C++20 implementation in a
same-worker matched comparison on VIBE test queries.

The author-operated result is **not** an official VIBE leaderboard
entry, and cross-worker speed comparisons are not treated as wins.

Success: single-core VIBE >=500 QPS at measured recall@100=1.00000,
and zero-step HingeZero query/witness tests pass. Otherwise record the
actual performance and do not promote the branch.

## Experimental readout — 25 October 2026

Single-core VIBE `yi-128-ip`: 187,843 × 128 float32 vectors,
k=100, five VIBE timed runs, zero HingeZero refinement steps.
Native C++20 original-record scoring and witness verification
retained unchanged. Nothing was committed to the open official PR.

**Two-stage int8 + int16 bound index:** VIBE recall@100 **1.00000**,
**348.83 QPS**, build **0.33 seconds**, measured process `index_size`
**173,376 KiB**. Experimental branch [`hingezero-cpp20-500qps-two-tier`](https://github.com/HingeZero/vibe/tree/hingezero-cpp20-500qps-two-tier).

In the **matched-host native 300-query test**, the submitted 16-bit
baseline reached **286.67 QPS** versus **354.29 QPS** for the two-tier
engine, a **1.236× throughput ratio**. All ranked top-100 IDs were
identical, and exact float32 scores were computed for only **113.6**
records per query on average (vs. baseline **102.9**).
[Successful 5-run VIBE and matched-host test](https://github.com/HingeZero/vibe/actions/runs/40200726345).

**Single-stage int8:** [successful test](https://github.com/HingeZero/vibe/actions/runs/40199951192):
329.74 VIBE QPS, perfect recall; matched-host 333.81 vs 274.31
QPS, 1.217×. It scored about 3,439 float32 records per query.

**Single-pass int8:** [successful test](https://github.com/HingeZero/vibe/actions/runs/40200158796):
379.73 VIBE QPS, perfect recall, but matched-host experimental 377.05
versus baseline 378.43 QPS (0.996×) and 7,197 original float32
record scores per query. The apparently higher standalone QPS was
runner variability, not an improvement.

**AVX-512 VNNI:** [successful compilation and parity](https://github.com/HingeZero/vibe/actions/runs/40200483251),
but runner was AMD EPYC 7763 and exposed only AVX2, so native
VNNI-specific execution has **not** been tested.

**Fused single-pass two-tier:** [successful test](https://github.com/HingeZero/vibe/actions/runs/40200890046):
318.18 VIBE QPS; matched-host 321.98 vs baseline 299.05 QPS
(1.077×), slower than the two-pass two-tier result.
Do not promote.

The best validated improvement from these experiments is the
**two-pass two-tier 8-bit + 16-bit index**. It improves same-worker
query throughput, but adds substantial index RAM (~173 MiB
process USS increase in the VIBE worker) and longer build time.

**500 QPS at 100% measured recall has not been achieved.**
At current best matched-host throughput (~354 QPS), 500 requires
reducing native per-query time from ~2.82 ms to ≤2.00 ms,
a further ~29% latency reduction (~41% higher throughput).
Prioritise reducing the full 187,843-signature scan with
safe block-level bounds and better cache locality, not
approximate neighbour omission. Measure on the **same worker**;
benchmark all four metrics and keep zero refinement.

Note: 100% measured VIBE recall and a Cauchy-style score-bound
calculation are not a floating-point proof for every possible query.
Independent numerical validation and official benchmark verification
remain outstanding.
