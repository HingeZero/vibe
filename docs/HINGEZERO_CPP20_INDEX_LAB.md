# HingeZero C++20 candidate-index investigation

**Development only — the upstream VIBE pull request retains the one
zero-refinement algorithm and the verified exhaustive SIMD scan.**

The index is deterministic spherical coarse clustering trained only on
stored corpus vectors (no labels or VIBE query ground truth). It creates
128 inverted lists, stores each original float32 record unchanged, and
searches the top `nprobe` lists according to centroid similarity.

The same HingeZero C++20 `refine(original,0)` pipeline, exact inner
products on candidate records, and original-byte witness verification
remain in place. No FAISS dependency is used. The native IVF index copies
the float32 records into contiguous posting lists for faster CPU scanning,
which increases memory usage. Newly appended records cause the index to
rebuild when the bank is remounted; the original append-only bank stays
unchanged.

This is **approximate** inner-product retrieval. Searching fewer lists
cannot mathematically guarantee 100% recall. A full `nprobe=128` search
provides an exact baseline and all returned IDs are witness-checked. Only
one final configuration should be submitted if the recall/QPS tradeoff is
acceptable, keeping `steps=0`.

Six diagnostic probe budgets: 4, 8, 16, 32, 64, 128, with five timed runs
and official VIBE recall@100. Do not claim an improvement until the
GitHub-hosted runner completes and validates the dataset.

## Verified GitHub result — 25 October 2026

[Official VIBE runner / five timed runs](https://github.com/HingeZero/vibe/actions/runs/40168064952) · [Raw artifact](https://github.com/HingeZero/vibe/actions/runs/40168064952/artifacts/12589905127)

| nprobe of 128 | Recall@100 | VIBE queries/sec |
|---:|---:|---:|
| 4 | 0.56634 | 7656.03 |
| 8 | 0.68914 | 4464.27 |
| 16 | 0.79880 | 2384.59 |
| 32 | 0.89580 | 1180.01 |
| 64 | 0.96435 | 578.06 |
| 128 | 1.00000 | 303.16 |

Dataset: yi-128-ip, 187,843 records, 128-dimensional float32,
k=100, one core, five timed runs, training corpus only for clustering.
Index construction in this run ~3.85 seconds (rebuilds when reopening);
posting storage duplicates raw float32 records in RAM. For nprobe<128,
nearest neighbours from unsearched clusters **can be missed**.

### Crucial same-worker comparison

[Same-machine exhaustive-vs-indexed diagnostic](https://github.com/HingeZero/vibe/actions/runs/40168064952)
using *identical* random training vectors, query vectors, CPU and k:
65,536 records; dimension 128; 256 queries; k=100.

| Kernel | QPS | Recall vs exact |
|---|---:|---:|
| Pre-index exhaustive AVX2 | 973.37 | 100% |
| IVF probing all 128 clusters | 880.08 | 100% |
| IVF probing 64 of 128 | 1816.18 | 78.254% |
| IVF probing 32 of 128 | 3224.99 | 52.824% |

Exact native top-100 parity passed for all-cluster search.

**Conclusion:** IVF at full recall was *slower* than the prior direct
AVX2 scanner on matched hardware in this controlled diagnostic.
Subset search did improve QPS but traded away accuracy; absolute
recall depends materially on the dataset. GitHub runner variability
makes the different-run 303.16 vs prior 151.27 QPS comparison
insufficient to establish a full-recall index speedup.

**Submission decision:** keep the existing single, C++20 zero-step,
100%-recall exhaustive implementation in PR #16 unchanged.
The indexed prototype remains available on this separate lab
branch for future accuracy/speed work. A 96.435%-recall,
578-QPS configuration should **not** silently replace a 100%-recall
submission.

## High-recall ANN experiment — 25 October 2026

The first native C++20 HingeZero navigable graph produced weak
recall on VIBE `yi-128-ip`. The graph's best result was 11.01%
recall@100, and a norm-corrected MIPS graph reached 67.64%
at 185.51 QPS; the graph route is not ready for submission.

This separate IVF branch uses the **same one native C++20 HingeZero
ANN candidate engine**, with zero refinement iterations and
the original float32 memory/witness pipeline. It sweeps
`nprobe=[64,80,96,104,112,120,128]` of 128 coarse clusters to
measure whether >=99% recall can be obtained with meaningfully
better QPS than the current exact submission.

This is **not** a claim of reaching 5,000 QPS. The experiment is
valuable only if the measured QPS at a fixed recall threshold
is competitive. Do not choose parameters using VIBE test-query
ground-truth IDs; inspect reported metrics only for scientific
evaluation, then seek independent measurements.

## High-recall follow-up (completed 25 October 2026)

[Full successful GitHub Actions VIBE run](https://github.com/HingeZero/vibe/actions/runs/40174681600)
([result artifact](https://github.com/HingeZero/vibe/actions/runs/40174681600/artifacts/12591348580)).
Exactly one C++20 HingeZero ANN algorithm swept seven `nprobe` budgets.
Dataset `yi-128-ip`, 187,843 float32 vectors, d=128, k=100,
one CPU benchmark core, five runs per setting, zero HingeZero refinements.

| Probed groups of 128 | Recall@100 | QPS |
|---:|---:|---:|
| 64 | 0.96435 | 199.09 |
| 80 | 0.98191 | 159.14 |
| 96 | 0.99254 | 134.35 |
| 104 | 0.99625 | 125.73 |
| 112 | 0.99827 | 117.61 |
| 120 | 0.99919 | 109.40 |
| 128 | 1.00000 | 106.38 |

Build time: **4.79 seconds**, plus separate bank storage;
the VIBE worker reported `index_size` **189,064 KiB** process
USS increase. The original records remain lossless and witnessed.

**Evaluation:** 5,000 QPS at 99% recall is NOT achieved.
The fastest 99%+ result here is **134.35 QPS at 99.254% recall**.
The pre-existing exact native submission ran at **287.76 QPS and
100% recall** in a separate worker, but absolute QPS across different
GitHub workers is not a causal comparison.
Earlier on another worker the same IVF 64-probe configuration measured
578.06 QPS at the *same 0.96435 recall*, demonstrating timing variance
of nearly 3x. More controlled matched-run evidence would be required
before any performance superiority claim.

This iteration confirms the bottleneck: probing most coarse lists
returns high recall at the expense of visiting/scoring too many
original 128-dimensional vectors. The ANN engine is **not promoted**.
Upstream [VIBE PR #16](https://github.com/vector-index-bench/vibe/pull/16)
is intentionally kept unchanged.

The separate navigable graph experiments were also not promoted:
- [Plain inner-product graph](https://github.com/HingeZero/vibe/actions/runs/40173668235):
  7.11% recall at 3,765.80 QPS; 11.01% at 502.64 QPS.
- [Norm-lift MIPS graph](https://github.com/HingeZero/vibe/actions/runs/40174074835):
  28.89% recall at 1,706.95 QPS; 67.64% at 185.51 QPS,
  build 286.20 s.
- [AVX2 MIPS graph](https://github.com/HingeZero/vibe/actions/runs/40174549363):
  ~28.89% recall at 3,235.42 QPS; 67.58% at 316.46 QPS,
  build **65.92 s**. Fast dot products accelerate graph construction
  but do not solve candidate graph recall.

Research next: combine improved MIPS-sensitive clustering with a
**secondary compressed residual score filter** inside each cell,
and measure actual score computations, total RAM, build time and QPS
on the same worker. Target a high-recall candidate budget of hundreds
to thousands rather than tens of thousands; exact float32 reranking
and HingeZero witnesses must remain. Do not label exact reranking as
a guarantee of exact recall after approximate candidate omission.
