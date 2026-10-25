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
