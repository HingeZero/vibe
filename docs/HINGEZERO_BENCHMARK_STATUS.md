# HingeZero VIBE benchmark status — 25 October 2026

**Scope:** two independently testable HingeZero implementations in the same pull
request. All figures are *author-operated GitHub Actions* measurements, not
organizer-certified leaderboard scores. Each configuration uses one timed run,
the `yi-128-ip` dataset (187,843 vectors; recall@100) and a single benchmark
CPU core. GitHub-hosted CPU type and measured memory may differ from official
leaderboard hardware.

## 1. C++ v3 / FAISS-assisted

Code: `vibe/algorithms/hingezero/`. FAISS IVF/PQ generates candidate IDs.
Pinned HingeZero C++ v3 iteratively refines/reroutes part of the search,
retains a per-query witnessed checkpoint, and exact-reranks original metric.

[Successful runner and artifacts](https://github.com/HingeZero/vibe/actions/runs/40131936103)

| nprobe | initial candidates | Recall@100 | QPS |
|---:|---:|---:|---:|
| 16 | 256 | 0.73211 | 673.93 |
| 16 | 512 | 0.74046 | 546.62 |
| 32 | 256 | 0.81893 | 457.32 |
| 32 | 512 | 0.82956 | 394.05 |
| 64 | 256 | 0.89058 | 274.31 |
| 64 | 512 | 0.90457 | 245.96 |

No same-index FAISS-only evaluation was included in this run, so improvement
attributable to HingeZero cannot be quantified from these six rows.

## 2. Standalone packed memory with HingeZero

Code: `vibe/algorithms/hingezero_full/`. No FAISS. Disk-backed append-only
float32 original records coexist with packed 1-bit sign signatures. Candidate
search uses one-bit Hamming scanning (not accurate for all floating metrics),
HingeZero eight-step associative routing selects expansion candidates, and
original float32 distances determine the final top-k.

[Successful smoke tests, benchmark, controls and raw artifacts](https://github.com/HingeZero/vibe/actions/runs/40163249408)

| initial candidate budget | HZ recall@100 | HZ QPS | Same-index no-HZ recall@100 | No-HZ QPS |
|---:|---:|---:|---:|---:|
| 256 | 0.10126 | 61.60 | 0.05427 | 67.70 |
| 512 | 0.10726 | 57.02 | 0.05427 | 67.23 |
| 1024 | 0.12631 | 50.98 | 0.08385 | 65.10 |

Expansion candidates = 512, alpha = 4.5, epsilon = 0.10,
lambda = 0.02, steps = 8; fixed field width 256.

Interpretation: HingeZero significantly increases absolute retrieved-neighbour
fraction *relative to this weak packed control* but the one-bit candidate index
is far less effective on arbitrary inner-product embeddings than the
FAISS-assisted adapter. The new variant is not an improvement over the
existing FAISS-based score. No statistical significance claim is made from
the single run.

## Memory-specific checks

The standalone backend's smoke suite covered append-prefix byte immutability,
persistent reopening, metadata source retention, rejecting modified stored
states, exact all-candidate VIBE scoring on four metrics, no-match abstention
and retaining-session rollback. These checks passed; they do not substitute
for adversarial security tests or factual provenance validation.

The benchmark adapter is **not** an encrypted secure vault, full semantic AI,
or an actual billion-vector VIBE run. It does not demonstrate that the one-bit
signature contains all information in a float32 memory record.

## What remains

Repeat the official benchmark across representative datasets and machines;
separately ablate HingeZero from FAISS-assisted query expansion under matched
candidate, reranking and compute budgets; study native packed binary datasets
where 1-bit patterns preserve the original memory representation rather than
converting unstructured floating-point embeddings into signs.
