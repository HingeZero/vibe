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

## 3. Original C++20 source and VIBE-real-vector adaptation (25 October 2026)

**This is now the canonical original-source adaptation, not the earlier Python
one-bit-sign approximation.** The original
[\`HingeZero_Native_1B.cpp\`](https://github.com/HingeZero/HingeZero-1B-Memory/blob/main/HingeZero_Native_1B.cpp)
is included unchanged in
\`vibe/algorithms/hingezero_cpp20/native/HingeZero_Native_1B.cpp\`.
The paired \`HingeZero_VIBE_Cpp20.cpp\` is compiled with \`-std=c++20\` and
adapts the same 256-memory uniform-evidence field, dual-tanh, eight-step
refinement, tagged witness, append-only disk bank and exhaustive scan to
original float32 vectors and VIBE's distance definitions.

**[Successful GitHub Actions parity/benchmark run](https://github.com/HingeZero/vibe/actions/runs/40165136818)**
and [raw artifact](https://github.com/HingeZero/vibe/actions/runs/40165136818/artifacts/12587822333).

- Original bipolar C++20 source-vs-adapted-kernel output parity: **maximum
  absolute refined-vector error 0.0** on the exact reference cue.
- Native source append-prefix immutability, VIBE float append/reopen, altered
  record witness rejection, top-100 raw exact scoring on IP/L2/cosine/normalized,
  and explicit anchored-query scoring checks: **passed**.
- \`yi-128-ip\`: 187,843 real-valued 128-dimensional reference vectors,
  recall@100, single benchmark CPU core, one timed run per method.
- **No FAISS** used for these three scans.

| C++20 variant | Recall@100 | QPS | Meaning |
|---|---:|---:|---|
| Raw exhaustive original-query control | 1.00000 | 11.15 | Exact baseline, not an HZ performance improvement |
| Real-vector anchored (95% original / 5% HZ-refined) | 0.87436 | 11.08 | Explicit test-dataset adaptation |
| Original HZ eight-step refined query | 0.11574 | 11.05 | Faithful source-style refinement without anchoring |

**Interpretation:** VIBE measures neighbours of the *original* cue. Original
HingeZero reconstructs/changes the cue; selecting neighbours of the altered
cue changes the retrieval objective and can lower recall@100 dramatically.
Anchoring improves agreement with VIBE's original cue metric, but it remains
slower and less accurate than exact exhaustive search and much slower than
the FAISS-assisted approximation.

The **bipolar 128-bit** storage from the original billion-record experiment
cannot losslessly represent arbitrary real-valued VIBE embeddings. This
C++20 adaptation deliberately stores every float32 coordinate. It is not a
billion-record run, does not establish lossless 1-bit VIBE storage, and does
not independently prove HingeZero superiority over established ANN methods.
