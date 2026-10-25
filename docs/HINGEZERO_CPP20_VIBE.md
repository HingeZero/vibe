# HingeZero C++20 — single VIBE submission (zero refinement iterations)

David Duffy · HingeZero · dataset: `yi-128-ip` (float32, 128 dimensions, inner product).

This PR offers **exactly one registered algorithm**, `hingezero`. It uses the
HingeZero C++20 memory architecture and a fixed **0 refinement steps**.
No unrefined-bypass control, no second algorithm, no FAISS candidate shortlist,
and no experimental 95/5 query anchoring are submitted.

The included `native/HingeZero_Native_1B.cpp` is a byte-identical copy of
[the published original bipolar memory engine](https://github.com/HingeZero/HingeZero-1B-Memory/blob/main/HingeZero_Native_1B.cpp).
The paired `native/HingeZero_VIBE_Cpp20.cpp` is the adapted backend needed
because VIBE's vectors are arbitrary real numbers, not exact bipolar patterns.
Using one bit per coordinate would destroy magnitude and precision. The
adapted bank therefore stores float32 vectors exactly on disk (4*d bytes/record),
supports append without rewriting prior records, and uses a mapped whole-bank
C++20 search with original VIBE distance metrics and record-identity witnesses.

For each query, **the same HingeZero pipeline always runs**:
1. Validate input and invoke the native `refine(query, 0)` function; it
   returns `x_0` unchanged and executes **zero** dual-tanh iterations.
2. Exhaustively scan the stored memory bank with the original cue using
   the dataset's exact metric.
3. Return the top 100 ranked stored IDs after stored-state witness checks.

The iterative dual-tanh mechanism is retained in the source engine, but is
**not executed** in this submitted configuration. Because the scanner
evaluates a conservative upper bound for every stored record without cue modification, 100% recall@100
is expected, and *does not demonstrate an additional accuracy improvement
from HingeZero*. Time is O(Pd) for compact-signature scanning, with far fewer full-precision comparisons in typical data; no sublinear worst-case complexity is claimed.
The tagged-state witness proves stored-ID/byte correspondence, not the truth
or factual provenance of a memory.

Run:
```bash
g++ -O3 -std=c++20 -fPIC -shared -DNDEBUG -ffp-contract=off vibe/algorithms/hingezero_cpp20/native/HingeZero_VIBE_Cpp20.cpp -o /tmp/libhz20.so
export HINGEZERO_CPP20_LIB=/tmp/libhz20.so
# Build the bundled original C++20 smoke binary as well
g++ -O3 -std=c++20 vibe/algorithms/hingezero_cpp20/native/HingeZero_Native_1B.cpp -o /tmp/hz20_original
export HZ20_ORIGINAL_EXE=/tmp/hz20_original
python verification/smoke_hingezero_cpp20.py
python run.py --local --force --dataset yi-128-ip --algorithm hingezero --count 100 --runs 1 --parallelism 1
python export_results.py --parallelism 1 --skip-pca-mahalanobis
```

CI re-runs the one HingeZero algorithm only and stores VIBE's raw scores.
Measurements are author-operated, not independently certified by VIBE.

## GitHub-hosted VIBE result (25 October 2026)

[Successful single-algorithm GitHub Actions run](https://github.com/HingeZero/vibe/actions/runs/40166191634) · [raw score artifact](https://github.com/HingeZero/vibe/actions/runs/40166191634/artifacts/12588457303)

| Dataset | Algorithm | Refinement iterations | Recall@100 | QPS |
|---|---|---:|---:|---:|
| yi-128-ip | hingezero | 0 | 1.00000 | 19.36 |

One run, one benchmark CPU core, author-operated GitHub Actions. The source
build, persistent append-prefix preservation, exact metric ranking on all four
supported distances, stored-state witness and the zero-iteration no-bypass
correctness checks passed. The zero-step mode explicitly calls the native
HingeZero `refine(original,0)` function; it does **not** take a separate
raw-search pathway.

Because this is an **exact exhaustive whole-bank scan**, 100% recall@100
should not be interpreted as a speed, capacity or accuracy breakthrough.
The independent VIBE organizer has not accepted or certified these results.

## Zero-step SIMD C++20 optimisation (25 October 2026)

**The submitted HingeZero algorithm remains one configuration, zero refinement
iterations, with the identical original record store, all-record search,
and witness validation.** The only change is how the C++20 inner-product
accumulation is performed: runtime AVX2 SIMD computes exact floating-point
inner products by promoting original float32 coordinates to double precision.
CPUs without AVX2 use a scalar fallback. Every vector is still scored; there
is no shortlist, external FAISS or alternative HingeZero mode.

GitHub-hosted, author-run verification:

- [Single-run VIBE performance](https://github.com/HingeZero/vibe/actions/runs/40166928416):
  recall@100 = **1.00000**, **311.85 QPS**.
- [Five-run VIBE evaluation](https://github.com/HingeZero/vibe/actions/runs/40167003998):
  recall@100 = **1.00000**, **204.60 QPS**.
- [Five-run VIBE evaluation with same-machine scalar/AVX2 diagnostic](https://github.com/HingeZero/vibe/actions/runs/40167139935):
  recall@100 = **1.00000**, **152.96 QPS**.
- The same-machine diagnostic (65,536 vectors; d=128; 192 queries;
  k=100) measured **31.67 QPS** for the original scalar C++20 engine
  and **443.85 QPS** for the AVX2-optimised engine: **14.01×** measured
  speedup, **identical top-100 IDs** across all queries.

QPS varies between GitHub-hosted runner samples. This does not establish
performance on VIBE's official leaderboard CPU. The difference arises
from CPU arithmetic optimisation, not from HingeZero nonlinear iterations
(which remain at zero) or a new approximate index.

## Certified C++20 exact-pruning acceleration — 25 October 2026

The submitted HingeZero configuration remains **steps=0**, and no FAISS or
second algorithm is registered. The search engine preserves the original
float32 memory bank and computes exact inner-product scores for potentially
top-100 records. A compact per-record int16 signature and stored error norms
supply an upper bound on the true inner product:

`x·q ≤ x'·q' + ||x−x'||·||q|| + ||x'||·||q−q'||`

Here `x'` and `q'` are the reconstructed 16-bit quantized representations.
The algorithm first evaluates exact scores for its approximate top 100 to
establish a real threshold. It then checks **all** upper bounds and skips
any exact score calculation only if the upper bound lies strictly below
the current hundredth-best *exact* score. Stored candidates are witness
checked after reranking. Conservative numerical rounding margins are added.

This does **not** store arbitrary float32 vectors solely in 16 bits: the
original float32 records remain intact, and the extra signatures require
approximately 256 bytes/record plus small index metadata. Append/persistence
checks still validate unchanged historical memory bytes, and the index is
rebuilt at mount. This is a deterministic **exact-retrieval index** based on
certified pruning, not a claim of a new mathematical HingeZero activation.

**[Successful five-run VIBE test](https://github.com/HingeZero/vibe/actions/runs/40169016186)**
([raw HDF5 artifact](https://github.com/HingeZero/vibe/actions/runs/40169016186/artifacts/12589451492)):
`yi-128-ip`, recall@100=**1.00000**, QPS=**292.33**, index-build=**0.19 s**
(single benchmark core, author-run GitHub worker).

On the same worker, against the previous exact SIMD implementation, with
the *same* synthetic 128-dimensional records and queries:

| Memory size | SIMD exhaustive QPS | Certified-pruning QPS | Speed-up | Exact results |
|---:|---:|---:|---:|---|
| 65,536 | 436.18 | 862.94 | 1.98× | Top-100 ID parity |
| 187,843 | 156.17 | 296.24 | 1.90× | Top-100 ID parity |

The larger controlled workload evaluated an average of **100.3 original
float32 records per query**, compared with 187,843 in exhaustive scoring,
because 16-bit score bounds safely rejected the rest. Every stored
quantized signature was still scanned, so these numbers do not demonstrate
constant-time query cost. No external VIBE maintainer verification is claimed.
