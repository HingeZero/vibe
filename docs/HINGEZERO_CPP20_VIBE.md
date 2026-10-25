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
exhaustively visits every record without cue modification, 100% recall@100
is expected, and *does not demonstrate an additional accuracy improvement
from HingeZero*. Time is O(Pd), not approximate-index O(sublinear) search.
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
