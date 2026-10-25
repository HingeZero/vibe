# HingeZero C++20 — VIBE single-core CPU submission

Author: David Duffy · HingeZero.

The one registered algorithm is `hingezero`, tested on `yi-128-ip`
at `k=100`, with `steps=0` and `budget=208`.

## Implementation

- The native HingeZero `refine(query,0)` entry point is called;
  **dual-tanh iterations are not executed** in this benchmark configuration.
- Signed 8-bit candidate signatures are compared with signed 16-bit query
  signatures using SIMD when available. All bank signatures are scanned.
- A **208-record approximate shortlist** is selected and rescored against the
  original float32 stored vectors using float64 inner products. The returned
  top-100 IDs are verified against their original byte-identical stored
  records before returning results.
- The original float32 store remains intact; the additional 8-bit
  representations are only for candidate selection.
- The shortlist is **not a certified exact-nearest-neighbour index**.
  A 100% measured recall@100 on VIBE does not guarantee the same for
  other datasets, queries or candidate budgets.

The original HingeZero C++20 bipolar memory source remains included
unchanged. The VIBE float32 adapter is necessary because the benchmark
uses continuous-valued embeddings, not bipolar records.

## Measured performance

On 25 October 2026, the same implementation and a twelve-budget sweep
recorded **866.66 QPS and 1.00000 recall@100** with budget 208, build time
approximately 0.14 seconds:

[Author-run VIBE Actions benchmark](https://github.com/HingeZero/vibe/actions/runs/40218473778).

Those numbers are **hardware-specific author measurements, not independently
certified VIBE leaderboard results**. An exact-ID parity test is also run
against the earlier exhaustive/certified implementation on a held-out
subset. The production submission pins just budget 208 and reports its
own separate five-run result.

## Reproduce

```bash
g++ -O3 -std=c++20 -fPIC -shared -DNDEBUG -ffp-contract=off vibe/algorithms/hingezero_cpp20/native/HingeZero_VIBE_Cpp20.cpp -o /tmp/libhz20.so
g++ -O3 -std=c++20 -ffp-contract=off vibe/algorithms/hingezero_cpp20/native/HingeZero_Native_1B.cpp -o /tmp/hz20_original
export HINGEZERO_CPP20_LIB=/tmp/libhz20.so
export HZ20_ORIGINAL_EXE=/tmp/hz20_original
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python verification/smoke_hingezero_cpp20.py
python run.py --local --force --runs 5 --parallelism 1 --dataset yi-128-ip --algorithm hingezero --count 100
python export_results.py --parallelism 1 --skip-pca-mahalanobis
```

For historical exact-index benchmarks see Git history (before this
shortlist-based implementation).
