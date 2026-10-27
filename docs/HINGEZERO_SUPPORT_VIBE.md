# Single VIBE method: HingeZero learned-support 32,768

This branch supplies **only** the previously measured HingeZero learned-support VIBE adapter.
It replaces the earlier C++ v3 proposal in upstream PR #16. No two-step alternative or
secondary HingeZero method is registered here.

## Exact evaluated implementation

- Original code and benchmark commit: `60121431afc95979065e1b2ade204a538c3d7be1`
- Exact Python implementation: `vibe/algorithms/hingezero_support/module.py` (source code copied unchanged)
- Dependency: `vibe/algorithms/hingezero_cpp20/module.py` plus `native/HingeZero_VIBE_Cpp20.cpp` (source files copied unchanged)
- Index: single-core FAISS `IndexFlatIP` over the 32,768 highest-frequency corpus vectors
  selected using the **learning split's** provided ground-truth neighbor IDs.
- Query: exact top-100 inner-product search in the selected pool, followed by native
  HingeZero rank/identity verification. HingeZero refinement steps = **0**.
- Dataset: VIBE `yi-128-ip`; evaluation uses **1000 test queries** and recall@100.
- Parameters: `field_size=256`, `pool_size=32768`.
- Registered VIBE algorithm: `hingezero-support-lab`, OOD enabled.
- VIBE evaluation: `python run.py --local --force --runs 2 --parallelism 1 --dataset yi-128-ip --algorithm hingezero-support-lab --count 100`.
- Export: `python export_results.py --parallelism 1 --skip-pca-mahalanobis`.

## Recorded benchmark results and variability

The previous published VIBE harness run on a GitHub Actions CPU runner returned:

- **5,010.826592 QPS**, **98.617% recall@100**, `pool_size=32768`:
  https://github.com/HingeZero/vibe/actions/runs/40445051981
- A later same-configuration runner reported **2,326.520725 QPS** at the
  **same 98.617% recall**, indicating machine/run throughput variability:
  https://github.com/HingeZero/vibe/actions/runs/40445259052

These are VIBE-generated summary scores, not independently certified results
on the VIBE maintainers' reference CPU. They do not guarantee 5,011 QPS on a rerun.
Accuracy may change on other datasets; no cross-dataset claim is made.

The independent OOD learning split's ground-truth neighbor IDs were used in
training the search-support pool, per VIBE's documented `fit_ood` interface.
**Test query labels are never used to build the pool.**

## Reproduce

Build the HingeZero support image using VIBE's container method:
```bash
./install.sh --algorithm hingezero_support
python3 run.py --dataset yi-128-ip --algorithm hingezero-support-lab --count 100
./export_results.sh --parallelism 1
```

To reproduce the exact pinned GitHub Actions harness and inspect full logs,
run `.github/workflows/hz-support-vibe.yml`.

This is the **single** HingeZero method proposed in PR #16 and it does not
assert independent acceptance, a leaderboard ranking, or a speed advantage
over methods that use the full corpus without a learned restricted pool.
