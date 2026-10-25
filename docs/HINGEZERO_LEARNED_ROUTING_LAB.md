# Experimental supervised query routing for HingeZero

A **research-only** VIBE CPU OOD implementation inspired by MLANN's
supervised learning-query mechanism. Uses an existing `hnswlib` graph to
find similar training queries. Candidate IDs are the stored original-memory
neighbours of those training queries. HingeZero's native C++ query ranks
only those original float32 records using float64 accumulation and checks
the stored witness.

- This is an **approximate** candidate router, not certified exact top-100.
- 100% recall is NOT guaranteed. Measure VIBE recall@100 for each query budget.
- Training uses **only** VIBE's authorized `learn` and `learn_neighbors`
  (OOD training data). Test queries and ground truth are never used to train.
- Single-core construction/search: `num_threads=1`, OpenBLAS 1.
- HNSW query routing is **existing technology**, not a novel HingeZero
  algorithm itself; speed gains must be attributed correctly.
- Zero iterative HingeZero refinement, original memory/witness preserved.
- This is a separate branch and algorithm, not the existing PR.
