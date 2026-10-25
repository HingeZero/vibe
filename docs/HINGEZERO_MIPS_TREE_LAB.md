# HingeZero VIBE norm-aware hierarchical index experiment

Native C++20 indexed original float32 bank, tested in isolation from the
submitted pull request.

The deterministic balanced tree uses norm-aware splits plus variance-based
coordinate splits. Each node stores its centroid, maximum original vector
norm and per-block residual radii; exact per-query upper bounds use the
minimum of a blockwise Cauchy bound and the node's maximum norm bound,
with rounding slack. Entire subtrees can be pruned only when they are
provably incapable of entering the current top 100.

A leaf uses the reordered 8-bit quantized signatures with the original
quantization error bound. A promising candidate is ranked against its
unaltered stored float32 vector in float64 and witness verified.

HingeZero zero-step cue refinement remains an actual no-op call, not a
disabled memory search.

To assess whether this works, compare top-100 ID lists bit-for-bit and
query throughput to the submitted 16-bit reference on the same GitHub
runner; report all results including regressions. Target 500+ QPS at
measured recall@100 1.00000, not assumed.
