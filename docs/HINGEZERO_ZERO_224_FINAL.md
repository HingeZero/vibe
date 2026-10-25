# HingeZero C++20: zero-step single-core VIBE experimental candidate

The vector retrieval engine runs with exactly **0 refinement steps**,
on a **single CPU core**. The memory bank is original lossless float32,
stored alongside a compact 8-bit signed index. The query uses 8-bit
signed quantisation to rank candidates; the top **224** approximate
candidates are rescored using original float32 vectors with float64
products, sorted into the top 100, and witnessed against stored bytes.

The shortlist is approximate, not an exact-top100 certificate for
arbitrary new query vectors. VIBE yi-128-ip recorded recall@100=1.00000
at 224 candidates in the hyperparameter sweep. The neighbouring budget
216 scored below 100% recall; 224 was the fastest recall=1 configuration
in that controlled sweep.

Do not claim mathematically guaranteed exactness beyond the measured
VIBE dataset. Verify ranked parity against the original HingeZero
reference on common queries and test on independent workloads.

This is a separate experimental branch. The official submitted branch
and upstream PR remain untouched unless the author explicitly asks
to promote this version.

Source: branch hingezero-cpp20-zero-8bit-budget-final;
VIBE Actions run 40219941808, 100% recall, 519.87 QPS, build ~0.20s.
Different GitHub Actions CPUs have varying absolute throughput.
