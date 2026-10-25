# C++20 HingeZero speed lab (isolated development branch)

The VIBE PR's single submitted algorithm remains HingeZero C++20 with
`steps=0`: **not** a bypass, no other variants or FAISS.

This experimental branch accelerates the exact all-record inner-product
scanning hot loop through a runtime-selected AVX2 path. Float32 coordinates
are promoted to double before multiplication to preserve VIBE's original
metric and avoid changing stored data. The original C++20 associative
architecture, immutable append prefix, witness checks, and full scanning
semantics remain unchanged. The alternative scalar path remains available
for non-AVX2 CPUs. No approximation or skipped records introduced.

Performance target: improve measured QPS without reducing recall@100
from 1.00000. SIMD speed improvements are hardware-dependent and should
be compared on an otherwise identical GitHub Actions machine. Further
major scaling would likely require a suitable approximate index, which
necessarily changes the recall/speed trade-off.
