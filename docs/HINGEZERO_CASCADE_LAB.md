# HingeZero C++20 zero-step PCA cascade laboratory

**Objective:** 5,000 QPS at exact top-100 recall on yi-128-ip.

Uses a corpus-only learned, orthonormal 128-dimensional eigenbasis and
staged signed-8-bit projection blocks. For each query, it scores a 1,024
record seed set exactly against the original float32 bank; the seed set
is generated using only eight projected dimensions, not test labels.

For each remaining record, a certified upper bound is calculated and
tightened after 8, 16, 32, 64, 96, 112, then all 128 principal
components. Bounds account for quantization residual error and
unexamined orthogonal subspace, plus numerical guard. Any record
whose bound is below the current exact kth score can be discarded
without affecting exact top-100 on correctly bounded inputs.
Every returned record is then witness-verified by HingeZero.
The persistent original float32 bank and zero iteration setting remain.

Index build performs deterministic Jacobi PCA on 2,048 corpus samples
and rotates the whole bank to int8 stage-major blocks. Build cost,
extra index RAM, numerical robustness, and append rebuild cost must
be reported. This is not the HingeZero associative correction step;
`steps=0` throughout, and no extra VIBE algorithms are registered.

Experiment selected by HINGEZERO_CASCADE=1 only on this development
branch. Do not promote if accuracy, correctness smoke, or repeatable
QPS is worse than the previously submitted ~288 QPS exact variant.
