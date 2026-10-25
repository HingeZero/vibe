# Exact C++20 HingeZero AVX512 speed investigation

This branch retains a **single zero-refinement HingeZero C++20 algorithm**,
unchanged original float32 bank, exhaustive scan of each stored record,
native source witness and append-prefix preservation. No approximation.

CPU dispatch selects float32→float64 AVX-512 double-precision inner
product when available, AVX2 otherwise, and scalar otherwise. The
original VIBE exact metric remains in force. The unchanged previous AVX2
kernel is compiled separately only for same-machine performance and exact
returned-ID validation, not submitted as a second algorithm.

Only promote if observed VIBE QPS improves meaningfully on repeat
measurements with recall@100 remaining 100%.
