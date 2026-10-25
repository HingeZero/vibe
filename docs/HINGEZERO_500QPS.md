# HingeZero VIBE 500+ QPS experiment

No changes to the submitted branch or to ANN ranking semantics.
This lab uses 8-bit quantised bound signatures (rather than 16-bit),
an AVX2/AVX-512BW/scalar runtime-dispatched inner product, and the
unchanged lossless original-memory scoring and witness verification.

The quantization error terms remain in the score bound. Every 100-ID
answer must equal the preceding exact C++20 implementation in a
same-worker matched comparison on VIBE test queries.

The author-operated result is **not** an official VIBE leaderboard
entry, and cross-worker speed comparisons are not treated as wins.

Success: single-core VIBE >=500 QPS at measured recall@100=1.00000,
and zero-step HingeZero query/witness tests pass. Otherwise record the
actual performance and do not promote the branch.
