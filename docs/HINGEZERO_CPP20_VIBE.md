# HingeZero C++20 original-binary-source and VIBE adaptation

Author: David Duffy / HingeZero. Method provenance:
[HingeZero/HingeZero-1B-Memory/HingeZero_Native_1B.cpp](https://github.com/HingeZero/HingeZero-1B-Memory/blob/main/HingeZero_Native_1B.cpp).

**Files:**
- native/HingeZero_Native_1B.cpp: bundled *unchanged original* C++20 128-bit bipolar engine, including its 8-step dual-tanh field, full billion-record streamed scan, append and witness.
- native/HingeZero_VIBE_Cpp20.cpp: C++20 adaptation for arbitrary float32 embedding dimensions and native VIBE Euclidean/IP/cosine scoring. Uses the same fixed first-256 local associative field, alpha=4.5, eps=0.10, lambda=0.02, steps=8, a whole-bank scan, and tagged-state verification. No FAISS and no one-bit sign-filter.
- module.py: thin ctypes/BaseANN adapter which stores and appends original float32 vectors without altering the VIBE train set.
- config.yml: **refined** (8 HingeZero iterations), **zero** (the same HingeZero refine/query/witness pathway with 0 iterations), and **anchored** (8 iterations with explicit original-query anchoring).

**Dataset adaptation:** \`yi-128-ip\` is a 128-dimensional real-valued inner-product dataset. The unchanged binary engine's 16 bytes/record can only represent bipolar 128-bit patterns. Unlike the earlier lossy sign-index attempt, we preserve all original float coordinates using 4*d bytes/record. 1-bit compression is not claimed for this dataset; to test the unchanged packed engine use its original bipolar smoke / 1B protocol, not yi-128-ip.

**Benchmark interpretation:** \`zero\` invokes the original native HingeZero refine/query/witness pathway with *zero iterations* (expected recall@100 near 1.0 on original-query VIBE) and is *not* evidence that HingeZero improves recall; \`refined\` faithfully ranks the full bank against the altered cue and may have lower VIBE recall because VIBE ground truth is defined by the **unaltered cue**. Both measure full scan CPU cost, *not* approximate-index speed. Their performance should not be compared to the previous FAISS-assisted 90.46% recall result as if they used the same search algorithm. There is no claim of cloud-free power savings, encrypted semantic memory, or billion vector capacity on VIBE.

**Witness limitation:** tag verification means an ID corresponds exactly to the originally stored float record; it does not establish that this is the intended target or that source text is true.

**Reproduction** (local install of g++20, NumPy and VIBE deps; test step compiles the kernel):
\`\`\`sh
g++ -O3 -std=c++20 -fPIC -shared vibe/algorithms/hingezero_cpp20/native/HingeZero_VIBE_Cpp20.cpp -o /tmp/libhz20.so
export HINGEZERO_CPP20_LIB=/tmp/libhz20.so
python verification/smoke_hingezero_cpp20.py
python run.py --local --dataset yi-128-ip --algorithm hingezero-cpp20-refined --count 100 --runs 1
python run.py --local --dataset yi-128-ip --algorithm hingezero-cpp20-zero --count 100 --runs 1
\`\`\`

The attached GitHub Actions workflow compiles original + adapted C++20, checks parity with the original 128-bit pattern bank, verifies append-prefix preservation and native witnesses, and measures both variants through the VIBE runner. Results are author-run until accepted by VIBE maintainers.


## VIBE real-valued query-anchoring (third variant)

\`hingezero-cpp20-anchored\` preserves the *original C++20 field, nonlinearity
and eight-step refinement*, then chooses an experimentally explicit
**95% original-unit-query / 5% refined-query** direction before the full scan.
This aims to avoid distorting native real-valued IP ranking merely due to a
uniform-evidence field. It is a deliberate **dataset adaptation**, not the
unaltered original binary-engine score or proof of improved index performance.
The unchanged \`refined\` and exact \`raw\` modes remain controls.


## Zero refinement is not a bypass

For `hingezero-cpp20-zero`, the constructor sets `mode="refined"` and the query argument `steps=0`. The C++20 kernel enters the normal HingeZero `refine()` function, validates the input, returns `x_0` unchanged without iterative state updates, then executes the same full-bank scanning and tagged-state witness verification as the eight-step variant. The separate `raw` bypass remains an internal diagnostic only; it is no longer registered as the benchmark's zero-step control.

The 0/8-step comparison therefore changes the iteration count, not the retrieval implementation. Because VIBE's ground truth uses the original query, zero-step exhaustive scan should return 100% recall@100, without implying a nontrivial HingeZero accuracy gain.
