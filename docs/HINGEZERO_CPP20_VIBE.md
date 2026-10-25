# HingeZero C++20 original-binary-source and VIBE adaptation

Author: David Duffy / HingeZero. Method provenance:
[HingeZero/HingeZero-1B-Memory/HingeZero_Native_1B.cpp](https://github.com/HingeZero/HingeZero-1B-Memory/blob/main/HingeZero_Native_1B.cpp).

**Files:**
- native/HingeZero_Native_1B.cpp: bundled *unchanged original* C++20 128-bit bipolar engine, including its 8-step dual-tanh field, full billion-record streamed scan, append and witness.
- native/HingeZero_VIBE_Cpp20.cpp: C++20 adaptation for arbitrary float32 embedding dimensions and native VIBE Euclidean/IP/cosine scoring. Uses the same fixed first-256 local associative field, alpha=4.5, eps=0.10, lambda=0.02, steps=8, a whole-bank scan, and tagged-state verification. No FAISS and no one-bit sign-filter.
- module.py: thin ctypes/BaseANN adapter which stores and appends original float32 vectors without altering the VIBE train set.
- config.yml: two fair variants, **refined** (HingeZero cue changes the full-bank ranking) and **raw** (the same C++20 scanning kernel without refinement).

**Dataset adaptation:** \`yi-128-ip\` is a 128-dimensional real-valued inner-product dataset. The unchanged binary engine's 16 bytes/record can only represent bipolar 128-bit patterns. Unlike the earlier lossy sign-index attempt, we preserve all original float coordinates using 4*d bytes/record. 1-bit compression is not claimed for this dataset; to test the unchanged packed engine use its original bipolar smoke / 1B protocol, not yi-128-ip.

**Benchmark interpretation:** \`raw\` is the reference exact full scan (expected recall@100 near 1.0) and is *not* evidence that HingeZero improves recall; \`refined\` faithfully ranks the full bank against the altered cue and may have lower VIBE recall because VIBE ground truth is defined by the **unaltered cue**. Both measure full scan CPU cost, *not* approximate-index speed. Their performance should not be compared to the previous FAISS-assisted 90.46% recall result as if they used the same search algorithm. There is no claim of cloud-free power savings, encrypted semantic memory, or billion vector capacity on VIBE.

**Witness limitation:** tag verification means an ID corresponds exactly to the originally stored float record; it does not establish that this is the intended target or that source text is true.

**Reproduction** (local install of g++20, NumPy and VIBE deps; test step compiles the kernel):
\`\`\`sh
g++ -O3 -std=c++20 -fPIC -shared vibe/algorithms/hingezero_cpp20/native/HingeZero_VIBE_Cpp20.cpp -o /tmp/libhz20.so
export HINGEZERO_CPP20_LIB=/tmp/libhz20.so
python verification/smoke_hingezero_cpp20.py
python run.py --local --dataset yi-128-ip --algorithm hingezero-cpp20-refined --count 100 --runs 1
python run.py --local --dataset yi-128-ip --algorithm hingezero-cpp20-raw --count 100 --runs 1
\`\`\`

The attached GitHub Actions workflow compiles original + adapted C++20, checks parity with the original 128-bit pattern bank, verifies append-prefix preservation and native witnesses, and measures both variants through the VIBE runner. Results are author-run until accepted by VIBE maintainers.
