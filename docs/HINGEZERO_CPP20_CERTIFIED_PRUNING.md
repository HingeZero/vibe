# HingeZero C++20 certified 16-bit candidate pruning experiment

This is a **development-only experiment**; [VIBE PR #16](https://github.com/vector-index-bench/vibe/pull/16)
remains the one registered C++20 HingeZero zero-step exhaustive SIMD algorithm.

The proposed native C++20 replacement creates one compact signed-int16 vector
representation per original float32 memory record. It quantizes each record
independently, stores its exact residual norm, quantized norm and scale, and
uses a signed 16-bit AVX2 dot product to generate a cheap approximate score
`a`.

For query q and its quantization q', and original vector x / quantized x':

`x·q = x'·q' + (x-x')·q + x'·(q-q')`

By Cauchy-Schwarz:

`x·q <= a + ||x-x'|| ||q|| + ||x'|| ||q-q'||`

The stored upper bound adds a small margin against finite-precision effects.
First, approximate top-k records are evaluated against the *original* float32
bank for a real kth-place threshold. Every other vector with an upper bound
above the current exact threshold is evaluated exactly. All others are
**certifiably excluded**; the original top-k exact ranking is preserved, with
the original HingeZero witness checks. No FAISS, no refinement steps, and
no heuristic nprobe cutoff.

An extra ~256 bytes/record is used for 16-bit signatures, plus three doubles.
The index is rebuilt at bank open, so index construction costs and append
update economics must be reported. This experiment currently supports
inner-product/normalized metrics using pruning; other metrics fall back to
exact unpruned scanning.

The test is only promoted if the full VIBE `yi-128-ip` benchmark confirms
both 100% recall and higher, repeatable throughput than the existing SIMD
submission. No performance improvement is presumed.
