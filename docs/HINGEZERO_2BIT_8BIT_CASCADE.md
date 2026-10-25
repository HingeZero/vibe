# Packed 2-bit / 8-bit certified HingeZero VIBE cascade

The fast exact-match VIBE pathway retains the packed two-bit vector representation
but uses a second eight-bit quantised index as a **certification accelerator**.
This is **not** a two-bit-only memory footprint or a demonstration of pure
two-bit certified search. It retains the original 32-bit float vectors.

1. Candidate ordering uses packed 2-bit code signatures.
2. Full-precision exact scores seed the top-100 heap.
3. Conservative 8-bit quantisation-error bounds discard candidates that
   cannot enter the exact top 100.
4. Original float32 vectors are ranked with float64 products; witness
   validation verifies original stored records.

No iterative refinement is performed at steps=0. The original memory
search remains active. Only promote this approach if it gains throughput
and preserves exact ranked top-100 parity on the same runner.
