# Zero-step HingeZero mixed 8x16-bit candidate scan

Experimental single-core CPU ANN path, preserving the original float32
memory bank, native HingeZero steps=0 call and tagged witness verification.

8-bit signed record signatures remain compact (128B for 128 dimensions).
The query is quantised to **16-bit**, reducing query quantisation error
so the exact float32 top100 is more likely to fit a small shortlist.

The shortlist is NOT mathematically certified. Benchmark recall@100=1.0
is a measured property on the tested data, not a proof for every cue.

AVX2 uses signed-int8 widening followed by signed-int16 dot products;
the scalar path is a safe fallback. Dot accumulation is chunked to avoid
signed int32 overflow on wider vectors. Exact rescoring uses float64.

Candidate budgets are explicit VIBE query hyperparameters, with original
VIBE indexing, single-core runner, and original witnesses retained.
Never change the original submitted PR branch without separate approval.
