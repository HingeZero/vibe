# HingeZero C++20 candidate-index investigation

**Development only — the upstream VIBE pull request retains the one
zero-refinement algorithm and the verified exhaustive SIMD scan.**

The index is deterministic spherical coarse clustering trained only on
stored corpus vectors (no labels or VIBE query ground truth). It creates
128 inverted lists, stores each original float32 record unchanged, and
searches the top `nprobe` lists according to centroid similarity.

The same HingeZero C++20 `refine(original,0)` pipeline, exact inner
products on candidate records, and original-byte witness verification
remain in place. No FAISS dependency is used. The native IVF index copies
the float32 records into contiguous posting lists for faster CPU scanning,
which increases memory usage. Newly appended records cause the index to
rebuild when the bank is remounted; the original append-only bank stays
unchanged.

This is **approximate** inner-product retrieval. Searching fewer lists
cannot mathematically guarantee 100% recall. A full `nprobe=128` search
provides an exact baseline and all returned IDs are witness-checked. Only
one final configuration should be submitted if the recall/QPS tradeoff is
acceptable, keeping `steps=0`.

Six diagnostic probe budgets: 4, 8, 16, 32, 64, 128, with five timed runs
and official VIBE recall@100. Do not claim an improvement until the
GitHub-hosted runner completes and validates the dataset.
