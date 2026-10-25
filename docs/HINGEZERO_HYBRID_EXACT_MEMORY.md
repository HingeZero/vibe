# HingeZero hybrid O(1)-in-bank exact identity + associative nearest-neighbour memory

**Experimental branch only.** No modification to VIBE PR #16 until validation.

Three operations implemented in the **same** C++20 HingeZero memory object:

1. `lookup_by_id(id)`: directly reads original float32 record from the
   memory-mapped bank, with original-ID witness validation. O(d) to copy
   a d-dimensional vector; independent of total stored record count.
2. `lookup_exact(vector)`: uses a hash table mapping immutable original
   float32 *bytes* to their oldest stored ID, then verifies the stored
   record and returns source metadata when present. **O(d) to hash the
   full key, expected O(1) table lookup in bank size**, for fixed d.
   Hash collision resolution verifies full byte equality. Duplicate
   bitwise-identical vectors return the oldest ID; all IDs remain readable.
   The HingeZero native `refine(q, 0)` entrypoint is invoked to validate
   the cue without any bit-correction iterations.
3. `query(cue, k)`: preserves the existing HingeZero C++20 certified
   score-bound candidate pruning, original float32 inner products, exact
   ranked top-100 and witness verification. It remains a nearest-neighbour
   computation and is **not** advertised as O(1).

Exact float32 bit identity is **not** semantic similarity, noise-tolerant
recognition, or the same as a top-100 VIBE nearest-neighbour answer.
Noise, partial cues, or bit flips use the associative nearest-neighbour
path, with optional HingeZero correction for non-VIBE cases. VIBE keeps
`steps=0`. There is no automatic fast-path substitution in VIBE because
an exact matching vector need not have the largest unnormalized inner product.

The hash table consumes O(P) additional RAM for P indexed records and
is rebuilt on each mount or append to preserve deterministic mapping
and append-only stored-byte integrity. An in-memory hash table is not
practically free at billion-record scale; exact-key retrieval is constant
*expected time in record count*, not constant RAM or an O(1) nearest
neighbor algorithm. Native/ctypes benchmark timings and VIBE top-k timings
are reported **separately**.
