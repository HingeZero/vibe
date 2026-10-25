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

## 25 October 2026: Verified results

[Author-operated successful hybrid CI run](https://github.com/HingeZero/vibe/actions/runs/40172479284)
([VIBE result artifact](https://github.com/HingeZero/vibe/actions/runs/40172479284/artifacts/12590293691))
on 187,843 original 128-dimensional `yi-128-ip` float32 records:

| Operation | Measured QPS | Verification |
|---|---:|---|
| Single native exact lookup via Python ctypes | 326,832.72 | bitwise original vector identity |
| One native-batched exact lookup call over 50,000 cues | **683,865.59** | bitwise original vector identity |
| Direct ID retrieval via Python ctypes | 732,654.29 | original record returned, witness check |
| VIBE nearest-neighbour top-100 | **472.71** | recall@100 **1.00000**, 5 timed runs |

Full bank build (including writing original bank and opening both index
types) was **0.381 seconds** in the separate exact-key test. The VIBE
five-run measurement reported build time **0.36 seconds**. These
measurements are host dependent and do not certify 1B-record scale.

The indexed route hashes every byte of the float32 pattern, so expected
complexity is **O(d)** plus expected **O(1)** hash lookup (for constant
dimension d), not truly constant with growing vector dimensions.
Direct ID lookup copies O(d) bytes and is constant in record count.
The output for an absent exact key is `None`, not a guessed identity.
Collision matches must compare the original bytes, and duplicates
return the oldest ID. Nearest-neighbour ranking continues through
the certified HingeZero search—not this hash route.

The reported QPS applies only to the specified operation; hundreds of
thousands of exact-key lookups per second is **not** a VIBE result and
does not imply O(1) top-k similarity matching.

## Matched-host validation (25 October 2026)

[Successful native parity and VIBE run](https://github.com/HingeZero/vibe/actions/runs/40172661463)
([raw result artifact](https://github.com/HingeZero/vibe/actions/runs/40172661463/artifacts/12590204377)).

The reference exact-pruning C++20 engine (without hash table) and the
new hybrid both ran against the same 187,843-record VIBE `yi-128-ip`
corpus on the **same GitHub CPU runner**, using the same 256 queries,
dimension 128 and k=100, each in alternating order twice.

| Native engine | Matched-host mean top-100 QPS | Mean bank/index build seconds |
|---|---:|---:|
| Certified top-k before hybrid | **296.96** | ~0.134 |
| Hybrid + exact hash index | **297.48** | ~0.427 |

All 256 top-100 ID arrays matched exactly, across every pass. Mean
QPS ratio hybrid/prior: **1.002×**. This demonstrates that the
additional exact-key interface does not materially speed up or slow
down the existing nearest-neighbour query on the tested host.
The higher 472.71 QPS reported in another author-run GitHub worker
was NOT a matched-hardware improvement, so should not be attributed
to this hash index.

The hybrid's own five-run VIBE result in this run was
**295.91 QPS at recall@100=1.00000**, with build ~0.49 seconds.

The VIBE-reported additional process unique-memory index size was
155,880 KiB for the hybrid run versus 145,488 KiB for the earlier
non-hybrid run (about 10.1 MiB difference). These are
different GitHub-hosted runners and process USS measurements, **not**
an isolated byte-accurate accounting for the hash index.

### Public API example

```python
import numpy as np
from vibe.algorithms.hingezero_cpp20.module import HingeZeroCpp20

bank = np.random.default_rng(1).normal(size=(1000, 128)).astype(np.float32)
hz = HingeZeroCpp20("ip", storage_path="/tmp/my-hingezero-hybrid-bank")
hz.fit(bank)

exact = hz.lookup_exact(bank[20])   # ID 20, source, original bytes, witness
missing = hz.lookup_exact(np.ones(128, dtype=np.float32))  # None
by_id = hz.lookup_by_id(20)         # original exact float32 vector

# An unfamiliar/noisy cue is handled by similarity retrieval, not
# mistakenly treated as an exact identity.
neighbors = hz.query(bank[20] + 0.001, 100)

# Efficient contiguous native batch lookup; absent IDs become -1.
ids = hz.lookup_exact_batch(bank[[2, 5, 20]])
```

Set the `HINGEZERO_CPP20_LIB` environment variable to the shared object
compiled from this development branch (as in the workflow). Use a new,
empty storage directory to prevent overwriting pre-existing memories.

**Deployment caveats:** bitwise float32 identity does not tolerate a
single modified input bit, the in-memory hash table grows linearly
with the number of records, and the VIBE float32 bank is not the
published 1-bit billion-memory engine. Those larger/bipolar deployments
require their own storage/index design and benchmarks; nothing here
establishes O(1) nearest-neighbour search or 1B-record scale.
