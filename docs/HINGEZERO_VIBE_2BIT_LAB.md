# HingeZero VIBE packed two-bit experimental index

Separate lab branch. Never change the submitted C++20 algorithm.

- 2 bits per dimension: four quantization levels per vector, stored as two bitplanes
- Each 128-dimensional signature occupies 32 bytes instead of the 128 bytes in the 8-bit lab or 256 bytes in the submitted 16-bit index
- Min/max record and query quantization; popcount binary-plane dot products
- Conservative residual-norm score bounds, original float32 exact rerank, witness validation
- Zero HingeZero iterative refinement steps; search and verification remain active
- The intact original float32 memory bank is retained; 2-bit storage affects filtering only

Success requires exact ranked top-100 agreement with submitted HingeZero
on identical VIBE queries and a meaningful same-worker QPS gain.
Record candidate full-scoring count because loose bounds can make the
smaller signature slower. 500 QPS remains a target, not a result.
