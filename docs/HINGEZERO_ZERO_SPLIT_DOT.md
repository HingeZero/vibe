# VIBE zero-step AVX2 fast 8-bit signed-dot experiment

Experiments keep the same signed 8-bit code, same 8-bit query quantizer,
same original float32 rerank, same native witness checks, same
single-core VIBE runner and steps=0. Shortlist fixed at 256.

AVX2 vpmaddubsw accepts unsigned×signed bytes but saturates at 16 bits.
We avoid saturation without lossy re-quantization:
  unsigned code = signed code + 128
  signed q = 2*floor(signed q/2) + parity(q), where parity is 0 or 1
  exact dot = 2*sum(unsigned code*half_q) + sum(unsigned code*parity_q)
              - 128*sum(q)
With signed q within [-127,127], half magnitude <=64;
each pair sum has magnitude <=32640 and cannot saturate.

We test full VIBE recall and exact ranked-ID parity against the
submitted certified reference and the existing 8-bit scorer on
the same CPU worker. The approximate shortlist still lacks a general
exact-top100 guarantee for arbitrary unseen cues.
