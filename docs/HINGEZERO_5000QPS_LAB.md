# HingeZero 5,000 QPS lab — exact 8-bit certified bounds

Target: reach 5,000 QPS on VIBE yi-128-ip at **100% exact
recall@100** without altering the single zero-step C++20 HingeZero
architecture. Target is **not a claimed achieved result**.

Development branch only: original HingeZero memory, append-only
float32 source bank, same HingeZero native `refine(q,0)`, same record
witness, same exact ranking. Quantized approximate score uses 8-bit
signed signatures (128 bytes per vector) instead of 16-bit signatures
(256 bytes per vector), retaining float32 records losslessly on disk.

Cauchy–Schwarz conservative bound and exact reranking remain:
`x·q ≤ x'·q' + ||x−x'|| ||q|| + ||x'|| ||q−q'||`.
AVX2 widens signed int8 to int16 before vector integer dot.
The reduced signature bandwidth may improve QPS at the cost of
looser score bounds and more exact float32 scoring. All candidate
pruning must remain correctness-preserving.

Do not update upstream PR until GitHub VIBE and exact-ID parity
pass, and measured throughput exceeds the current ~288–292 QPS.
