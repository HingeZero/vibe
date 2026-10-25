# HingeZero C++20 VIBE ANN graph laboratory

One **C++20 HingeZero** algorithm, no FAISS, external ANN shortlist, or
competing algorithm submitted. Target dataset: `yi-128-ip`, 187,843
float32 vectors, 128 dimensions, 100 nearest-neighbour IDs.

The native HingeZero memory engine keeps the append-only original
float32 bank, witness validation and the same `refine(q,0)` entrypoint.
**Zero refinement steps means no iterative bit corrections; it does not
disable memory retrieval or witnesses.**

An offline deterministic C++20 navigable proximity graph is constructed
from the stored corpus **without evaluation queries or ground truth**.
It uses an exponentially layered memory navigator, 24 bottom-layer
connections, 12 connections per upper layer, construction exploration
70, greedy entry descent, and candidate beam search. Candidate records
are reranked using their original float32 inner-product scores in
float64 arithmetic. Witness checks validate returned IDs against the
original bytes.

Five `ef_search` values within **one ANN engine** are tested for a
recall@100 versus QPS curve: 128, 256, 512, 1024, 2048.
No result is claimed before tests. A good result needs recall >=0.99
at the highest possible single-core VIBE QPS; the development goal is
5,000 QPS with recall >=0.99, but that is not guaranteed.

HNSW-like navigable proximity graphs are well-established prior art;
the goal is a VIBE-specific memory indexing implementation for HingeZero,
not claiming invention of the general graph navigation technique.
Original VIBE pull request #16 is **not modified** until validation
establishes a meaningfully competitive result.
