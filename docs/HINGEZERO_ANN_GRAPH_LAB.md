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

## Initial graph benchmark diagnosis and MIPS lift experiment

[Graph v1 VIBE run](https://github.com/HingeZero/vibe/actions/runs/40173668235)
produced 7.11–11.01% recall@100 across ef_search 128–2048.
Despite 502–3,766 QPS, the first graph failed ANN quality. Build 124.72s.

The correction here is to **construct the graph in a metric-compatible
transformed space**: y=(x,sqrt(R²-||x||²)), with R² slightly larger
than the corpus maximum ||x||². Then ||y|| is constant and graph-build
navigation is related to Euclidean distance between corpus vectors.
For VIBE queries transformed to (q,0), the search similarity
is q·y=q·x, preserving maximum-inner-product candidate ranking.

This conversion is established MIPS indexing prior art, not a
new HingeZero mathematical claim. It changes only graph construction;
the original memory bank, zero-step HingeZero cue, real float32
inner-product reranking, and witness validation remain unchanged.
Larger M/efConstruction are an intentional build-time/recall tradeoff.

No promotion unless recall and QPS materially justify it.
