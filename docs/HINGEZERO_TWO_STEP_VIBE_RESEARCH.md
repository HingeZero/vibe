# HingeZero Two-Step VIBE Research

This isolated branch activates exactly **two** iterations of the native dual-tanh HingeZero refinement before exact candidate search.

- VIBE dataset: `yi-128-ip`, `k=100`, single CPU core, five runs.
- Refiner uses the existing 256 unit-normalized field memories, `alpha=4.5`, `epsilon=0.1`, `lambda=0.02`, with an `L2` normalization of the query state on each step.
- Retrieval computes exact original float32 scores against the **refined** query and verifies original-vector witnesses.
- VIBE measures recall against the **original unrefined query**. Refinement changes the question being asked; it can reduce the official recall even when search is exact for the refined question.
- No promotion to the submitted zero-step PR without review and explicit consent.
