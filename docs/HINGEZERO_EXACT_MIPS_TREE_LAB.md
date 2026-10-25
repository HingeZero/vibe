# Exact hierarchical MIPS experiment

Separate experimental branch. Does not change the submitted VIBE entry.

The original float32 vectors remain authoritative. An offline binary
ball-tree stores mean, enclosing radius and original record IDs.
For a query q, the node upper bound is dot(q,mean)+norm(q)*radius
with a small numerical safety allowance. Best-bound-first traversal
skips subtrees only when they cannot enter the current exact top 100.

No approximation or 2-bit reconstruction is used in final ranking.
The zero-step HingeZero refine function is still called, and witness
verification remains enabled. The reference exact C++20 implementation
must return identical ranked top-100 IDs on matched VIBE queries.

Success criterion: >=500 single-core VIBE QPS with 100% measured recall
and matched-reference top-100 parity. Otherwise do not promote.

This is an established ball-tree branch-and-bound method, not a claim
that hierarchical MIPS itself is novel.
