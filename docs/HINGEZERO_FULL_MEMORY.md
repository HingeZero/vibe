# Standalone HingeZero memory engine for VIBE

This `hingezero-full` contribution is **not FAISS-assisted**. It has an
independent on-disk, append-only float32 record bank plus one-bit-packed
signatures; candidate retrieval uses bounded-memory Hamming scan instead of a
third-party index. It implements David Duffy's eight-step dual-tanh
associative-memory field, explicit stored-state witnesses, persistent append
operations, source references, optional abstention and independent session
retention. An identical packed-search control without the HingeZero recurrence
is registered as `hingezero-packed-control`.

## Design and correctness

- `fit` appends every corpus record without training on queries or ground truth.
- Exact source vectors remain on disk to preserve inner-product, L2 and cosine
  scoring; packed sign bits are only a lossy candidate index.
- The 1-bit index uses O(P * ceil(d/8)) scanning work per query, chunked to
  keep temporary memory bounded. Neither O(1) total I/O nor faster-than-FAISS
  search is claimed. The binary signature alone is not a sufficient statistic
  for arbitrary real-vector nearest-neighbour ordering.
- Final rankings use **original query** and exact original float32 candidates,
  not a different distance definition. HingeZero influences the second search
  candidate set through 8-step local field refinement.
- New records are appended to both files with unchanged original-row bytes.
  Reopen validates expected sizes. A witness checks that a returned record
  matches the original stored vector; it cannot verify external factual truth.
- Source metadata and abstaining single-memory recall live outside VIBE's ANN
  `query` API; VIBE requires exactly k record IDs. A sequential
  `RetainingSession` is separately available and deliberately not shared
  between independent benchmark queries.
- Raw source metadata is *unencrypted*. This is **not** a care product, secure
  vault, or semantic encoder. The separate secure-memory project has access
  controls, consent and encryption not reproduced here.
- Float32 storage uses d*4 bytes/record **in addition** to packed d/8 bytes.
  Capacity is limited by available disk and VIBE's int32 ID format. We do not
  claim that this VIBE adapter has tested one billion records.

## Testing
```sh
python3 verification/smoke_hingezero_full.py
python3 run.py --local --dataset yi-128-ip --algorithm hingezero-full --count 100 --runs 1
python3 run.py --local --dataset yi-128-ip --algorithm hingezero-packed-control --count 100 --runs 1
```

Use the GitHub Actions workflow on the dedicated development branch to run
both implementations on the same cloud hardware. These are author-run
measurements until independently reviewed by VIBE maintainers.
The pre-existing `hingezero` FAISS-assisted integration is retained as a
separate benchmark, not silently replaced.
