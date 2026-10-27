# HingeZero VIBE dataset reproduction, 27 October 2026

Data: official yi-128-ip.hdf5 (SHA256 23d76fdbb0808909d02be83f95c074c3ccf54018b03a80d54e8e31ad1815071e).
Size: 187,843 vectors x 128 dimensions; 300 queries, first test rows; inner product; recall@100.
Host: GitHub Actions Ubuntu (Linux 7.0.0-1017-azure), CPython 3.11.17, NumPy 1.26.4, FAISS 1.12.0. One CPU thread. nlist=512, nprobe=16, pool multiplier=4.
Run: https://github.com/HingeZero/vibe/actions/runs/40492568304

Configuration | Recall@100 | Queries/s
--- | --- | ---
0-step control | 0.7220667 | 3473.265
2-step HingeZero | 0.6898333 | 2975.268

Two-step refinement reduced recall by 0.0322333 absolute and query throughput by roughly 14.3% in this run.
Build time was 4.367 seconds. Results are for one parameter choice, one host run, 300 of the dataset's queries, and do not establish official VIBE leaderboard standing. This negative result is retained as recorded, not replaced by favorable synthetic measurements. Follow official VIBE benchmarking instructions for full hardware-controlled comparison.
