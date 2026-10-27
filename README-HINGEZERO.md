# HingeZero VIBE experimental integration

HingeZero two-step associative-field refinement over an IVF shortlist. The HingeZero component modifies candidate ranking; FAISS IVF remains the underlying search index.

This is an experimental implementation, not a verified independent VIBE score. The zero-step control isolates the effect of the two HingeZero refinement steps.

Parameters: alpha=0.25, epsilon=0.10, lambda=0.02, field=64 candidates, 0 versus 2 refinement steps.

For official VIBE evaluation:
  ./install.sh --algorithm hingezero
  python3 run.py --dataset yi-128-ip --algorithm hingezero-ivf --count 100
  ./export_results.sh --parallelism 1

For reproducible independent inspection using original VIBE dataset and provided neighbor ground truth:
  python -m pip install numpy==1.26.4 h5py==3.11.0 faiss-cpu==1.12.0
  curl -fL -o yi-128-ip.hdf5 https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/yi-128-ip.hdf5
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/bench_hz_official.py --dataset yi-128-ip.hdf5 --queries 300

This script reports unranked QPS and recall@100 on the specified subset of queries, alongside a file SHA256 and hardware information. Do not compare the output against hardware-mismatched official leaderboard numbers. Full VIBE evaluation must be performed separately on VIBE hardware using their runner and configuration. No accuracy or speed gains are asserted without measured evidence.
