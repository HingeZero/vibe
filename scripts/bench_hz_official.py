import argparse
import hashlib
import json
import os
import platform
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import faiss
import h5py
import numpy as np

from vibe.algorithms.hingezero.module import HingeZeroIVF


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--queries', type=int, default=0)
    p.add_argument('--nlist', type=int, default=512)
    p.add_argument('--nprobe', type=int, default=16)
    p.add_argument('--candidate-multiplier', type=int, default=4)
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--output', default='hz_vibe_results.json')
    a = p.parse_args()
    with h5py.File(a.dataset, 'r') as h5:
        X = h5['train'][:]
        Q = h5['test'][:a.queries or None]
        gt = h5['neighbors'][:len(Q), :a.k]
        metric = h5.attrs['distance']
        if isinstance(metric, bytes):
            metric = metric.decode('utf-8')
    if a.k > gt.shape[1]:
        raise ValueError('Dataset has fewer ground truth neighbors than requested k')
    if len(Q) != len(gt):
        raise ValueError('Query and ground truth lengths do not match')
    algorithm = HingeZeroIVF(metric, a.nlist, a.candidate_multiplier)
    build_start = time.perf_counter()
    algorithm.fit(X)
    build_s = time.perf_counter() - build_start
    results = []
    for steps in (0, 2):
        algorithm.set_query_arguments(a.nprobe, steps, 0.2)
        for q in Q[:min(10, len(Q))]:
            algorithm.query(q, a.k)
        tic = time.perf_counter()
        got = [algorithm.query(q, a.k) for q in Q]
        elapsed = time.perf_counter() - tic
        correct = sum(len(set(g.tolist()) & set(t.tolist())) for g, t in zip(got, gt))
        results.append({'steps': steps, 'recall_at_k': correct / (len(Q) * a.k),
                        'qps': len(Q) / elapsed, 'total_queries': len(Q),
                        'elapsed_seconds': elapsed})
    output = {'protocol': 'offline VIBE-dataset-compatible comparison; not official VIBE leaderboard',
              'dataset': os.path.basename(a.dataset), 'dataset_sha256': sha256(a.dataset),
              'train_rows': len(X), 'dimension': X.shape[1], 'query_rows': len(Q),
              'metric': metric, 'k': a.k, 'build_seconds': build_s,
              'nlist': a.nlist, 'nprobe': a.nprobe,
              'candidate_multiplier': a.candidate_multiplier,
              'alpha': 0.25, 'eps': 0.1, 'lam': 0.02, 'blend': 0.2,
              'cpu': platform.processor(), 'platform': platform.platform(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'faiss': getattr(faiss, '__version__', 'unknown'), 'results': results}
    with open(a.output, 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
