import faiss
import numpy as np

from ..base.module import BaseANN


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class HingeZeroIVF(BaseANN):
    def __init__(self, metric, n_list, candidate_multiplier):
        if metric not in ('cosine', 'normalized', 'ip', 'euclidean'):
            raise ValueError(f'Unsupported metric: {metric}')
        self.metric = metric
        self.n_list = int(n_list)
        self.candidate_multiplier = int(candidate_multiplier)
        self.alpha = 0.25
        self.eps = 0.10
        self.lam = 0.02
        self.field_size = 64
        self.set_query_arguments(16, 2, 0.2)

    def fit(self, X):
        self.data = np.ascontiguousarray(X, dtype=np.float32).copy()
        if self.metric == 'cosine':
            self.data = np.ascontiguousarray(_unit(self.data))
        d = self.data.shape[1]
        self.ip = self.metric != 'euclidean'
        quantizer = faiss.IndexFlatIP(d) if self.ip else faiss.IndexFlatL2(d)
        faiss.omp_set_num_threads(1)
        self.index = faiss.IndexIVFFlat(
            quantizer, d, self.n_list,
            faiss.METRIC_INNER_PRODUCT if self.ip else faiss.METRIC_L2,
        )
        self.index.train(self.data)
        self.index.add(self.data)
        self.index.nprobe = self.n_probe

    def set_query_arguments(self, n_probe, steps, blend):
        self.n_probe = int(n_probe)
        self.steps = int(steps)
        self.blend = float(blend)
        if self.steps not in (0, 2) or not 0 <= self.blend <= 1:
            raise ValueError('Invalid HingeZero settings')
        if hasattr(self, 'index'):
            self.index.nprobe = self.n_probe

    def query(self, vector, n):
        q = np.ascontiguousarray(np.asarray(vector, np.float32))
        if self.metric == 'cosine':
            q = _unit(q)
        pool = max(n, n * self.candidate_multiplier, self.field_size)
        _, candidate_ids = self.index.search(q[None, :], pool)
        ids = candidate_ids[0]
        ids = ids[ids >= 0]
        if len(ids) == 0:
            return np.empty(0, dtype=np.int64)
        C = self.data[ids]
        refined = q
        if self.steps:
            field = _unit(C[:self.field_size])
            x = _unit(q)
            for _ in range(self.steps):
                h = field @ x
                response = np.tanh(h) + self.alpha * np.tanh(2.0 * h)
                drive = (response @ field) / (np.abs(response).sum() + 1e-12)
                x = _unit((1 - self.lam) * x + self.eps * drive)
            refined = x * np.linalg.norm(q)
        effective = (1 - self.blend) * q + self.blend * refined
        if self.ip:
            scores = C @ effective
        else:
            diff = C - effective
            scores = -np.einsum('ij,ij->i', diff, diff)
        order = np.lexsort((ids, -scores))[:n]
        return ids[order].astype(np.int64, copy=False)

    def __str__(self):
        return (f'HingeZeroIVF(metric={self.metric}, n_list={self.n_list}, '
                f'candidate_multiplier={self.candidate_multiplier}, '
                f'n_probe={self.n_probe}, steps={self.steps}, blend={self.blend}, '
                f'alpha={self.alpha}, eps={self.eps}, lam={self.lam}, '
                f'field_size={self.field_size})')
