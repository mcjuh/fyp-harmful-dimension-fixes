"""One interface over the two suites (retrieval nDCG@10, classification probe accuracy) so the paper's analyses
(truncation, attribution, ranked removal, PCA, ...) are written once.

An *evaluator* represents one task (a retrieval dataset's queries, or a classification task's eval split) of one
model at one layer, and exposes, for a (B, 768) boolean KEEP-mask array:
    metric(keep, exps=None)      -> (B,)    nDCG@10, or probe accuracy
    per_sample(keep, exps=None)  -> (B, N)  per-query nDCG@10, or per-item correctness (for the bootstrap)
    fidelity(keep, exps=None)    -> (B,)    label-free: Spearman of full vs truncated document scores (retrieval),
                                            agreement of predictions with the full-embedding probe (classification)
    base() -> float,  single_deltas() -> (768,)
`exps` only matters for classification (10 few-shot experiments): None = average over all experiments; an int array of
length B pairs mask i with experiment exps[i] (used for the random-K masks).
"""

from typing import Dict, Optional

import numpy as np
import torch

from fyp.classification import ALL_CLS_KEYS, load_table
from fyp.models import cache_dir, final_layer
from fyp.paths import CLS_CACHE, RET_CACHE
from fyp.probe_gpu import HISTORY, LogRegGPU, ProbeState
from fyp.protocol import N_DIMS
from fyp.retrieval import RET_KEYS, RetrievalEval, load_ret_table, make_eval

SWEEP_TOL = 1e-4          # gradient tolerance of the sweep probes (0 prediction flips vs 1e-6 on banking77 dev)
BASE_TOL = 1e-6


class RetTask(RetrievalEval):
    kind = "ret"

    def metric(self, keep, exps=None):
        return super().metric(keep)

    def per_sample(self, keep, exps=None):
        return self.per_query(keep)

    def fidelity(self, keep, exps=None):
        return super().fidelity(keep)

    def both(self, keep, exps=None):
        return self.metric(keep), self.fidelity(keep)


class ClsTask:
    kind = "cls"

    def __init__(self, Xtr, ytr, Xev, yev, device: str = "cuda", tol: float = SWEEP_TOL):
        self.device, self.tol = device, tol
        self.E = len(Xtr)
        self.probes = [LogRegGPU(Xtr[e], ytr[e], device) for e in range(self.E)]
        self.Xev = torch.as_tensor(np.asarray(Xev), dtype=torch.float64, device=device)
        self.yev = torch.as_tensor(np.asarray(yev), device=device)
        self.m = len(yev)
        self.base_state = [p.fit(None, tol=BASE_TOL) for p in self.probes]
        self.base_pred = [p.predict(st, self.Xev)[0] for p, st in zip(self.probes, self.base_state)]
        self.base_correct = np.array([(bp == self.yev).double().mean().item() for bp in self.base_pred])
        self.n_not_converged = 0
        self.n_fits = 0

    def _chunk(self, e: int) -> int:
        return max(4, min(1024, int(3e9 / (8 * self.probes[e].P * (2 * HISTORY + 10)))))

    def _predict(self, p, st: ProbeState) -> torch.Tensor:
        """(b, m) predicted labels, computed in blocks of masks so the (block, m, classes) logits stay near 1.6 GB
        (the full test splits reach m = 400,000)."""
        sub = max(1, int(2e8 // (self.m * p.Kp)))
        out = [p.predict(ProbeState(st.W[s:s + sub], st.b[s:s + sub], st.n_iter[s:s + sub], st.converged[s:s + sub], st.classes), self.Xev)
               for s in range(0, st.W.shape[0], sub)]
        return torch.cat(out, 0)

    def _run(self, keep: np.ndarray, e: int) -> torch.Tensor:
        """(B, m) predicted labels for B keep masks, probe of experiment e warm-started from its full-embedding fit."""
        p, out = self.probes[e], []
        step = max(4, min(self._chunk(e), int(1e9 // (2 * self.m))))      # keeps the (step, m) prediction block under ~2 GB
        for s in range(0, len(keep), step):
            st = p.fit(keep[s:s + step], warm=self.base_state[e], tol=self.tol)
            self.n_not_converged += int((~st.converged).sum())
            self.n_fits += len(st.converged)
            out.append(self._predict(p, st))
        return torch.cat(out, 0)

    def _reduce(self, keep, exps, fn):
        """Apply fn(pred (b, m), experiment) -> (b, ...) tensor, averaged over experiments (exps None) or paired."""
        keep = np.asarray(keep, dtype=bool)
        if exps is None:
            acc = None
            for e in range(self.E):
                r = fn(self._run(keep, e), e).double()
                acc = r if acc is None else acc + r
            return (acc / self.E).cpu().numpy()
        exps = np.asarray(exps)
        res = None
        for e in np.unique(exps):
            idx = np.flatnonzero(exps == e)
            r = fn(self._run(keep[idx], int(e)), int(e)).double().cpu().numpy()
            if res is None:
                res = np.zeros((len(keep),) + r.shape[1:])
            res[idx] = r
        return res

    def metric(self, keep, exps=None):
        return self._reduce(keep, exps, lambda pred, e: (pred == self.yev[None]).sum(1).double() / self.m)

    def per_sample(self, keep, exps=None):
        return self._reduce(keep, exps, lambda pred, e: (pred == self.yev[None]).double())

    def fidelity(self, keep, exps=None):
        return self._reduce(keep, exps, lambda pred, e: (pred == self.base_pred[e][None]).sum(1).double() / self.m)

    def both(self, keep, exps=None):
        """(metric, fidelity) from one set of probe fits."""
        r = self._reduce(keep, exps, lambda pred, e: torch.stack(
            [(pred == self.yev[None]).sum(1).double() / self.m, (pred == self.base_pred[e][None]).sum(1).double() / self.m], 1))
        return r[:, 0], r[:, 1]

    def base(self) -> float:
        return float(self.base_correct.mean())

    def single_deltas(self) -> np.ndarray:
        return self.metric(~np.eye(N_DIMS, dtype=bool)) - self.base()


# ------------------------------------------------------------------ loading
def _layer_slice(path, layer: Optional[int], model: str):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:          # final-layer cache (CLS_CACHE)
        assert layer in (None, final_layer(model)), f"{path} holds the final layer only"
        return np.ascontiguousarray(arr, dtype=np.float32)
    return np.ascontiguousarray(arr[:, final_layer(model) if layer is None else layer, :], dtype=np.float32)


def load_ret_tasks(model: str, split: str, layer: Optional[int] = None, device: str = "cuda",
                   datasets=None, transform=None) -> Dict[str, RetTask]:
    """transform: optional (n,768)->(n,d') map applied to queries and documents (centring, PCA rotation)."""
    out = {}
    for key in datasets or RET_KEYS:
        tab = load_ret_table(RET_CACHE, key)
        q = _layer_slice(cache_dir(RET_CACHE, model) / f"{key}.queries.npy", layer, model)
        d = _layer_slice(cache_dir(RET_CACHE, model) / f"{key}.corpus.npy", layer, model)
        if transform is not None:
            q, d = transform(q), transform(d)
        ev = make_eval(tab, q, d, split, device)
        ev.__class__ = RetTask
        out[key] = ev
    return out


def load_cls_tasks(model: str, split: str, layer: Optional[int] = None, device: str = "cuda",
                   tasks=None, tol: float = SWEEP_TOL, transform=None, cache=None) -> Dict[str, ClsTask]:
    """cache: CLS_CACHE (default; final layer, full test splits) or CLS_ALL_LAYERS_CACHE (every layer, test capped; for layer-wise work)."""
    cache = cache or CLS_CACHE
    out = {}
    for key in tasks or ALL_CLS_KEYS:
        tab = load_table(cache, key)
        emb = _layer_slice(cache_dir(cache, model) / f"{key}.emb.npy", layer, model)
        if transform is not None:
            emb = transform(emb)
        ev_rows, y_ev = tab[f"{split}_rows"], tab[f"{split}_y"]
        Xtr = np.stack([emb[tab["train_rows"][e]] for e in range(tab["train_rows"].shape[0])])
        out[key] = ClsTask(Xtr, tab["train_y"], emb[ev_rows], y_ev, device, tol)
    return out


def load_tasks(suite: str, model: str, split: str, **kw):
    return (load_ret_tasks if suite == "ret" else load_cls_tasks)(model, split, **kw)
