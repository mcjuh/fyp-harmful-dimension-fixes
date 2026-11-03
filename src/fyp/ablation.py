"""Closed-form coordinate deletion for cosine-similarity STS evaluation.

For one task (n pairs, embeddings a, b in float64), deleting coordinate set S gives
    cos_S = (dot - sum_{d in S} a_d b_d) / sqrt((|a|^2 - sum_S a_d^2) (|b|^2 - sum_S b_d^2)),
so single-dimension deltas and nested removal curves only need per-pair, per-coordinate products.
Spearman uses average ranks for ties (matches scipy.stats.spearmanr).
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


def _centered_unit(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    return x / np.linalg.norm(x, axis=0)


def spearman_columns(pred: np.ndarray, target_rank_unit: np.ndarray) -> np.ndarray:
    """Spearman between each column of pred (n, m) and a target whose average ranks have been
    centred and scaled to unit norm (`rank_unit(target)`). Returns shape (m,)."""
    r = rankdata(pred, axis=0)
    return target_rank_unit @ _centered_unit(r)


def rank_unit(target: np.ndarray) -> np.ndarray:
    return _centered_unit(rankdata(target)[:, None])[:, 0]


@dataclass
class TaskData:
    """Per-task precomputed quantities (all float64)."""
    name: str
    P: np.ndarray       # (n, D) a_d * b_d
    A2: np.ndarray      # (n, D) a_d^2
    B2: np.ndarray      # (n, D) b_d^2
    dot: np.ndarray     # (n,)
    na: np.ndarray      # (n,) |a|^2
    nb: np.ndarray      # (n,) |b|^2
    gold: np.ndarray    # (n,)
    gold_ru: np.ndarray  # rank_unit(gold)

    @classmethod
    def from_pairs(cls, name: str, a: np.ndarray, b: np.ndarray, gold: np.ndarray) -> "TaskData":
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        P, A2, B2 = a * b, a * a, b * b
        return cls(name, P, A2, B2, P.sum(1), A2.sum(1), B2.sum(1),
                   np.asarray(gold, dtype=np.float64), rank_unit(gold))

    @property
    def n(self) -> int:
        return len(self.gold)

    def cos_full(self) -> np.ndarray:
        return self.dot / np.sqrt(self.na * self.nb)

    def rho_full(self) -> float:
        return float(spearman_columns(self.cos_full()[:, None], self.gold_ru)[0])

    def cos_single_deletions(self) -> np.ndarray:
        """(n, D): cosine with each single coordinate deleted."""
        return (self.dot[:, None] - self.P) / np.sqrt((self.na[:, None] - self.A2) * (self.nb[:, None] - self.B2))

    def single_deltas(self) -> np.ndarray:
        """Delta_t(d) = rho({d}) - rho(empty) for every coordinate d, shape (D,)."""
        rho = spearman_columns(self.cos_single_deletions(), self.gold_ru)
        return rho - self.rho_full()

    def cos_nested(self, order: Sequence[int], ns: Iterable[int]) -> np.ndarray:
        """(n, len(ns)): cosine after deleting the first n coordinates of `order`, for each n in ns."""
        order = np.asarray(order, dtype=np.int64)
        ns = np.asarray(list(ns), dtype=np.int64)
        assert ns.max(initial=0) <= len(order)
        z = np.zeros((self.n, 1))
        cP = np.concatenate([z, np.cumsum(self.P[:, order], axis=1)], axis=1)[:, ns]
        cA = np.concatenate([z, np.cumsum(self.A2[:, order], axis=1)], axis=1)[:, ns]
        cB = np.concatenate([z, np.cumsum(self.B2[:, order], axis=1)], axis=1)[:, ns]
        return (self.dot[:, None] - cP) / np.sqrt((self.na[:, None] - cA) * (self.nb[:, None] - cB))

    def rho_nested(self, order: Sequence[int], ns: Iterable[int]) -> np.ndarray:
        return spearman_columns(self.cos_nested(order, ns), self.gold_ru)

    def fidelity_nested(self, order: Sequence[int], ns: Iterable[int]) -> np.ndarray:
        """Label-free: Spearman(full cosine, truncated cosine) (label-free fidelity)."""
        return spearman_columns(self.cos_nested(order, ns), rank_unit(self.cos_full()))

    def subset(self, mask: np.ndarray, name: Optional[str] = None) -> "TaskData":
        return TaskData(name or self.name, self.P[mask], self.A2[mask], self.B2[mask], self.dot[mask],
                        self.na[mask], self.nb[mask], self.gold[mask], rank_unit(self.gold[mask]))


def brute_force_rho(a: np.ndarray, b: np.ndarray, gold: np.ndarray, deleted) -> float:
    """Independent reference for the closed form: physically drop the columns, then
    cos = dot / sqrt(|a|^2 |b|^2) and scipy's Spearman. (Not dot / (|a| |b|): that gives identical-sentence
    pairs 1 +- 1 ulp instead of an exact tie at 1.0, which moves single-dim deltas by up to ~3.5e-5 on STS12.)"""
    keep = np.setdiff1d(np.arange(a.shape[1]), np.asarray(list(deleted), dtype=int))
    aa, bb = a[:, keep], b[:, keep]
    cos = (aa * bb).sum(1) / np.sqrt((aa * aa).sum(1) * (bb * bb).sum(1))
    return float(spearmanr(cos, gold).statistic)


def haar_orthogonal(dim: int, seed: int) -> np.ndarray:
    """Haar-random orthogonal matrix: QR of a standard Gaussian, with the signs of R's diagonal
    folded into Q so the distribution is uniform (Mezzadri 2007). rotation null random-rotation null."""
    Q, R = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return Q * np.sign(np.diag(R))


def ranked_order(score: np.ndarray, descending: bool, tie_seed: int) -> np.ndarray:
    """Order by score (descending = best-first); exact ties broken by a seeded random
    permutation, never by index."""
    tiebreak = np.random.default_rng(tie_seed).permutation(len(score))
    primary = -score if descending else score
    return np.lexsort((tiebreak, primary))
