"""Exact pair-bootstrap Spearman via multiplicity weights.

A bootstrap resample of n pairs is equivalent to integer weights W[b, i] = number of times pair i
was drawn. Spearman on the resampled data (average ranks for ties, as scipy) equals the
W-weighted Pearson correlation between W-weighted average ranks on the original pairs, so no
resampled arrays are materialised. Using one W for every method gives the paired bootstrap.
A cluster bootstrap (resampling groups of pairs that share sentences) is just a different W.
"""

import numpy as np


def bootstrap_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    W = np.zeros((n_boot, n), dtype=np.float64)
    np.add.at(W, (np.arange(n_boot)[:, None], idx), 1.0)
    return W


def resample_counts(n_samples: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_samples) multiplicity counts of a multinomial resample, for bootstrapping a per-sample *mean*
    (nDCG@10 per query, correctness per test item) in the retrieval/classification suite.

    Same distribution as `bootstrap_weights` (draw n indices with replacement), but a different random stream,
    so the two cannot be swapped without changing the CIs already reported: the STS pipeline keeps
    `bootstrap_weights` (exact weighted-rank Spearman), the suite keeps this."""
    return np.random.default_rng(seed).multinomial(n_samples, np.full(n_samples, 1.0 / n_samples), size=n_boot)


def bootstrap_means(per_sample: np.ndarray, n_boot: int, seed: int, chunk: int = 250) -> np.ndarray:
    """(n_boot, n_methods) bootstrap means of per_sample (n_methods, N): `resample_counts(N, n_boot, seed) @ per_sample.T / N`,
    drawn in blocks of `chunk` resamples from one generator (identical values; bounds memory for the full classification
    test splits, N up to 400,000)."""
    N = per_sample.shape[1]
    rng = np.random.default_rng(seed)
    p = np.full(N, 1.0 / N)
    return np.concatenate([(rng.multinomial(N, p, size=min(chunk, n_boot - s)) @ per_sample.T) / N
                           for s in range(0, n_boot, chunk)])


def pair_components(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Connected component (0..C-1) of each pair in the graph with sentences as nodes and pairs as
    edges; u, v are the pairs' sentence ids. Pairs in different components share no sentence."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    u, v = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
    n = int(max(u.max(), v.max())) + 1
    _, lab = connected_components(coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)), directed=False)
    return np.unique(lab[u], return_inverse=True)[1]


def cluster_bootstrap_weights(comp: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Cluster bootstrap as multiplicity weights: draw the C clusters with replacement; a pair's weight
    is the number of times its cluster was drawn. With one pair per cluster (comp = arange(n)) this
    is exactly `bootstrap_weights(n, n_boot, seed)`."""
    comp = np.asarray(comp)
    return bootstrap_weights(int(comp.max()) + 1, n_boot, seed)[:, comp]


def weighted_avg_ranks(values: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(B, n) average ranks of `values` within each weighted resample."""
    order = np.argsort(values, kind="stable")
    sv = values[order]
    new_group = np.r_[True, sv[1:] != sv[:-1]]
    starts = np.flatnonzero(new_group)
    gid = np.cumsum(new_group) - 1
    G = np.add.reduceat(W[:, order], starts, axis=1)          # (B, n_groups) weight per tie group
    group_rank = np.cumsum(G, axis=1) - G + (G + 1) / 2
    ranks = np.empty_like(W)
    ranks[:, order] = group_rank[:, gid]
    return ranks


def weighted_pearson(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    sw = W.sum(1, keepdims=True)
    xm = x - (W * x).sum(1, keepdims=True) / sw
    ym = y - (W * y).sum(1, keepdims=True) / sw
    return (W * xm * ym).sum(1) / np.sqrt((W * xm * xm).sum(1) * (W * ym * ym).sum(1))


def bootstrap_spearman(pred: np.ndarray, gold: np.ndarray, W: np.ndarray) -> np.ndarray:
    """pred (n, m) -> (B, m) bootstrap Spearman values, all columns under the same weights W."""
    pred = pred.reshape(len(gold), -1)
    gr = weighted_avg_ranks(gold, W)
    return np.stack([weighted_pearson(weighted_avg_ranks(pred[:, j], W), gr, W) for j in range(pred.shape[1])], axis=1)


def bootstrap_spearman_torch(pred: np.ndarray, gold: np.ndarray, W: np.ndarray, device: str = "cuda",
                             chunk: int = 16) -> np.ndarray:
    """Same result as `bootstrap_spearman` (float64 throughout), batched over columns on `device`.

    Per column, in sorted order with tie groups [s, e]: rank = C[s-1] + (C[e] - C[s-1] + 1) / 2,
    where C is the cumulative resample weight, i.e. the weighted average rank of `weighted_avg_ranks`.
    The weighted Pearson is order-invariant, so it is computed in each column's sorted order.
    """
    import torch

    pred = pred.reshape(len(gold), -1)
    n, m = pred.shape
    Wt = torch.as_tensor(W, dtype=torch.float64, device=device)                       # (B, n)
    gr = torch.as_tensor(weighted_avg_ranks(gold, W), dtype=torch.float64, device=device)
    B = Wt.shape[0]
    sw = Wt.sum(1)[:, None]                                                           # (B, 1)
    ar = torch.arange(n, device=device)
    out = []
    for j0 in range(0, m, chunk):
        P = torch.as_tensor(np.ascontiguousarray(pred[:, j0:j0 + chunk].T), dtype=torch.float64, device=device)
        c = P.shape[0]
        sv, order = torch.sort(P, dim=1, stable=True)                                  # (c, n)
        new = torch.ones_like(sv, dtype=torch.bool)
        new[:, 1:] = sv[:, 1:] != sv[:, :-1]
        last = torch.ones_like(new)
        last[:, :-1] = new[:, 1:]
        start = torch.cummax(torch.where(new, ar, 0), dim=1).values
        end = torch.flip(torch.cummin(torch.flip(torch.where(last, ar, n - 1), [1]), dim=1).values, [1])
        ws = Wt[:, order]                                                             # (B, c, n)
        cpad = torch.cat([torch.zeros(B, c, 1, dtype=torch.float64, device=device), torch.cumsum(ws, 2)], 2)
        before = torch.gather(cpad, 2, start.expand(B, c, n))
        upto = torch.gather(cpad, 2, (end + 1).expand(B, c, n))
        x = before + (upto - before + 1) / 2
        del cpad, before, upto
        y = gr[:, order]                                                              # (B, c, n)
        x = x - (ws * x).sum(2, keepdim=True) / sw[:, :, None]
        y = y - (ws * y).sum(2, keepdim=True) / sw[:, :, None]
        r = (ws * x * y).sum(2) / torch.sqrt((ws * x * x).sum(2) * (ws * y * y).sum(2))
        out.append(r.cpu().numpy())
        del ws, x, y
    return np.concatenate(out, axis=1)
