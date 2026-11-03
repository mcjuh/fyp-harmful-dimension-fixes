"""Post-hoc geometry corrections applied to frozen embeddings before a cut.

Every transform is fitted on the final-layer embeddings of the 20k all-nli sample, never on evaluation data, and returned as a
map (n, 768) -> (n, 768) in float64. The cut (random / last / first K%) is applied afterwards by the usual evaluators.
For whitened and PCA coordinates the variance-descending component order makes the `last` cut "keep the first k components".
"""

import numpy as np

from fyp.protocol import N_DIMS

TRANSFORMS = ["centre", "standardise", "mean_r1", "mean_r2", "abtt", "whiten"]
ABTT_M = N_DIMS // 100        # components removed by all-but-the-top: d/100 (Mu & Viswanath 2018), 7 for d = 768


def _unit_rows(Z: np.ndarray) -> np.ndarray:
    return Z / np.sqrt((Z * Z).sum(1, keepdims=True))


def fit(name: str, X: np.ndarray):
    """Fit transform `name` on X (n, 768) and return f(Z) -> float64 (n, 768)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    if name == "centre":
        return lambda Z: np.asarray(Z, np.float64) - mu
    if name == "standardise":      # Timkey & van Schijndel 2021: z-score every coordinate
        sd = X.std(0)
        assert (sd > 0).all()
        return lambda Z: (np.asarray(Z, np.float64) - mu) / sd
    if name == "mean_r1":          # Ren et al. 2026: subtract the mean, renormalise
        return lambda Z: _unit_rows(np.asarray(Z, np.float64) - mu)
    if name == "mean_r2":          # Ren et al. 2026: project out the mean direction, renormalise
        u = mu / np.sqrt(mu @ mu)
        return lambda Z: _unit_rows((lambda W: W - np.outer(W @ u, u))(np.asarray(Z, np.float64)))
    if name in ("abtt", "whiten"):
        _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)       # rows of Vt: components, variance-descending
        if name == "abtt":         # Mu & Viswanath 2018: centre, project out the top ABTT_M components
            V = Vt[:ABTT_M]
            return lambda Z: (lambda W: W - (W @ V.T) @ V)(np.asarray(Z, np.float64) - mu)
        assert S.min() > 0
        scale = np.sqrt(len(X) - 1) / S                              # Su et al. 2021: unit covariance in the PCA basis
        return lambda Z: ((np.asarray(Z, np.float64) - mu) @ Vt.T) * scale
    raise ValueError(name)
