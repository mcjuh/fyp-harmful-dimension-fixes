"""Geometry of a set of embeddings, and per-dimension properties, for relating geometry to dimension harm.

`geometry(X, a, b)` returns (props, summary): `props` is {name: (768,) array} of per-coordinate properties and `summary` the
model-level quantities (mean random-pair cosine, effective rank, IsoScore, cosine share of the largest dimension).
`a, b` are the sentence pairs used for the cosine contribution; with None, random pairs of X are used.
"""

import numpy as np
from scipy.stats import kurtosis

SEED_PAIRS = 11000
N_RANDOM_PAIRS = 20000
N_TOP = 7        # same number of directions as all-but-the-top (N_DIMS // 100)
PROPS = ["mean_abs", "mean_z_abs", "variance", "kurtosis", "cos_contrib", "pc_energy"]


def _norm2(x):
    return np.einsum("nd,nd->n", x, x)


def isoscore(eigvals):
    """IsoScore (Rudman et al., 2022) from the variances along the principal axes: 1 = uniform use of all n axes."""
    n = len(eigvals)
    sigma_hat = np.sqrt(n) * eigvals / np.linalg.norm(eigvals)
    defect = np.linalg.norm(sigma_hat - 1) / np.sqrt(2 * (n - np.sqrt(n)))
    return float((n - defect ** 2 * (n - np.sqrt(n)) - 1) / (n - 1))


def cosine_contribution(a, b):
    """Timkey & van Schijndel: mean over pairs of a_d b_d / sqrt(|a|^2 |b|^2), one value per dimension."""
    return (a * b / np.sqrt(_norm2(a) * _norm2(b))[:, None]).mean(0)


def geometry(X, a=None, b=None):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    p = S[S > 0] / S.sum()
    eig = np.zeros(X.shape[1])
    eig[:len(S)] = S ** 2 / (len(X) - 1)       # fewer sentences than dimensions: the missing axes have zero variance

    rng = np.random.default_rng(SEED_PAIRS)
    j1, j2 = rng.integers(0, len(X), N_RANDOM_PAIRS), rng.integers(0, len(X), N_RANDOM_PAIRS)
    keep = j1 != j2
    ra, rb = X[j1[keep]], X[j2[keep]]
    rand_cc = cosine_contribution(ra, rb)
    cc = rand_cc if a is None else cosine_contribution(a, b)
    props = {"mean_abs": np.abs(mu), "mean_z_abs": np.abs((mu - mu.mean()) / mu.std()), "variance": X.var(0),
             "kurtosis": kurtosis(X, axis=0), "cos_contrib": cc, "pc_energy": (Vt[:N_TOP] ** 2).sum(0)}
    summary = {"n_sentences": len(X), "mean_random_cosine": float(rand_cc.sum()),
               "effective_rank": float(np.exp(-(p * np.log(p)).sum())), "isoscore": isoscore(eig),
               "top_dim_cosine_share": float(rand_cc.max() / rand_cc.sum())}
    return props, summary
