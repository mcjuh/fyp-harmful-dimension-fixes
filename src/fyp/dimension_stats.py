"""Evaluator-independent pieces of the paper's dimension analyses, written once for the STS pipeline
(closed-form `TaskData`) and the retrieval/classification suite (`fyp.suite` evaluators).

  score_summary          Attribution: counts of degrading, informative, neutral dims, uniformity over blocks of 64, lag-1 sign autocorrelation
  shared_dims_table      Shared dimensions: dims degrading in >= n tasks vs a matched-size random null
  outlier_sets           Outliers: |z| > 3 coordinates of the mean embedding vs random non-outlier sets
  removal_curve_specs    Ranked removal: best-/worst-first through all dims, degrading-only / informative-only (ranked and random orders)
  estimates_table        headline paired-bootstrap rows (mean and 95% percentile CI per quantity)
  headline_estimates     RP at K = 25 / 50 for random mean, last-K and the ranked curves, and their differences
  pca_estimates          PCA-384 vs random-50% estimates
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from fyp.ablation import ranked_order
from fyp.protocol import (N_AUTOCORR_PERM, N_DIMS, N_GRID, N_SHARED_NULL, SEED_AUTOCORR, SEED_SHARED, SEED_TIE,
                          SEEDS_OUTLIER, Z_OUTLIER)


# ------------------------------------------------------------------ attribution
def lag1_sign_autocorr(score) -> float:
    s = np.sign(score)
    s = s - s.mean()
    return float((s[:-1] * s[1:]).sum() / (s * s).sum())


def score_summary(score: np.ndarray, seed: int = SEED_AUTOCORR) -> dict:
    """Counts of D (score > 0, strict), I (< 0) and neutral (== 0); fraction of D per block of 64 coordinates; lag-1
    autocorrelation of sign(score) along the index with a 1,000-permutation null. The count of D is never a finding alone."""
    D, I = score > 0, score < 0
    obs = lag1_sign_autocorr(score)
    rng = np.random.default_rng(seed)
    null = np.array([lag1_sign_autocorr(rng.permutation(score)) for _ in range(N_AUTOCORR_PERM)])
    blocks = D.reshape(12, 64).mean(1)
    return {"n_degrading": int(D.sum()), "n_informative": int(I.sum()), "n_neutral": int((score == 0).sum()), "score_min": float(score.min()),
            "score_max": float(score.max()), "score_std": float(score.std()), "lag1_sign_autocorr": obs,
            "lag1_perm_p_two_sided": float((1 + (np.abs(null) >= abs(obs)).sum()) / (N_AUTOCORR_PERM + 1)),
            **{f"degrading_frac_block{i}": float(v) for i, v in enumerate(blocks)},
            "degrading_frac_block_min": float(blocks.min()), "degrading_frac_block_max": float(blocks.max())}


# ------------------------------------------------------------------ shared dimensions
def _tail(h):
    """counts of dims degrading in >= n tasks, from the histogram over n."""
    return np.cumsum(h[..., ::-1], axis=-1)[..., ::-1]


def shared_dims_table(degrading: np.ndarray, model: str) -> pd.DataFrame:
    """`degrading` is the (T, 768) boolean matrix of D_t = {d : delta_t(d) > 0}. Observed histogram of the number of
    tasks each dim is degrading in, against 1,000 draws of independent uniform random sets of the same sizes |D_t|."""
    T = len(degrading)
    obs = np.bincount(degrading.sum(0), minlength=T + 1)
    rng = np.random.default_rng(SEED_SHARED)
    sizes = degrading.sum(1)
    null = np.empty((N_SHARED_NULL, T + 1))
    for i in range(N_SHARED_NULL):
        c = np.zeros(N_DIMS, dtype=int)
        for sz in sizes:
            c[rng.choice(N_DIMS, size=sz, replace=False)] += 1
        null[i] = np.bincount(c, minlength=T + 1)
    nt = _tail(null)
    return pd.DataFrame({"model": model, "n_tasks": np.arange(T + 1), "observed": obs, "null_mean": null.mean(0),
                         "null_lo": np.percentile(null, 2.5, 0), "null_hi": np.percentile(null, 97.5, 0),
                         "observed_ge": _tail(obs), "null_ge_mean": nt.mean(0), "null_ge_lo": np.percentile(nt, 2.5, 0),
                         "null_ge_hi": np.percentile(nt, 97.5, 0),
                         "p_ge_upper": [(1 + (nt[:, j] >= _tail(obs)[j]).sum()) / (N_SHARED_NULL + 1) for j in range(T + 1)]})


# ------------------------------------------------------------------ outliers
def outlier_sets(mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[str, int], np.ndarray]]:
    """z-score the 768 coordinates of the mean embedding `mu`; outliers are |z| > 3. Returns (z, outlier dims,
    {("outliers", -1): outliers, ("random", s): the same number of random non-outlier dims, s in 8000..8009}).
    With no outliers the dict is empty: nothing is removed and that is the result."""
    z = (mu - mu.mean()) / mu.std()
    outl = np.flatnonzero(np.abs(z) > Z_OUTLIER)
    if len(outl) == 0:
        return z, outl, {}
    non = np.setdiff1d(np.arange(N_DIMS), outl)
    sets = {("outliers", -1): outl}
    for s in SEEDS_OUTLIER:
        sets[("random", s)] = np.random.default_rng(s).choice(non, size=len(outl), replace=False)
    return z, outl, sets


# ------------------------------------------------------------------ ranked removal
def removal_curve_specs(score: np.ndarray, seeds_d: Sequence[int], seeds_i: Sequence[int], tie_seed: int = SEED_TIE):
    """Removal curves for one ranking `score` (positive = removing the dimension helps).

    Returns (curves, best, worst, nD, nI) where `curves[(name, seed)] = (order, ns)` evaluates deleting the first n
    entries of `order` for each n in `ns`; seed is -1 for the ranked curves. Names: best_first and worst_first through all
    768 dims (the K grid plus |D| resp. |I|), degrading_only_ranked / informative_only_ranked (stop at |D| / |I|), and degrading_only_random /
    informative_only_random (random orders within the degrading dimensions / I, one per seed). Ties in the score are broken by a seeded permutation, never by index."""
    nD, nI = int((score > 0).sum()), int((score < 0).sum())
    best, worst = ranked_order(score, True, tie_seed), ranked_order(score, False, tie_seed)
    Didx, Iidx = best[:nD], worst[:nI]
    ns_d = sorted({n for n in N_GRID if n <= nD} | {nD})
    ns_i = sorted({n for n in N_GRID if n <= nI} | {nI})
    curves = {("best_first", -1): (best, sorted(set(N_GRID) | {nD})), ("worst_first", -1): (worst, sorted(set(N_GRID) | {nI})),
              ("degrading_only_ranked", -1): (Didx, ns_d), ("informative_only_ranked", -1): (Iidx, ns_i)}
    for s in seeds_d:
        curves[("degrading_only_random", s)] = (np.random.default_rng(s).permutation(Didx), ns_d)
    for s in seeds_i:
        curves[("informative_only_random", s)] = (np.random.default_rng(s).permutation(Iidx), ns_i)
    return curves, best, worst, nD, nI


# ------------------------------------------------------------------ headline bootstrap
def headline_estimates(rp, K: int, random_names: Sequence[str], ranked_protocols: Sequence[str]) -> Dict[str, np.ndarray]:
    """Bootstrap distributions of RP at K for the random-K mean, last-K, and best-/worst-first under each protocol, plus
    each one's difference from the random mean. `rp(name)` returns the per-replicate RP of a named method; the method
    names are `last_{K}`, `{protocol}_{best|worst}_{K}` and, for each entry of `random_names`, `{name}_{K}`."""
    est = {"random_mean": np.mean([rp(f"{nm}_{K}") for nm in random_names], 0), "last": rp(f"last_{K}"),
           **{f"{p}_{c}": rp(f"{p}_{c}_{K}") for p in ranked_protocols for c in ("best", "worst")}}
    for nm in list(est):
        if nm != "random_mean":
            est[f"{nm}_minus_random_mean"] = est[nm] - est["random_mean"]
    return est


def pca_estimates(full: np.ndarray, cent: np.ndarray, pca: np.ndarray, random50: np.ndarray) -> Dict[str, np.ndarray]:
    """PCA-384 vs random 50% (PCA-384). `full`, `cent`, `pca`, `random50` are per-replicate macro metrics (`random50`
    already divided by `full`): the raw full embedding, the centred full embedding, PCA keeping 384 components,
    and the mean over random-50% truncations."""
    est = {"pca50_RP_raw": pca / full, "pca50_RP_centred": pca / cent, "random50_RP": random50, "centred_full_RP_raw": cent / full}
    est["pca50_RP_raw_minus_random50"] = est["pca50_RP_raw"] - random50
    return est


def estimates_table(est: Dict[str, np.ndarray], before: dict, after: dict = None) -> List[dict]:
    """One row per quantity: bootstrap mean and the 2.5 / 97.5 percentile CI. Columns are `before` (e.g. model, K), then
    quantity, then `after` (e.g. scheme), then boot_mean, ci_lo, ci_hi, which fixes the CSV column order."""
    return [{**before, "quantity": q, **(after or {}), "boot_mean": float(v.mean()), "ci_lo": float(np.percentile(v, 2.5)),
             "ci_hi": float(np.percentile(v, 97.5))} for q, v in est.items()]
