"""Does a dimension's geometry predict whether deleting it helps?  STS, final layer, float64.

Harm is the train_ranked score: Delta(d) = rho(without d) - rho(full) on the filtered STS-B train pairs (R),
so no evaluation data is used. Per-dimension geometry is measured on the same R sentences:
  mean_abs      |mean of coordinate d|
  mean_z_abs    |z| of the mean embedding (the outlier criterion)
  variance      variance of coordinate d
  kurtosis      excess kurtosis of coordinate d
  cos_contrib   Timkey & van Schijndel cosine contribution: mean over R pairs of a_d b_d / sqrt(|a|^2 |b|^2)
  pc_energy     share of coordinate d in the top-7 principal directions (sum of squared loadings)
Per-dimension table: Spearman(Delta, property) within each model.
Per-model table: mean cosine of random sentence pairs (Ethayarajh), effective rank (Roy & Vetterli), IsoScore (Rudman et al.), the share of the
mean cosine held by the single largest dimension, and the fraction of degrading dimensions (Delta > 0).

Usage: python scripts/sts_geometry_vs_harm.py [--models ...] [--out DIR] [--truncation-dir DIR]
       Matryoshka pair: --models bert-mrl bert-no-mrl --truncation-dir results/matryoshka/sts --out results/matryoshka/sts
Outputs: geometry_vs_harm_by_dimension.csv, geometry_vs_harm_by_model.csv (+ .meta.json) in --out.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, spearmanr

from fyp.corpora import R_KEYS
from fyp.evaluation import load_embeddings, load_pairs_index
from fyp.models import MODEL_SPECS
from fyp.paths import OVERLAP_MASKS, STS_DIAGNOSTICS, STS_TRUNCATION
from fyp.runtime import run_metadata, save_table, setup_device

SEED_PAIRS = 11000
N_RANDOM_PAIRS = 20000
N_TOP = 7        # same number of directions as all-but-the-top (N_DIMS // 100)
PROPS = ["mean_abs", "mean_z_abs", "variance", "kurtosis", "cos_contrib", "pc_energy"]


def cosine_rows(a, b):
    return np.einsum("nd,nd->n", a, b) / np.sqrt(np.einsum("nd,nd->n", a, a) * np.einsum("nd,nd->n", b, b))


def isoscore(eigvals):
    """IsoScore (Rudman et al., 2022) from the variances along the principal axes: 1 = uniform use of all n axes."""
    n = len(eigvals)
    sigma_hat = np.sqrt(n) * eigvals / np.linalg.norm(eigvals)
    defect = np.linalg.norm(sigma_hat - 1) / np.sqrt(2 * (n - np.sqrt(n)))
    return float((n - defect ** 2 * (n - np.sqrt(n)) - 1) / (n - 1))


def analyse(model, delta):
    rk = R_KEYS[0]
    layer = MODEL_SPECS[model].final_layer
    idx = load_pairs_index(rk)
    mask = np.load(OVERLAP_MASKS)[f"R__{rk}"]
    i1, i2 = idx["idx1"][mask], idx["idx2"][mask]
    rows = np.unique(np.concatenate([i1, i2]))
    X = np.asarray(load_embeddings(model, rk)[rows, layer, :], dtype=np.float64)
    pos = {r: k for k, r in enumerate(rows)}
    a, b = X[[pos[i] for i in i1]], X[[pos[i] for i in i2]]

    mu = X.mean(0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    p = S / S.sum()
    iso = isoscore(S ** 2 / (len(X) - 1))
    erank = float(np.exp(-(p * np.log(p)).sum()))

    cc = (a * b / np.sqrt(np.einsum("nd,nd->n", a, a) * np.einsum("nd,nd->n", b, b))[:, None]).mean(0)
    rng = np.random.default_rng(SEED_PAIRS)
    j1, j2 = rng.integers(0, len(X), N_RANDOM_PAIRS), rng.integers(0, len(X), N_RANDOM_PAIRS)
    keep = j1 != j2
    rand_cos = cosine_rows(X[j1[keep]], X[j2[keep]]).mean()
    rand_cc = (X[j1[keep]] * X[j2[keep]] / np.sqrt(np.einsum("nd,nd->n", X[j1[keep]], X[j1[keep]])
                                                  * np.einsum("nd,nd->n", X[j2[keep]], X[j2[keep]]))[:, None]).mean(0)

    props = {"mean_abs": np.abs(mu), "mean_z_abs": np.abs((mu - mu.mean()) / mu.std()), "variance": X.var(0),
             "kurtosis": kurtosis(X, axis=0), "cos_contrib": cc, "pc_energy": (Vt[:N_TOP] ** 2).sum(0)}
    dim_rows = [{"model": model, "property": k, "spearman_with_delta": float(spearmanr(delta, v)[0])}
                for k, v in props.items()]
    model_row = {"model": model, "n_sentences": len(X), "mean_random_cosine": float(rand_cos), "effective_rank": erank, "isoscore": iso,
                 "top_dim_cosine_share": float(rand_cc.max() / rand_cc.sum()), "frac_degrading": float((delta > 0).mean())}
    return dim_rows, model_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--out", default=str(STS_DIAGNOSTICS))
    ap.add_argument("--truncation-dir", default=str(STS_TRUNCATION), help="folder holding leave_one_out_scores.csv")
    args = ap.parse_args()
    dev = setup_device()
    scores = pd.read_csv(Path(args.truncation_dir) / "leave_one_out_scores.csv")
    scores = scores[scores["protocol"] == "train_ranked"]
    models = args.models or list(scores["model"].unique())

    dim_rows, model_rows = [], []
    for m in models:
        delta = scores[scores["model"] == m].sort_values("dim")["score"].to_numpy()
        d, r = analyse(m, delta)
        dim_rows += d
        model_rows.append(r)
        print(m, {x["property"]: round(x["spearman_with_delta"], 3) for x in d})

    meta = run_metadata(dev, script="scripts/sts_geometry_vs_harm.py", protocol_id="sts",
                        seeds={"random_pairs": SEED_PAIRS}, extra={"n_random_pairs": N_RANDOM_PAIRS, "n_top": N_TOP})
    out = pd.DataFrame(dim_rows)
    mt = pd.DataFrame(model_rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_table(out, Path(args.out) / "geometry_vs_harm_by_dimension.csv", meta)
    save_table(mt, Path(args.out) / "geometry_vs_harm_by_model.csv", meta)
    print(out.pivot(index="model", columns="property", values="spearman_with_delta").round(3).to_string())
    print(mt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
