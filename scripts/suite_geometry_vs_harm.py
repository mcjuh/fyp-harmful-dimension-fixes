"""Does a dimension's geometry predict whether deleting it helps?  Retrieval and classification suites, final layer, float64.

One fold per dataset t. Harm is the held-out (held-out) score of fold t: Delta(d) averaged over the OTHER datasets, so it never
uses dataset t's labels or relevance judgements. Geometry is measured on the unlabelled embeddings of dataset t itself
(retrieval: the corpus; classification: all rows of the task), capped at 20,000 random rows (seed 11001). Properties are those of
scripts/sts_geometry_vs_harm.py, with the cosine contribution taken over random pairs of the dataset (there are no sentence pairs).
Per fold: Spearman(Delta, property) over the 768 dimensions; then mean, standard deviation and range over the folds of a model.
Datasets with fewer rows than dimensions (amazon_counterfactual, 491) have rank-deficient geometry: read their folds with care.

Usage: python scripts/suite_geometry_vs_harm.py --suite ret|cls [--models ...] [--truncation-dir DIR] [--out DIR]
       Matryoshka pair: --models bert-mrl bert-no-mrl --truncation-dir results/matryoshka/<retrieval|classification> --out results/matryoshka/<retrieval|classification>
Outputs: geometry_vs_harm_by_fold.csv, geometry_vs_harm_by_fold_geometry.csv, geometry_vs_harm_summary.csv (+ .meta.json).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fyp.geometry import N_RANDOM_PAIRS, N_TOP, SEED_PAIRS, geometry
from fyp.models import cache_dir
from fyp.paths import CLS_CACHE, RET_CACHE, SUITE, SUITE_DIR
from fyp.runtime import run_metadata, save_table, setup_device
from fyp.suite import _layer_slice

SEED_ROWS = 11001
MAX_ROWS = 20000


def dataset_embeddings(suite, model, key):
    path = cache_dir(RET_CACHE, model) / f"{key}.corpus.npy" if suite == "ret" else cache_dir(CLS_CACHE, model) / f"{key}.emb.npy"
    X = _layer_slice(path, None, model)
    if len(X) > MAX_ROWS:
        X = X[np.sort(np.random.default_rng(SEED_ROWS).choice(len(X), MAX_ROWS, replace=False))]
    return np.asarray(X, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["ret", "cls"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--truncation-dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    base = SUITE / SUITE_DIR[args.suite]
    tdir, out = Path(args.truncation_dir or base), Path(args.out or base)
    dev = setup_device()
    scores = pd.read_csv(tdir / "leave_one_out_scores.csv.gz")
    scores = scores[scores["protocol"] == "held_out"]
    models = args.models or list(scores["model"].unique())

    rows, geo = [], []
    for m in models:
        for fold in scores[scores["model"] == m]["fold"].unique():
            delta = scores[(scores["model"] == m) & (scores["fold"] == fold)].sort_values("dim")["score"].to_numpy()
            props, summary = geometry(dataset_embeddings(args.suite, m, fold))
            rows += [{"model": m, "fold": fold, "property": k, "spearman_with_delta": float(spearmanr(delta, v)[0])}
                     for k, v in props.items()]
            geo.append({"model": m, "fold": fold, **summary, "frac_degrading": float((delta > 0).mean())})
        print(m, "done")

    by_fold = pd.DataFrame(rows)
    g = by_fold.groupby(["model", "property"])["spearman_with_delta"]
    summ = pd.DataFrame({"mean": g.mean(), "sd": g.std(), "min": g.min(), "max": g.max(),
                         "n_positive": g.apply(lambda x: int((x > 0).sum())), "n_folds": g.count()}).reset_index()
    meta = run_metadata(dev, script="scripts/suite_geometry_vs_harm.py", protocol_id=SUITE_DIR[args.suite],
                        seeds={"random_pairs": SEED_PAIRS, "row_subsample": SEED_ROWS},
                        extra={"suite": args.suite, "n_random_pairs": N_RANDOM_PAIRS, "n_top": N_TOP, "max_rows": MAX_ROWS})
    out.mkdir(parents=True, exist_ok=True)
    save_table(by_fold, out / "geometry_vs_harm_by_fold.csv", meta)
    save_table(pd.DataFrame(geo), out / "geometry_vs_harm_by_fold_geometry.csv", meta)
    save_table(summ, out / "geometry_vs_harm_summary.csv", meta)
    print(summ.pivot(index="model", columns="property", values="mean").round(3).to_string())
    print(summ.pivot(index="model", columns="property", values="n_positive").to_string())


if __name__ == "__main__":
    main()
