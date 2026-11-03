"""Suite ablation: the paper's analyses the analyses on one suite (retrieval = NanoBEIR, or classification =
12 MTEB tasks) for the six frozen encoders, final layer.

  truncation curves (last / first / random), label-free fidelity, per-dimension attribution
  ranked removal curves (best/worst-first, degrading-/informative-only), shared degrading dims, outlier dims
  PCA (fit on the 20k all-nli sample) vs random 50% + centring-matched control
  CI paired bootstrap at K = 25, 50 (queries / test items)
Rankings: in_sample = in-sample (the paper's design, labelled IN-SAMPLE), held_out = leave-one-dataset/task-out.

Eval lock: `--split test` writes under results/retrieval_classification/ (the final run, done after the protocol was fixed);
`--split dev` is for development and must write to a scratch directory (--out).

Usage:
  python scripts/retrieval_classification_truncation_and_ranked_removal.py --suite ret --split test                       # final
  python scripts/retrieval_classification_truncation_and_ranked_removal.py --suite cls --split dev --out <scratch> --quick --tasks banking77 imdb --models bert-raw
  python scripts/retrieval_classification_truncation_and_ranked_removal.py --suite ret --split test --reuse-parts         # re-summarise from saved per-model parts
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import fyp.suite_analysis as A
from fyp.bootstrap import bootstrap_means
from fyp.classification import ALL_CLS_KEYS, load_table
from fyp.dimension_stats import estimates_table, pca_estimates, score_summary
from fyp.models import MODEL_KEYS, ALL_MODEL_KEYS, cache_dir, final_layer
from fyp.timing import StageTimer
from fyp.paths import CACHE as NLI_CACHE, CLS_CACHE, RET_CACHE, SUITE as FINAL_OUT, SUITE_DIR
from fyp.protocol import N50, N_BOOT, N_GRID, SEED_AUTOCORR, SEED_BOOT, SEED_SHARED, SEED_TIE
from fyp.retrieval import RET_KEYS
from fyp.runtime import add_device_args, run_metadata, save_table, setup_device
from fyp.suite import SWEEP_TOL, load_tasks

PROTOCOL_ID = "retrieval_classification"


def mean_embedding(suite: str, model: str, tasks_keys, layer: int) -> np.ndarray:
    """Mean final-layer embedding for the outlier analysis: retrieval = all query texts of the chosen datasets (the paper's definition);
    classification = the few-shot training texts of the chosen tasks (all 10 draws). Label-free."""
    rows = []
    for key in tasks_keys:
        if suite == "ret":
            rows.append(np.load(cache_dir(RET_CACHE, model) / f"{key}.queries.npy", mmap_mode="r")[:, layer, :])
        else:
            tab = load_table(CLS_CACHE, key)
            emb = np.load(cache_dir(CLS_CACHE, model) / f"{key}.emb.npy", mmap_mode="r")
            rows.append(emb[np.unique(tab["train_rows"])] if emb.ndim == 2 else emb[np.unique(tab["train_rows"]), layer, :])
    return np.concatenate(rows).astype(np.float64).mean(0)


def run_pca(suite, model, split, device, layer, tasks_keys, tasks_raw, quick):
    """PCA (sklearn, full SVD, fit on the 20k all-nli sample's final-layer embeddings, never on E) vs random 50%."""
    X = np.asarray(np.load(cache_dir(NLI_CACHE, model) / "nli20k.emb.npy", mmap_mode="r")[:, layer, :], dtype=np.float64)
    pca = PCA(n_components=768, svd_solver="full").fit(X)
    mean, comps = pca.mean_, pca.components_
    kw = {"datasets": tasks_keys} if suite == "ret" else {"tasks": tasks_keys}
    t_pca = load_tasks(suite, model, split, device=device, transform=lambda Z: (np.asarray(Z, np.float64) - mean) @ comps.T, **kw)
    drop_low = np.arange(768 - 1, -1, -1)              # delete the lowest-variance components first
    rows = []
    for k, t in t_pca.items():
        base_c = t.base()
        ms = t.metric(A.keep_from_order(drop_low, N_GRID))
        rows += [{"model": model, "method": "pca", "seed": -1, "task": k, "n": n, "metric": float(m), "base_centred": base_c}
                 for n, m in zip(N_GRID, ms)]
    if suite == "ret":                                  # centring-matched random control (cosine is not centring-invariant)
        t_c = load_tasks(suite, model, split, device=device, transform=lambda Z: np.asarray(Z, np.float64) - mean, **kw)
        for k, t in t_c.items():
            for s in A.random_seeds("ret"):
                order = np.random.default_rng(s).permutation(768)
                m = t.metric(A.keep_from_order(order, [N50]))[0]
                rows.append({"model": model, "method": "random_centred", "seed": s, "task": k, "n": N50, "metric": float(m),
                             "base_centred": t.base()})
    else:                                               # a probe is centring- and rotation-invariant (intercept, L2): PCA-768 == raw
        for k, t in t_pca.items():
            assert abs(t.base() - tasks_raw[k].base()) < 2e-3, (k, t.base(), tasks_raw[k].base())
    # paired bootstrap: PCA-384 vs random-50% (raw baseline) and vs the centred full embedding
    names = list(t_pca)
    boot = {}
    for ti, k in enumerate(names):
        tr, tp = tasks_raw[k], t_pca[k]
        cols = [tr.per_sample(np.ones((1, 768), bool)), tp.per_sample(np.ones((1, 768), bool)),
                tp.per_sample(A.keep_from_order(drop_low, [N50]))]
        for s in A.random_seeds(tr.kind):
            order = np.random.default_rng(s).permutation(768)
            cols.append(tr.per_sample(A.keep_from_order(order, [N50]), A.paired_experiments(tr, s, 1)))
        ps = np.concatenate(cols, 0)
        boot[k] = bootstrap_means(ps, N_BOOT, SEED_BOOT + ti)
    mb = lambda j: np.mean([boot[k][:, j] for k in names], axis=0)
    full, cent, pc = mb(0), mb(1), mb(2)
    rnd = np.mean([mb(3 + i) for i in range(boot[names[0]].shape[1] - 3)], axis=0) / full
    ci = pd.DataFrame(estimates_table(pca_estimates(full, cent, pc, rnd), {"model": model}))
    return pd.DataFrame(rows), ci, pd.DataFrame({"model": model, "n_components": np.arange(1, 769),
                                                 "cum_explained_variance": np.cumsum(pca.explained_variance_ratio_)})


def run_model(suite, model, split, device, tasks_keys, quick):
    t0 = time.time()
    timer = StageTimer(model=model, suite=suite, split=split)
    layer = final_layer(model)
    kw = {"datasets": tasks_keys} if suite == "ret" else {"tasks": tasks_keys}
    tasks = load_tasks(suite, model, split, device=device, **kw)
    print(f"  {model}: loaded {len(tasks)} tasks ({time.time() - t0:.0f}s)", flush=True)
    timer.mark("load_tasks", "load")
    out = {"baseline": pd.DataFrame([{"model": model, "task": k, "metric_full": t.base(), "n_eval": int(
        t.nq if suite == "ret" else t.m)} for k, t in tasks.items()])}
    out["truncation"], out["fidelity"] = A.run_truncation(tasks, model)
    timer.mark("truncation_curves_T1_T2", "bookkeeping")
    print(f"  {model}: truncation and fidelity done ({time.time() - t0:.0f}s)", flush=True)
    deltas = {k: t.single_deltas() for k, t in tasks.items()}
    timer.mark("attribution_leave_one_out", "one_time")
    print(f"  {model}: attribution deltas done ({time.time() - t0:.0f}s)", flush=True)
    units = A.make_units(deltas)
    out["deltas"] = pd.concat([pd.DataFrame({"model": model, "task": k, "dim": np.arange(768), "delta": v}) for k, v in deltas.items()])
    out["scores"] = pd.concat([pd.DataFrame({"model": model, "protocol": p, "fold": f, "dim": np.arange(768), "score": sc})
                                  for (p, f), (sc, _) in units.items()])
    out["attribution_summary"] = pd.DataFrame([{"model": model, "protocol": p, "fold": f, "n_ranking_tasks": len(tasks) if p == "in_sample" else len(tasks) - 1,
                                       **score_summary(sc)} for (p, f), (sc, _) in units.items()])
    timer.mark("ranking_units_and_tables", "bookkeeping")
    out["ranked_removal"] = A.run_removal_curves(tasks, units, model)
    timer.mark("removal_curves_T4", "bookkeeping")
    print(f"  {model}: ranked removal done ({time.time() - t0:.0f}s)", flush=True)
    out["shared"] = A.run_shared(deltas, model)
    timer.mark("shared_dimensions_T5", "bookkeeping")
    mu = mean_embedding(suite, model, list(tasks), layer)
    out["outliers"], out["outlier_z"] = A.run_outliers(tasks, mu, model)
    timer.mark("outliers_T6", "bookkeeping")
    out["pca"], out["pca_ci"], out["pca_variance"] = run_pca(suite, model, split, device, layer, list(tasks), tasks, quick)
    timer.mark("pca_T7_fit_transform_curves_bootstrap", "bookkeeping")
    out["ci"] = A.run_headline_ci(tasks, units, model)
    timer.mark("headline_bootstrap", "bookkeeping")
    if suite == "cls":
        out["probe_stats"] = pd.DataFrame([{"model": model, "task": k, "n_fits": t.n_fits, "n_not_converged": t.n_not_converged,
                                            "tol": t.tol} for k, t in tasks.items()])
    out["timing"] = timer.table()
    print(f"  {model}: done in {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, choices=["ret", "cls"])
    p.add_argument("--split", choices=("dev", "test"), default="test")
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=ALL_MODEL_KEYS)
    p.add_argument("--tasks", nargs="+", default=None, help="subset of datasets/tasks (development only)")
    p.add_argument("--out", type=Path, default=None, help="output directory (required for --split dev); default results/retrieval_classification/<suite>")
    p.add_argument("--quick", action="store_true", help="development: 10 random-K seeds instead of 100 (retrieval)")
    p.add_argument("--reuse-parts", action="store_true", help="skip models whose part file exists; summarise all parts")
    add_device_args(p)
    args = p.parse_args()
    out_root = (args.out or (FINAL_OUT / SUITE_DIR[args.suite])).resolve()
    if args.split == "dev" and args.out is None:
        raise SystemExit("dev runs must not write to results/retrieval_classification (eval lock): pass --out <scratch dir>")
    if args.split == "dev" and out_root.is_relative_to(FINAL_OUT.resolve()):
        raise SystemExit("dev runs must not write under results/retrieval_classification (eval lock)")
    if args.split == "test" and (args.quick or args.tasks):
        raise SystemExit("the final (test) run uses all tasks and all seeds")
    if args.quick:
        A.RET_RANDOM = 10
    dev = setup_device(args.require_gpu, args.device)
    keys = args.tasks or (RET_KEYS if args.suite == "ret" else ALL_CLS_KEYS)
    parts = out_root / "_parts"
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for model in args.models:
        path = parts / f"{args.suite}_{model}.pkl"
        if args.reuse_parts and path.exists():
            continue
        res = run_model(args.suite, model, args.split, dev["device"], keys, args.quick)
        path.write_bytes(pickle.dumps(res))
    # ------------------------------------------------------------ summarise: concatenate parts
    present = [m for m in ALL_MODEL_KEYS if (parts / f"{args.suite}_{m}.pkl").exists() and (m in args.models or args.reuse_parts)]
    allres = [pickle.loads((parts / f"{args.suite}_{m}.pkl").read_bytes()) for m in present]
    files = {"baseline": "full_embedding_scores.csv", "truncation": "truncation_curves.csv.gz", "fidelity": "fidelity.csv.gz", "deltas": "leave_one_out_deltas.csv.gz",
             "scores": "leave_one_out_scores.csv.gz", "attribution_summary": "leave_one_out_summary.csv", "ranked_removal": "ranked_removal_curves.csv.gz", "shared": "shared_dimensions.csv",
             "outliers": "outlier_removal.csv", "outlier_z": "outlier_zscores.csv.gz", "pca": "pca_curves.csv.gz", "pca_ci": "pca_ci.csv",
             "pca_variance": "pca_variance.csv.gz", "ci": "bootstrap_ci_k25_k50.csv", "probe_stats": "probe_convergence.csv", "timing": "timing.csv"}
    meta = run_metadata(dev, script="scripts/retrieval_classification_truncation_and_ranked_removal.py", protocol_id=PROTOCOL_ID, seeds={
        "random_K": f"0..{A.RET_RANDOM - 1} (ret) / 0..{A.CLS_RANDOM - 1} paired with few-shot experiments (cls)",
        "tie_break": SEED_TIE, "degrading_only_random": "1000+", "informative_only_random": "2000+", "shared_null": SEED_SHARED,
        "outlier_random_sets": "8000-8009", "bootstrap": f"{SEED_BOOT}+task index, {N_BOOT} resamples",
        "autocorr_perm": SEED_AUTOCORR}, extra={
        "suite": args.suite, "split": args.split, "models": present, "tasks": keys, "probe_tol": SWEEP_TOL if args.suite == "cls" else None,
        "quick": args.quick, "seconds": round(time.time() - t0, 1)})
    for key, fname in files.items():
        dfs = [r[key] for r in allres if key in r]
        if not dfs:
            continue
        save_table(pd.concat(dfs, ignore_index=True), out_root / fname, meta)
    print(f"wrote {out_root} ({len(present)} models) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
