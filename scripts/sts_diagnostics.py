"""STS dimension analysis, diagnostics. Final layer, float64, closed-form deletion.

  reliability and nulls
      (a) split-half reliability of the train_ranked score: 100 halvings of R (seeds 4000-4099)
      (b) Delta vectors of R (filtered STS-B train) vs AUX (SICK-R train, unfiltered)
      (c) pair bootstrap of every Delta_t(d), 1000 resamples per task (seed 5000 + task index),
          95% percentile CI; dims whose CI excludes 0. Tasks: R and each E task
      (d) random-rotation null: 20 Haar rotations (seeds 6000-6019); Delta per rotated coordinate
  shared degrading dims across the 7 E tasks vs matched-size random sets (seed 7000); descriptive
  outlier dims (|z| > 3 on the mean embedding over R's sentences) vs 10 random
      non-outlier sets of the same size (seeds 8000-8009)
  PCA (fit on the 20k NLI sample) keeping 384 components vs random 50% truncation (seeds 0-99)

Supplementary items are labelled `supp` in the outputs.

Usage: python scripts/sts_diagnostics.py [--models ...] [--out DIR] [--require-gpu]
       python scripts/sts_diagnostics.py --smoke --out <dir>   # DEV tasks stand in for E (development only)
Outputs: results/sts/diagnostics/*.csv (+ .meta.json), figures/sts_diagnostics_*.png;
render the report with scripts/report_sts_diagnostics.py. `--out DIR` writes results and figures to DIR.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from fyp.ablation import TaskData, haar_orthogonal
from fyp.bootstrap import bootstrap_spearman_torch, bootstrap_weights
from fyp.corpora import AUX_KEYS, DEV_KEYS, E_KEYS, NLI_KEY, R_KEYS
from fyp.dimension_stats import estimates_table, outlier_sets, pca_estimates, shared_dims_table
from fyp.evaluation import load_embeddings, load_layer_pairs, load_pairs_index
from fyp.models import MODEL_KEYS, ALL_MODEL_KEYS, MODEL_SPECS
from fyp.paths import FIGURES, OVERLAP_MASKS, STS_TRUNCATION, STS_DIAGNOSTICS
from fyp.plotting import BAND, C, GRID, TEXT_2, model_grid, style_axis
from fyp.protocol import (N50, N_BOOT, N_DIMS, N_GRID, SEED_BOOT, SEED_DELTA_BOOT, SEEDS_RANDOM_K, SEEDS_ROTATION, SEEDS_SPLIT_HALF,
                          Z_OUTLIER)
from fyp.runtime import add_device_args, run_metadata, save_table, setup_device, write_metadata

PROTOCOL_ID = "sts"
ABLATION_RESULTS = STS_TRUNCATION    # consistency checks against the truncation and ranked removal results

EVAL = list(E_KEYS)   # replaced by DEV_KEYS under --smoke
DEVICE = "cpu"


def boot_deltas(t: TaskData, seed: int) -> np.ndarray:
    """(N_BOOT, 768) bootstrap Delta_t(d): the same resample for rho({d}) and rho(empty) (paired)."""
    cols = np.concatenate([t.cos_full()[:, None], t.cos_single_deletions()], axis=1)
    v = bootstrap_spearman_torch(cols, t.gold, bootstrap_weights(t.n, N_BOOT, seed), device=DEVICE)
    return v[:, 1:] - v[:, :1]


def delta_stats(d):
    q1, q3 = np.percentile(d, [25, 75])
    return {"std": float(d.std()), "frac_pos": float((d > 0).mean()), "iqr_supp": float(q3 - q1), "max_abs": float(np.abs(d).max())}


# ------------------------------------------------------------------ per model
def run_model(model, smoke):
    t0 = time.time()
    layer = MODEL_SPECS[model].final_layer
    masks = np.load(OVERLAP_MASKS)
    raw = {k: load_layer_pairs(model, k, layer) for k in EVAL}
    E = {k: TaskData.from_pairs(k, *raw[k]) for k in EVAL}
    (rk,), (ak,) = R_KEYS, AUX_KEYS
    ra, rb, rg = load_layer_pairs(model, rk, layer)
    rmask = masks[f"R__{rk}"]
    raw[rk] = (ra[rmask], rb[rmask], rg[rmask])
    R = TaskData.from_pairs(rk, *raw[rk])
    AUX = TaskData.from_pairs(ak, *load_layer_pairs(model, ak, layer))
    base = {k: E[k].rho_full() for k in EVAL}
    base_mean = np.mean([base[k] for k in EVAL])
    dR = R.single_deltas()
    dE = {k: E[k].single_deltas() for k in EVAL}
    if not smoke:  # consistency with sts_ablation (same data, same code path)
        s4 = pd.read_csv(ABLATION_RESULTS / "leave_one_out_scores.csv")
        s4 = s4[(s4["model"] == model) & (s4["protocol"] == "train_ranked")].sort_values("dim")["score"].to_numpy()
        assert np.max(np.abs(s4 - dR)) < 1e-12, "train_ranked score differs from the sts_ablation result"
    out = {}

    # ---- split-half reliability: split-half reliability of the train_ranked score ----
    rows = []
    for s in SEEDS_SPLIT_HALF:
        m1 = np.zeros(R.n, dtype=bool)
        m1[np.random.default_rng(s).permutation(R.n)[: R.n // 2]] = True
        d1, d2 = R.subset(m1).single_deltas(), R.subset(~m1).single_deltas()
        r = spearmanr(d1, d2).statistic
        rows.append({"model": model, "seed": s, "n_half1": int(m1.sum()), "n_half2": int((~m1).sum()),
                     "spearman": r, "pearson": pearsonr(d1, d2).statistic,
                     "spearman_brown": 2 * r / (1 + r), "sign_agreement": float((np.sign(d1) == np.sign(d2)).mean())})
    out["r5a"] = pd.DataFrame(rows)

    # ---- ranking-set agreement: R vs AUX ----
    dA = AUX.single_deltas()
    out["r5b"] = pd.DataFrame([{"model": model, "R": rk, "AUX": ak, "n_R": R.n, "n_AUX": AUX.n,
                                "pearson": pearsonr(dR, dA).statistic, "spearman": spearmanr(dR, dA).statistic,
                                "sign_agreement": float((np.sign(dR) == np.sign(dA)).mean()),
                                "n_degrading_R": int((dR > 0).sum()), "n_degrading_AUX": int((dA > 0).sum()),
                                "n_degrading_both": int(((dR > 0) & (dA > 0)).sum())}])

    # ---- bootstrap of per-dimension scores: bootstrap CIs on every Delta ----
    tasks = {rk: R, **E}
    per_dim, summ, bootE = [], [], {}
    for ti, (k, t) in enumerate(tasks.items()):
        db = boot_deltas(t, SEED_DELTA_BOOT + ti)
        if k in EVAL:
            bootE[k] = db
        d = dR if k == rk else dE[k]
        lo, hi = np.percentile(db, 2.5, axis=0), np.percentile(db, 97.5, axis=0)
        per_dim.append(pd.DataFrame({"model": model, "task": k, "dim": np.arange(N_DIMS), "delta": d, "ci_lo": lo, "ci_hi": hi,
                                     "boot_se": db.std(0, ddof=1)}))
        summ.append({"model": model, "unit": k, "kind": "R (train_ranked score)" if k == rk else "E task", "n_pairs": t.n,
                     "n_ci_above_0": int((lo > 0).sum()), "n_ci_below_0": int((hi < 0).sum()),
                     "n_ci_excl_0": int(((lo > 0) | (hi < 0)).sum()), "median_boot_se": float(np.median(db.std(0, ddof=1))),
                     "median_abs_delta": float(np.median(np.abs(d)))})
    # supp: in_sample score = macro mean over E of independent per-task resamples
    sb = np.mean([bootE[k] for k in EVAL], axis=0)
    lo, hi = np.percentile(sb, 2.5, axis=0), np.percentile(sb, 97.5, axis=0)
    p0 = np.mean([dE[k] for k in EVAL], axis=0)
    summ.append({"model": model, "unit": "in_sample score (macro over E, supp)", "kind": "supp", "n_pairs": sum(E[k].n for k in EVAL),
                 "n_ci_above_0": int((lo > 0).sum()), "n_ci_below_0": int((hi < 0).sum()),
                 "n_ci_excl_0": int(((lo > 0) | (hi < 0)).sum()), "median_boot_se": float(np.median(sb.std(0, ddof=1))),
                 "median_abs_delta": float(np.median(np.abs(p0)))})
    out["bootstrap_dims"], out["r5c"] = pd.concat(per_dim), pd.DataFrame(summ)

    # ---- rotation null: random-rotation null ----
    rows = []
    real = {k: (dR if k == rk else dE[k]) for k in tasks}
    real["in_sample score"], real["train_ranked score"] = p0, dR
    for k, d in real.items():
        rows.append({"model": model, "unit": k, "basis": "real", "seed": -1, **delta_stats(d)})
    max_base_diff = 0.0
    for s in SEEDS_ROTATION:
        Q = haar_orthogonal(N_DIMS, s)
        drot = {}
        for k in tasks:
            a, b, g = raw[k]
            tr = TaskData.from_pairs(k, a @ Q, b @ Q, g)
            max_base_diff = max(max_base_diff, abs(tr.rho_full() - tasks[k].rho_full()))
            drot[k] = tr.single_deltas()
        drot["in_sample score"], drot["train_ranked score"] = np.mean([drot[k] for k in EVAL], axis=0), drot[rk]
        rows += [{"model": model, "unit": k, "basis": "rotated", "seed": s, **delta_stats(d)} for k, d in drot.items()]
    assert max_base_diff < 1e-6, max_base_diff  # rotation leaves cosines (hence rho(empty)) unchanged
    out["r5d"] = pd.DataFrame(rows).assign(max_base_rho_diff=max_base_diff)

    # ---- Shared degrading dims across E tasks (in-sample, descriptive) ----
    Dm = np.stack([dE[k] > 0 for k in EVAL])                     # (T, 768)
    out["r6"] = shared_dims_table(Dm, model)
    out["shared_sizes"] = pd.DataFrame({"model": model, "task": EVAL, "n_degrading_t": Dm.sum(1)})
    # supp: pairwise agreement of Delta_t between E tasks
    pw = [{"model": model, "task_a": EVAL[i], "task_b": EVAL[j], "spearman": spearmanr(dE[EVAL[i]], dE[EVAL[j]]).statistic,
           "sign_agreement": float((np.sign(dE[EVAL[i]]) == np.sign(dE[EVAL[j]])).mean())}
          for i in range(len(EVAL)) for j in range(i + 1, len(EVAL))]
    out["shared_pairwise"] = pd.DataFrame(pw)

    # ---- Outlier dims ----
    idx = load_pairs_index(rk)
    rows_R = np.unique(np.concatenate([idx["idx1"][rmask], idx["idx2"][rmask]]))
    mu = np.asarray(load_embeddings(model, rk)[rows_R, layer, :], dtype=np.float64).mean(0)
    z, outl, sets = outlier_sets(mu)
    nout = len(outl)
    out["outlier_dims"] = pd.DataFrame({"model": model, "dim": outl, "mean_value": mu[outl], "z": z[outl], "train_ranked_score": dR[outl],
                                   "in_sample_score_insample": p0[outl]})
    r7 = []
    for (name, seed), S in sets.items():
        rho = {k: E[k].rho_nested(S, [nout])[0] for k in EVAL}
        r7.append({"model": model, "set": name, "seed": seed, "n_removed": nout,
                   "RP": np.mean(list(rho.values())) / base_mean, **{f"ratio_{k}": rho[k] / base[k] for k in EVAL}})
    out["r7"] = pd.DataFrame(r7) if r7 else pd.DataFrame([{"model": model, "set": "outliers", "seed": -1, "n_removed": 0, "RP": 1.0}])
    out["outlier_z"] = pd.DataFrame({"model": model, "dim": np.arange(N_DIMS), "z": z})

    # ---- PCA (fit on NLI) vs random truncation ----
    X = np.asarray(load_embeddings(model, NLI_KEY)[:, layer, :], dtype=np.float64)
    pca = PCA(n_components=N_DIMS, svd_solver="full").fit(X)
    mean = pca.mean_
    Ec = {k: TaskData.from_pairs(k, raw[k][0] - mean, raw[k][1] - mean, raw[k][2]) for k in EVAL}
    Ep = {k: TaskData.from_pairs(k, pca.transform(raw[k][0]), pca.transform(raw[k][1]), raw[k][2]) for k in EVAL}
    base_c = {k: Ec[k].rho_full() for k in EVAL}
    drop_low = np.arange(N_DIMS - 1, -1, -1)  # delete the lowest-variance components first
    rows = []
    for k in EVAL:
        rho = Ep[k].rho_nested(drop_low, N_GRID)
        assert abs(rho[0] - base_c[k]) < 1e-9, (k, rho[0], base_c[k])  # PCA's K=0 anchor = centred baseline
        rows += [{"model": model, "method": "pca", "seed": -1, "task": k, "n": n, "rho": r} for n, r in zip(N_GRID, rho)]
        for s in SEEDS_RANDOM_K:
            perm = np.random.default_rng(s).permutation(N_DIMS)
            rows.append({"model": model, "method": "random", "seed": s, "task": k, "n": N50, "rho": E[k].rho_nested(perm, [N50])[0]})
            rows.append({"model": model, "method": "random_centred_supp", "seed": s, "task": k, "n": N50,
                         "rho": Ec[k].rho_nested(perm, [N50])[0]})
    r8 = pd.DataFrame(rows)
    out["pca_curves"] = r8
    out["pca_base"] = pd.DataFrame([{"model": model, "task": k, "rho_raw": base[k], "rho_centred": base_c[k]} for k in EVAL])
    out["pca_variance"] = pd.DataFrame({"model": model, "n_components": np.arange(1, N_DIMS + 1),
                                  "cum_explained_variance": np.cumsum(pca.explained_variance_ratio_)})
    if not smoke:
        r1 = pd.read_csv(ABLATION_RESULTS / "truncation_summary.csv")
        s4 = r1[(r1["model"] == model) & (r1["method"] == "random") & (r1["n"] == N50)]["RP_mean"].iloc[0]
        here = r8[r8["method"] == "random"].groupby("seed")["rho"].mean().mean() / base_mean
        assert abs(s4 - here) < 1e-12, (s4, here)

    # headline-style paired bootstrap (seed 9000 + task index, as in sts_ablation): PCA-384 vs random-50%
    boot = {}
    for ti, k in enumerate(EVAL):
        cols = [E[k].cos_full()[:, None], Ec[k].cos_full()[:, None], Ep[k].cos_nested(drop_low, [N50])]
        cols += [E[k].cos_nested(np.random.default_rng(s).permutation(N_DIMS), [N50]) for s in SEEDS_RANDOM_K]
        boot[k] = bootstrap_spearman_torch(np.concatenate(cols, axis=1), E[k].gold,
                                           bootstrap_weights(E[k].n, N_BOOT, SEED_BOOT + ti), device=DEVICE)
    mb = lambda j: np.mean([boot[k][:, j] for k in EVAL], axis=0)
    full, cent, pc = mb(0), mb(1), mb(2)
    rnd = np.mean([mb(3 + i) for i in range(len(SEEDS_RANDOM_K))], axis=0) / full
    out["pca_ci"] = pd.DataFrame(estimates_table(pca_estimates(full, cent, pc, rnd), {"model": model}))
    print(f"  {model}: done in {time.time() - t0:.0f}s (outliers: {nout}; max rotated base diff {max_base_diff:.1e})")
    return out


# ------------------------------------------------------------------ figures
def fig_r5(res, models, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    y = np.arange(len(models))[::-1]
    a = res["r5a"]
    for yi, m in zip(y, models):
        v = a[a["model"] == m]["spearman"]
        axes[0].plot([np.percentile(v, 2.5), np.percentile(v, 97.5)], [yi, yi], color=C["blue"], lw=2)
        axes[0].plot(v.mean(), yi, "o", color=C["blue"], ms=7)
    axes[0].set_title("(a) split-half Spearman of the train_ranked score\n(mean, 95% range over 100 halvings of R)", fontsize=9)
    b = res["r5b"].set_index("model")
    axes[1].barh(y, [b.loc[m, "spearman"] for m in models], color=C["orange"], height=0.5)
    axes[1].set_title("(b) Spearman of Δ vectors:\nR (STS-B train, filtered) vs AUX (SICK-R train)", fontsize=9)
    c = res["r5c"]
    for off, (unit_kind, col, lbl) in zip((0.15, -0.15), (("R (train_ranked score)", C["blue"], "R"), ("E task", C["aqua"], "E tasks (each)"))):
        for yi, m in zip(y, models):
            v = c[(c["model"] == m) & (c["kind"] == unit_kind)]["n_ci_excl_0"] / N_DIMS
            axes[2].plot(v, np.full(len(v), yi + off), "o", color=col, ms=6, alpha=0.8, label=lbl if m == models[0] else None)
    axes[2].axvline(0.05, color=TEXT_2, lw=0.8, ls=":")
    axes[2].set_title("(c) fraction of dims whose 95% bootstrap CI of Δ\nexcludes 0 (dotted: 5%, all-null reference)", fontsize=9)
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.set_yticks(y); ax.set_yticklabels(models, fontsize=8); style_axis(ax); ax.grid(axis="x", color=GRID, lw=0.8)
    fig.suptitle(f"Reliability of per-dimension attribution", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_r5d(res, models, path):
    """Per model (rows) and unit (dots): rotated-basis mean over 20 rotations relative to the real axes."""
    d = res["r5d"]
    rk = R_KEYS[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    y = np.arange(len(models))[::-1]
    specs = (("std", "ratio", "(a) std(Δ): rotated / real\n(<1: real axes more dispersed)"),
             ("iqr_supp", "ratio", "(b) IQR(Δ): rotated / real (supp.)\nrobust to a few extreme dims"),
             ("frac_pos", "diff", "(c) fraction with Δ > 0:\nreal − rotated"))
    for ax, (col, how, title) in zip(axes, specs):
        labelled = set()
        for yi, m in zip(y, models):
            dm = d[d["model"] == m]
            real = dm[dm["basis"] == "real"].set_index("unit")[col]
            rot = dm[dm["basis"] == "rotated"].groupby("unit")[col].mean()
            for u in real.index:
                if u in ("in_sample score", "train_ranked score"):
                    continue
                v = rot[u] / real[u] if how == "ratio" else real[u] - rot[u]
                is_r = u == rk
                lbl = "R (STS-B train)" if is_r else "E tasks (each)"
                ax.plot(v, yi + (0.18 if is_r else -0.12), "o", color=C["blue"] if is_r else C["aqua"], ms=6, alpha=0.85,
                        label=None if lbl in labelled else lbl)
                labelled.add(lbl)
        ax.axvline(1 if how == "ratio" else 0, color=TEXT_2, lw=1)
        ax.set_title(title, fontsize=9); style_axis(ax); ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].set_yticks(y); axes[0].set_yticklabels(models, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("rotation null: single-coordinate Δ in the real basis vs 20 Haar-random rotated bases (mean over rotations)", fontsize=11)
    fig.tight_layout()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))   # below the panels: no data under it
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_r6(res, models, path):
    def draw(ax, m):
        d = res["r6"][res["r6"]["model"] == m]
        ax.bar(d["n_tasks"] - 0.2, d["observed"], width=0.38, color=C["blue"], label="observed")
        ax.bar(d["n_tasks"] + 0.2, d["null_mean"], width=0.38, color=BAND, label="matched-size random null (mean)")
        ax.errorbar(d["n_tasks"] + 0.2, d["null_mean"], yerr=[d["null_mean"] - d["null_lo"], d["null_hi"] - d["null_mean"]],
                    fmt="none", ecolor=TEXT_2, lw=1, capsize=2, label="null 95% range")
    model_grid(models, draw, title="Shared degrading dims across the 7 E tasks (IN-SAMPLE, descriptive)",
               xlabel="number of E tasks in which the dim is degrading (Δ_t > 0)", ylabel="number of dims", path=path, ncol=3)


def fig_r7(res, models, path):
    def draw(ax, m):
        z = res["outlier_z"][res["outlier_z"]["model"] == m]
        out = z["z"].abs() > Z_OUTLIER
        ax.vlines(z["dim"][~out], 0, z["z"][~out], color=BAND, lw=0.8, label="|z| ≤ 3")
        ax.vlines(z["dim"][out], 0, z["z"][out], color=C["orange"], lw=1.8, label="outlier (|z| > 3)")
        for s in (-Z_OUTLIER, Z_OUTLIER):
            ax.axhline(s, color=TEXT_2, lw=0.8, ls=":")
        ax.set_title(f"{m} ({int(out.sum())} outliers)", fontsize=10)
    model_grid(models, draw, title="Outlier dimensions of the mean final-layer embedding over R's sentences",
               xlabel="dimension index", ylabel="z-score of mean embedding coordinate", path=path, ncol=2)


def fig_r8(res, models, path, ablation_r1):
    def draw(ax, m):
        base = res["pca_base"][res["pca_base"]["model"] == m]
        bm, bc = base["rho_raw"].mean(), base["rho_centred"].mean()
        if ablation_r1 is not None:
            r = ablation_r1[(ablation_r1["model"] == m) & (ablation_r1["method"] == "random") & (ablation_r1["n"].isin(N_GRID))].sort_values("n")
            ax.fill_between(r["K"], r["band_lo"], r["band_hi"], color=BAND, alpha=0.6, lw=0, label="random-K 95% band")
            ax.plot(r["K"], r["RP_mean"], color=TEXT_2, lw=1.5, label="random-K mean")
        p = res["pca_curves"]
        p = p[(p["model"] == m) & (p["method"] == "pca")].groupby("n")["rho"].mean()
        ax.plot(100 * p.index / N_DIMS, p.values / bm, color=C["blue"], lw=2, label="PCA (fit on NLI), RP vs raw baseline")
        ax.plot(100 * p.index / N_DIMS, p.values / bc, color=C["aqua"], lw=1.5, ls="--", label="PCA, RP vs centred baseline")
        ax.axvline(50, color=TEXT_2, lw=0.8, ls=":")
    model_grid(models, draw, title="PCA truncation (label-free, fit on 20k NLI sentences) vs random truncation",
               xlabel="K = % of dimensions (components) removed", ylabel="relative performance (RP) on E", path=path, ncol=4)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=ALL_MODEL_KEYS)
    ap.add_argument("--smoke", action="store_true", help="development run: DEV tasks stand in for E; requires --out")
    ap.add_argument("--out", type=Path, default=None, help="write results and figures here instead of results/sts/diagnostics and figures/")
    add_device_args(ap)
    args = ap.parse_args()
    global EVAL, DEVICE
    if args.smoke:
        assert args.out is not None, "--smoke needs --out (never writes to results/)"
        EVAL = list(DEV_KEYS)
    models = [m for m in ALL_MODEL_KEYS if m in args.models]
    dev = setup_device(args.require_gpu, args.device)
    DEVICE = dev["device"]
    out_dir = args.out or STS_DIAGNOSTICS
    fig_dir = args.out or FIGURES
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    seeds = {"split_half": "4000-4099", "delta_bootstrap": f"{SEED_DELTA_BOOT}+task_index (R=0, E tasks 1-7)", "rotations": "6000-6019",
             "shared_null": 7000, "outlier_random_sets": "8000-8009", "random_K": "0-99", "pca_bootstrap": f"{SEED_BOOT}+E task index"}
    meta = run_metadata(dev, script="scripts/sts_diagnostics.py", protocol_id=PROTOCOL_ID, seeds=seeds,
                        extra={"note": "ablation arithmetic is float64 numpy on CPU; bootstrap Spearman is float64 torch on the device",
                               "models": models, "eval_tasks": EVAL, "smoke": args.smoke})

    parts = [run_model(m, args.smoke) for m in models]
    res = {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}
    names = {"r5a": "reliability_split_half.csv", "r5b": "ranking_set_agreement.csv", "r5c": "leave_one_out_bootstrap_summary.csv",
             "bootstrap_dims": "leave_one_out_bootstrap_dims.csv.gz", "r5d": "rotation_null.csv", "r6": "shared_dimensions.csv", "shared_sizes": "shared_dimensions_task_sizes.csv",
             "shared_pairwise": "shared_dimensions_pairwise.csv", "r7": "outlier_removal.csv", "outlier_dims": "outlier_dimensions.csv", "outlier_z": "outlier_zscores.csv.gz",
             "pca_curves": "pca_curves.csv.gz", "pca_base": "pca_baselines.csv", "pca_variance": "pca_variance.csv.gz", "pca_ci": "pca_ci.csv"}
    for k, name in names.items():
        save_table(res[k], out_dir / name, meta)
    ablation_r1 = None if args.smoke else pd.read_csv(ABLATION_RESULTS / "truncation_summary.csv")
    figs = {"sts_reliability.png": lambda p: fig_r5(res, models, p),
            "sts_rotation_null.png": lambda p: fig_r5d(res, models, p),
            "sts_shared_dimensions.png": lambda p: fig_r6(res, models, p),
            "sts_outliers.png": lambda p: fig_r7(res, models, p),
            "sts_pca.png": lambda p: fig_r8(res, models, p, ablation_r1)}
    for name, fn in figs.items():
        fn(fig_dir / name)
        write_metadata(fig_dir / name, meta)
    print(f"git {meta['git_commit']} dirty={meta['git_dirty']}")


if __name__ == "__main__":
    main()
