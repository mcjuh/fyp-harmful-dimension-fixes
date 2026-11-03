"""STS ablation, truncation and ranked removal + headline paired-bootstrap CIs.

All on the final layer of each model, float64, closed-form deletion (fyp.ablation).
  truncation curves on E: last-K, first-K, random-K (seeds 0-99)
  label-free fidelity: Spearman(full cos, truncated cos)
  per-dimension attribution under in_sample / train_ranked / held_out (per fold): Delta, score, degrading/informative/neutral,
      block-of-64 fractions, lag-1 sign autocorrelation vs permutation null (seed 3000)
  ranked removal curves on E: best-first, worst-first, degrading-only / informative-only (ranked + 100 random orders),
      vs the random band; interaction check (sum of single scores vs actual joint change)
  CI  paired bootstrap (1000 pair-resamples per task, seed 9000 + task index) of RP at K=25%, 50%

Usage: python scripts/sts_truncation_and_ranked_removal.py [--models ...] [--reuse-parts] [--out DIR] [--require-gpu]
Outputs: results/sts/truncation_and_ranked_removal/*.csv (+ .meta.json), figures/sts_*.png; render the report with
scripts/report_sts.py. `--out DIR` writes results, parts and figures to DIR (development / verification).
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fyp.ablation import TaskData, brute_force_rho, ranked_order
from fyp.bootstrap import bootstrap_spearman, bootstrap_weights
from fyp.corpora import E_KEYS, R_KEYS
from fyp.dimension_stats import estimates_table, headline_estimates, removal_curve_specs, score_summary
from fyp.evaluation import load_layer_pairs
from fyp.models import MODEL_KEYS, ALL_MODEL_KEYS, MODEL_SPECS
from fyp.timing import StageTimer
from fyp.paths import FIGURES, OVERLAP_MASKS, STS_TRUNCATION
from fyp.plotting import BAND, C, TEXT_2, group_title, model_grid, style_axis
from fyp.protocol import (CONTRASTIVE_MODELS as CONTRASTIVE, RANKING_POWER_K, K_GRID, N25, N50, N_BOOT, N_DIMS, N_GRID, SEED_AUTOCORR,
                          SEED_BOOT, SEED_TIE, SEEDS_DEGRADING_RANDOM, SEEDS_INFORMATIVE_RANDOM, SEEDS_RANDOM_K, n_remove)
from fyp.runtime import add_device_args, run_metadata, save_table, setup_device, write_metadata

PROTOCOL_ID = "sts"
RUN_MODELS = list(MODEL_KEYS)  # set from --models in main()


# ------------------------------------------------------------------ data
def load_tasks(model):
    layer = MODEL_SPECS[model].final_layer
    masks = np.load(OVERLAP_MASKS)
    E = {k: TaskData.from_pairs(k, *load_layer_pairs(model, k, layer)) for k in E_KEYS}
    R = {k: TaskData.from_pairs(k, *load_layer_pairs(model, k, layer)).subset(masks[f"R__{k}"]) for k in R_KEYS}
    held_out = {t: {u: E[u].subset(masks[f"held_out__{t}__{u}"], f"{u}|not-{t}") for u in E_KEYS if u != t} for t in E_KEYS}
    return layer, E, R, held_out



# ------------------------------------------------------------------ per model
def run_model(model, meta_extra):
    t0 = time.time()
    timer = StageTimer(model=model, suite="sts")
    layer, E, R, held_out = load_tasks(model)
    base = {k: E[k].rho_full() for k in E_KEYS}
    timer.mark("load_tasks", "load")

    # --- brute-force spot check on real data (closed form vs masked cosine + scipy) ---
    a, b, _ = load_layer_pairs(model, "sts12_test", layer)
    perm = np.random.default_rng(0).permutation(N_DIMS)
    gold = E["sts12_test"].gold
    cf = E["sts12_test"].rho_nested(perm, [n_remove(90)])[0]
    bf = brute_force_rho(a, b, gold, perm[:n_remove(90)])
    d3 = E["sts12_test"].single_deltas()[[0, 300, 767]]
    bf3 = [brute_force_rho(a, b, gold, [d]) - base["sts12_test"] for d in (0, 300, 767)]
    assert abs(cf - bf) < 1e-10 and np.max(np.abs(d3 - bf3)) < 1e-10, (cf, bf, d3, bf3)
    timer.mark("closed_form_spot_check", "bookkeeping")

    # --- Per-dimension deltas and scores per protocol unit ---
    deltas_rows, units = [], {}
    dE = {k: E[k].single_deltas() for k in E_KEYS}
    dR = {k: R[k].single_deltas() for k in R_KEYS}
    for k in E_KEYS:
        deltas_rows.append(("in_sample", "all", k, dE[k]))
    units[("in_sample", "all")] = {"score": np.mean([dE[k] for k in E_KEYS], 0), "eval": E_KEYS, "rank_tasks": {k: E[k] for k in E_KEYS},
                            "rank_deltas": {k: dE[k] for k in E_KEYS}}
    for k in R_KEYS:
        deltas_rows.append(("train_ranked", "all", k, dR[k]))
    units[("train_ranked", "all")] = {"score": np.mean([dR[k] for k in R_KEYS], 0), "eval": E_KEYS, "rank_tasks": dict(R),
                            "rank_deltas": dict(dR)}
    for t in E_KEYS:
        rt = {**R, **held_out[t]}
        rd = {**dR, **{u: held_out[t][u].single_deltas() for u in held_out[t]}}
        for k, v in rd.items():
            deltas_rows.append(("held_out", t, k, v))
        units[("held_out", t)] = {"score": np.mean(list(rd.values()), 0), "eval": [t], "rank_tasks": rt, "rank_deltas": rd}
    deltas_df = pd.concat([pd.DataFrame({"model": model, "protocol": p, "fold": f, "ranking_task": k,
                                         "dim": np.arange(N_DIMS), "delta": v}) for p, f, k, v in deltas_rows])
    scores_df = pd.concat([pd.DataFrame({"model": model, "protocol": p, "fold": f, "dim": np.arange(N_DIMS), "score": u["score"]})
                           for (p, f), u in units.items()])
    r3_rows = [{"model": model, "protocol": p, "fold": f, "n_ranking_tasks": len(u["rank_tasks"]),
                "ranking_pairs": int(sum(x.n for x in u["rank_tasks"].values())), **score_summary(u["score"], SEED_AUTOCORR)}
               for (p, f), u in units.items()]

    timer.mark("attribution_leave_one_out_and_ranking", "one_time")
    special_n = sorted({int((u["score"] > 0).sum()) for u in units.values()} | {int((u["score"] < 0).sum()) for u in units.values()})
    ns_all = sorted(set(N_GRID) | set(special_n))

    # --- truncation curves and fidelity on E ---
    orders = {"last": np.arange(N_DIMS - 1, -1, -1), "first": np.arange(N_DIMS)}
    for s in SEEDS_RANDOM_K:
        orders[f"random:{s}"] = np.random.default_rng(s).permutation(N_DIMS)
    r1_rows, r2_rows = [], []
    for name, order in orders.items():
        method, seed = (name.split(":")[0], int(name.split(":")[1])) if ":" in name else (name, -1)
        for k in E_KEYS:
            rho = E[k].rho_nested(order, ns_all)
            assert abs(rho[0] - base[k]) < 1e-12  # K=0 anchor
            r1_rows += [{"model": model, "method": method, "seed": seed, "task": k, "n": n, "rho": r} for n, r in zip(ns_all, rho)]
            fid = E[k].fidelity_nested(order, N_GRID)
            r2_rows += [{"model": model, "method": method, "seed": seed, "task": k, "n": n, "K": kk, "fidelity": f}
                        for n, kk, f in zip(N_GRID, K_GRID, fid)]
    r1 = pd.DataFrame(r1_rows)
    timer.mark("truncation_curves_R1_R2", "bookkeeping")

    # --- Ranked removal curves ---
    r4_rows, inter_rows = [], []
    for (p, f), u in units.items():
        score = u["score"]
        curves, best, worst, nD, nI = removal_curve_specs(score, SEEDS_DEGRADING_RANDOM, SEEDS_INFORMATIVE_RANDOM)
        for (cn, seed), (order, ns) in curves.items():
            for k in u["eval"]:
                rho = E[k].rho_nested(order, ns)
                assert abs(rho[0] - base[k]) < 1e-12
                r4_rows += [{"model": model, "protocol": p, "fold": f, "curve": cn, "seed": seed, "task": k, "n": n, "rho": r,
                             "n_degrading": nD, "n_informative": nI} for n, r in zip(ns, rho)]
        # interaction check along best-first up to |D|
        ns = [n for n in curves[("degrading_only_ranked", -1)][1] if n > 0]
        rank_base = {k: x.rho_full() for k, x in u["rank_tasks"].items()}
        actual_rank = np.mean([u["rank_tasks"][k].rho_nested(best, ns) - rank_base[k] for k in u["rank_tasks"]], 0)
        e_single = np.mean([dE[k] for k in u["eval"]], 0)
        actual_e = np.mean([E[k].rho_nested(best, ns) - base[k] for k in u["eval"]], 0)
        for i, n in enumerate(ns):
            inter_rows.append({"model": model, "protocol": p, "fold": f, "n": n,
                               "predicted_ranking": float(score[best[:n]].sum()), "actual_ranking": float(actual_rank[i]),
                               "predicted_E": float(e_single[best[:n]].sum()), "actual_E": float(actual_e[i])})
    r4 = pd.DataFrame(r4_rows)
    timer.mark("removal_curves_R4", "bookkeeping")

    # --- headline paired bootstrap at K = 25, 50 ---
    ci_rows = []
    boot = {}
    for ti, k in enumerate(E_KEYS):
        cols, names = [E[k].cos_full()[:, None]], ["full"]
        for nm in ("last",):
            cols.append(E[k].cos_nested(orders[nm], [N25, N50])); names += [f"{nm}_25", f"{nm}_50"]
        for s in SEEDS_RANDOM_K:
            cols.append(E[k].cos_nested(orders[f"random:{s}"], [N25, N50])); names += [f"random{s}_25", f"random{s}_50"]
        for p in ("train_ranked", "held_out"):
            sc = units[(p, "all" if p == "train_ranked" else k)]["score"]
            for cn, desc in (("best", True), ("worst", False)):
                cols.append(E[k].cos_nested(ranked_order(sc, desc, SEED_TIE), [N25, N50])); names += [f"{p}_{cn}_25", f"{p}_{cn}_50"]
        W = bootstrap_weights(E[k].n, N_BOOT, SEED_BOOT + ti)
        vals = bootstrap_spearman(np.concatenate(cols, axis=1), E[k].gold, W)
        boot[k] = dict(zip(names, vals.T))
    full_mean = np.mean([boot[k]["full"] for k in E_KEYS], 0)
    rp = lambda nm: np.mean([boot[k][nm] for k in E_KEYS], 0) / full_mean
    for K in (25, 50):
        est = headline_estimates(rp, K, [f"random{s}" for s in SEEDS_RANDOM_K], ("train_ranked", "held_out"))
        ci_rows += estimates_table(est, {"model": model, "K": K})

    timer.mark("headline_bootstrap", "bookkeeping")
    print(f"  {model}: done in {time.time() - t0:.0f}s")
    return {"deltas": deltas_df, "scores": scores_df, "attribution_summary": pd.DataFrame(r3_rows), "truncation": r1, "fidelity": pd.DataFrame(r2_rows),
            "ranked_removal": r4, "interaction": pd.DataFrame(inter_rows), "ci": pd.DataFrame(ci_rows),
            "baseline": pd.DataFrame([{"model": model, "task": k, "rho_full": base[k]} for k in E_KEYS]), "timing": timer.table()}


# ------------------------------------------------------------------ aggregation
def rp_curve(df, base_mean, group_cols):
    """RP = mean over tasks of rho / mean over tasks of baseline, per group."""
    g = df.groupby(group_cols + ["n"])["rho"].mean().reset_index()
    g["RP"] = g["rho"] / base_mean
    return g


def summarize(res):
    base = res["baseline"]
    bm = base.groupby("model")["rho_full"].mean()
    r1 = res["truncation"]
    rows = []
    for model, d in r1.groupby("model"):
        per_seed = d.groupby(["method", "seed", "n"])["rho"].mean().reset_index()
        per_seed["RP"] = per_seed["rho"] / bm[model]
        for (method, n), g in per_seed.groupby(["method", "n"]):
            v = g.sort_values("seed")["RP"].to_numpy()
            rows.append({"model": model, "method": method, "n": n, "K": 100 * n / N_DIMS, "RP_mean": v.mean(),
                         "RP_std100": v.std(ddof=1) if len(v) > 1 else 0.0, "RP_std_first10": v[:10].std(ddof=1) if len(v) > 1 else 0.0,
                         "band_lo": np.percentile(v, 2.5), "band_hi": np.percentile(v, 97.5)})
    r1s = pd.DataFrame(rows)

    # per-task random band (ratio to that task's baseline)
    r1t = r1.merge(base, on=["model", "task"])
    r1t["ratio"] = r1t["rho"] / r1t["rho_full"]
    r1ts = (r1t.groupby(["model", "method", "task", "n"])["ratio"]
            .agg(mean="mean", band_lo=lambda x: np.percentile(x, 2.5), band_hi=lambda x: np.percentile(x, 97.5)).reset_index())

    # Ranked removal aggregated: in_sample/train_ranked macro over E; held_out = mean over folds' own task / mean baseline (grid points only)
    r4 = res["ranked_removal"]
    agg = []
    for (model, p), d in r4.groupby(["model", "protocol"]):
        dd = d if p != "held_out" else d[d["n"].isin(N_GRID)]
        g = dd.groupby(["curve", "seed", "n"])
        m = g["rho"].mean()
        cnt = g["rho"].size()
        m = m[cnt == len(E_KEYS)].reset_index()  # all 7 tasks/folds present at this n
        m["RP"] = m["rho"] / bm[model]
        m["model"], m["protocol"] = model, p
        agg.append(m)
    r4a = pd.concat(agg)
    r4s = []
    for (model, p, curve, n), g in r4a.groupby(["model", "protocol", "curve", "n"]):
        v = g["RP"].to_numpy()
        r4s.append({"model": model, "protocol": p, "curve": curve, "n": n, "K": 100 * n / N_DIMS, "RP": v.mean(),
                    "band_lo": np.percentile(v, 2.5) if len(v) > 1 else np.nan, "band_hi": np.percentile(v, 97.5) if len(v) > 1 else np.nan})
    r4s = pd.DataFrame(r4s)

    # per-fold / per-protocol special point K=|D| (degrading-only ranked end point)
    dpt = []
    for (model, p, f), d in r4[r4["curve"] == "degrading_only_ranked"].groupby(["model", "protocol", "fold"]):
        nD = int(d["n_degrading"].iloc[0])
        e = d[d["n"] == nD].merge(base, on=["model", "task"])
        dpt.append({"model": model, "protocol": p, "fold": f, "n_degrading": nD,
                    "rho_mean_at_n_degrading": e["rho"].mean(), "base_mean": e["rho_full"].mean()})
    dpt = pd.DataFrame(dpt)
    return r1s, r1ts, r4s, dpt


def hypotheses(r1s, r4s, dpt):
    rows = []
    rand = r1s[r1s["method"] == "random"].set_index(["model", "n"])
    for model in RUN_MODELS:
        h1 = rand.loc[(model, n_remove(50)), "RP_mean"]
        row = {"model": model, "group": "contrastive" if model in CONTRASTIVE else "raw",
               "random50_RP": h1, "random_removal_robust": bool(h1 >= 0.90) if model in CONTRASTIVE else None}
        for p in ("in_sample", "train_ranked", "held_out"):
            s = r4s[(r4s["model"] == model) & (r4s["protocol"] == p)].set_index(["curve", "n"])
            above, below, ever_above = [], [], False
            for K in K_GRID[1:]:
                n = n_remove(K)
                hi, lo = rand.loc[(model, n), "band_hi"], rand.loc[(model, n), "band_lo"]
                bf, wf = s.loc[("best_first", n), "RP"], s.loc[("worst_first", n), "RP"]
                ever_above |= bool(bf > hi)
                if K in RANKING_POWER_K:
                    above.append(bool(bf > hi)); below.append(bool(wf < lo))
            row[f"{p}_best_above_band_all_K10_50"] = all(above)
            row[f"{p}_worst_below_band_all_K10_50"] = all(below)
            row[f"{p}_best_above_band_count_K10_50"] = f"{sum(above)}/{len(above)}"
            row[f"{p}_worst_below_band_count_K10_50"] = f"{sum(below)}/{len(below)}"
            row[f"{p}_best_ever_above_band"] = ever_above
            best50 = s.loc[("best_first", n_remove(50)), "RP"]
            row[f"{p}_best_first_RP50"] = best50
        row["heldout_ranking_power"] = all(row[f"{p}_best_above_band_all_K10_50"] and row[f"{p}_worst_below_band_all_K10_50"] for p in ("train_ranked", "held_out"))
        row["no_heldout_power"] = not (row["train_ranked_best_ever_above_band"] or row["held_out_best_ever_above_band"])
        row["in_sample_only_gain"] = row["in_sample_best_ever_above_band"] and not (row["train_ranked_best_ever_above_band"] or row["held_out_best_ever_above_band"])
        for p in ("train_ranked", "held_out"):
            d = dpt[(dpt["model"] == model) & (dpt["protocol"] == p)]
            row[f"{p}_degrading_only_RP_at_n_degrading"] = d["rho_mean_at_n_degrading"].mean() / d["base_mean"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_r1(r1s, path):
    def draw(ax, model):
        d = r1s[(r1s["model"] == model) & (r1s["n"].isin(N_GRID))]
        r = d[d["method"] == "random"].sort_values("n")
        ax.fill_between(r["K"], r["band_lo"], r["band_hi"], color=BAND, alpha=0.6, lw=0, label="random-K 95% band")
        ax.plot(r["K"], r["RP_mean"], color=TEXT_2, lw=1.5, label="random-K mean (100 seeds)")
        for m, c in (("last", C["blue"]), ("first", C["orange"])):
            x = d[d["method"] == m].sort_values("n")
            ax.plot(x["K"], x["RP_mean"], color=c, lw=2, label=f"{m}-K")
        ax.axhline(0.9, color=TEXT_2, lw=0.8, ls=":")
    model_grid(RUN_MODELS, draw, title=f"Truncation curves on E (label-free); dotted = 0.90",
               xlabel="K = % of dimensions removed", ylabel="relative performance (RP)", path=path, ncol=4, title_fn=group_title)


def fig_r4(r1s, r4s, protocol, path):
    def draw(ax, model):
        r = r1s[(r1s["model"] == model) & (r1s["method"] == "random") & (r1s["n"].isin(N_GRID))].sort_values("n")
        ax.fill_between(r["K"], r["band_lo"], r["band_hi"], color=BAND, alpha=0.6, lw=0, label="random-K 95% band")
        s = r4s[(r4s["model"] == model) & (r4s["protocol"] == protocol)]
        for cv, c, lbl in (("best_first", C["blue"], "best-first"), ("worst_first", C["orange"], "worst-first"),
                           ("degrading_only_ranked", C["aqua"], "degrading-only, ranked"), ("informative_only_ranked", C["yellow"], "informative-only, ranked")):
            x = s[s["curve"] == cv].sort_values("n")
            if protocol == "held_out":
                x = x[x["n"].isin(N_GRID)]
            ax.plot(x["K"], x["RP"], color=c, lw=2 if cv in ("best_first", "worst_first") else 1.5,
                    ls="-" if cv in ("best_first", "worst_first") else "--", label=lbl)
            if cv == "degrading_only_ranked" and protocol != "held_out" and len(x):
                ax.plot(x["K"].iloc[-1], x["RP"].iloc[-1], "o", color=c, ms=6)
    label = {"in_sample": "in_sample: ranked on E, evaluated on E (IN-SAMPLE, optimistic)", "train_ranked": "train_ranked: ranked on filtered STS-B train, evaluated on E",
             "held_out": "held_out: leave-one-task-out (fold t evaluated on t only; grid points)"}[protocol]
    model_grid(RUN_MODELS, draw, title=f"Ranked removal curves: {label}", xlabel="K = % of dimensions removed",
               ylabel="relative performance (RP) on E", path=path, ncol=5)


def fig_r3(scores, path):
    def draw(ax, model):
        d = scores[scores["model"] == model]
        for p, c in (("in_sample", C["blue"]), ("train_ranked", C["orange"])):
            v = np.sort(d[d["protocol"] == p]["score"].to_numpy())[::-1]
            ax.plot(np.arange(N_DIMS), 1e4 * v, color=c, lw=1.8, label=f"{p}" + (" (in-sample)" if p == "in_sample" else ""))
        for i, (f, g) in enumerate(d[d["protocol"] == "held_out"].groupby("fold")):
            v = np.sort(g["score"].to_numpy())[::-1]
            ax.plot(np.arange(N_DIMS), 1e4 * v, color=C["aqua"], lw=0.8, alpha=0.7, label="held_out folds" if i == 0 else None)
        ax.axhline(0, color=TEXT_2, lw=0.8)
    model_grid(RUN_MODELS, draw, title="Per-dimension attribution scores, sorted (count of degrading dims is not evidence on its own)",
               xlabel="dimension rank (by score, descending)", ylabel="score(d) × 1e4  (>0: removal helps)", path=path, ncol=3)


def fig_per_task(r1ts, r4, base, path):
    """Per-task curves (ratio to task baseline) for train_ranked and held_out best/worst-first vs per-task random band."""
    fig, axes = plt.subplots(max(len(RUN_MODELS), 2), len(E_KEYS), figsize=(21, 15), sharex=True)
    for i, model in enumerate(RUN_MODELS):
        for j, k in enumerate(E_KEYS):
            ax = axes[i, j]
            r = r1ts[(r1ts["model"] == model) & (r1ts["method"] == "random") & (r1ts["task"] == k) & (r1ts["n"].isin(N_GRID))].sort_values("n")
            K = 100 * r["n"] / N_DIMS
            ax.fill_between(K, r["band_lo"], r["band_hi"], color=BAND, alpha=0.6, lw=0)
            b0 = base[(base["model"] == model) & (base["task"] == k)]["rho_full"].iloc[0]
            for p, fold, ls in (("train_ranked", "all", "-"), ("held_out", k, "--")):
                for cv, c in (("best_first", C["blue"]), ("worst_first", C["orange"])):
                    x = r4[(r4["model"] == model) & (r4["protocol"] == p) & (r4["fold"] == fold) & (r4["curve"] == cv)
                           & (r4["task"] == k) & (r4["n"].isin(N_GRID))].sort_values("n")
                    ax.plot(100 * x["n"] / N_DIMS, x["rho"] / b0, color=c, lw=1.3, ls=ls)
            if i == 0:
                ax.set_title(k, fontsize=10)
            if j == 0:
                ax.set_ylabel(model, fontsize=10)
            style_axis(ax)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C["blue"], lw=1.5), Line2D([], [], color=C["orange"], lw=1.5),
         Line2D([], [], color=TEXT_2, lw=1.5, ls="-"), Line2D([], [], color=TEXT_2, lw=1.5, ls="--"),
         plt.Rectangle((0, 0), 1, 1, color=BAND)]
    fig.legend(h, ["best-first", "worst-first", "train_ranked", "held_out (fold = this task)", "random-K 95% band"],
               loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Per-task ranked removal curves: ρ_t(S_K)/ρ_t(∅) vs K (% removed); held-out protocols only", y=0.999, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=ALL_MODEL_KEYS)
    ap.add_argument("--reuse-parts", action="store_true",
                    help="load per-model raw outputs from <out>/_parts/ instead of recomputing "
                         "(their computing commit is recorded in the metadata)")
    ap.add_argument("--out", type=Path, default=None, help="write results, parts and figures here instead of results/sts/truncation_and_ranked_removal and figures/")
    add_device_args(ap)
    args = ap.parse_args()
    global RUN_MODELS
    RUN_MODELS = [m for m in ALL_MODEL_KEYS if m in args.models]
    dev = setup_device(args.require_gpu, args.device)
    out_dir, fig_dir = (args.out, args.out) if args.out else (STS_TRUNCATION, FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds = {"random_K": "0-99", "tie_break": SEED_TIE, "autocorr_perm": SEED_AUTOCORR, "degrading_only_random": "1000-1099",
             "informative_only_random": "2000-2099", "bootstrap": f"{SEED_BOOT}+task_index"}
    meta = run_metadata(dev, script="scripts/sts_truncation_and_ranked_removal.py", protocol_id=PROTOCOL_ID, seeds=seeds,
                        extra={"note": "all ablation arithmetic is float64 numpy on CPU (closed form); device field records torch device selection only",
                               "models": args.models})

    parts_dir = out_dir / "_parts"
    parts_dir.mkdir(exist_ok=True)
    parts, part_commits = [], {}
    for m in RUN_MODELS:
        pf = parts_dir / f"{m}.pkl"
        if args.reuse_parts and pf.exists():
            saved = pd.read_pickle(pf)
            part, part_commits[m] = saved["part"], saved["git_commit"]
        else:
            part = run_model(m, meta)
            pd.to_pickle({"part": part, "git_commit": meta["git_commit"], "git_dirty": meta["git_dirty"]}, pf)
            part_commits[m] = meta["git_commit"]
        parts.append(part)
    meta["raw_parts_computed_at_commit"] = part_commits
    res = {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}
    r1s, r1ts, r4s, dpt = summarize(res)
    hyp = hypotheses(r1s, r4s, dpt)

    files = {"truncation_curves.csv.gz": res["truncation"], "truncation_summary.csv": r1s, "truncation_summary_per_task.csv": r1ts,
             "fidelity.csv.gz": res["fidelity"], "leave_one_out_deltas.csv.gz": res["deltas"], "leave_one_out_scores.csv": res["scores"],
             "leave_one_out_summary.csv": res["attribution_summary"], "ranked_removal_curves.csv.gz": res["ranked_removal"], "ranked_removal_summary.csv": r4s,
             "degrading_only_removal.csv": dpt, "interaction_check.csv": res["interaction"], "bootstrap_ci_k25_k50.csv": res["ci"],
             "full_embedding_scores.csv": res["baseline"], "criteria_check.csv": hyp}
    if "timing" in res:
        files["timing.csv"] = res["timing"]
    for name, df in files.items():
        save_table(df, out_dir / name, meta)

    figs = {"sts_truncation.png": lambda p: fig_r1(r1s, p), "sts_leave_one_out_scores.png": lambda p: fig_r3(res["scores"], p),
            **{f"sts_ranked_removal_{pp}.png": (lambda p, pp=pp: fig_r4(r1s, r4s, pp, p)) for pp in ("in_sample", "train_ranked", "held_out")},
            "sts_ranked_removal_per_task.png": lambda p: fig_per_task(r1ts, res["ranked_removal"], res["baseline"], p)}
    for name, fn in figs.items():
        fn(fig_dir / name)
        write_metadata(fig_dir / name, meta)
    print(hyp.to_string())
    print(f"git {meta['git_commit']} dirty={meta['git_dirty']}")


if __name__ == "__main__":
    main()
