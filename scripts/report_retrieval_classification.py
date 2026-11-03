"""Suite report: summaries, hypothesis checks, figures and results/retrieval_classification/report.md from results/retrieval_classification/{retrieval,classification}/.

Reads the CSVs written by scripts/retrieval_classification_truncation_and_ranked_removal.py (final test-split run) and scripts/select_layer.py. Nothing is recomputed here.
`--transforms` instead summarises the corrections-before-the-cut run against the raw curves into results/transformations/.
Usage: python scripts/report_retrieval_classification.py [--root results/retrieval_classification] [--fig-dir figures]
       python scripts/report_retrieval_classification.py --transforms
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fyp.models import ALL_MODEL_KEYS as MODELS
from fyp.paths import FIGURES, MRL_PAIR, STS_TRUNCATION, SUITE, TRANSFORMS, SUITE_DIR
from fyp.plotting import BAND, C, TEXT_2, group_title, model_grid
from fyp.protocol import CONTRASTIVE_MODELS as CONTRASTIVE, RANKING_POWER_K, K_GRID, N_DIMS, N_GRID, n_remove
from fyp.reporting import md_table
from fyp.runtime import run_metadata, setup_device, write_metadata
from fyp.transforms import TRANSFORMS as TRANSFORM_NAMES

SUITES = {"ret": ("retrieval (NanoBEIR, 13 datasets, nDCG@10)", "nDCG@10"),
          "cls": ("classification (MTEB, 12 tasks, probe accuracy)", "accuracy")}


def md(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    return md_table(df, fmt)


def load(root: Path, suite: str):
    d = root / SUITE_DIR[suite]
    f = lambda n: pd.read_csv(d / n)
    return {"baseline": f("full_embedding_scores.csv"), "truncation": f("truncation_curves.csv.gz"), "fidelity": f("fidelity.csv.gz"), "attribution_summary": f("leave_one_out_summary.csv"),
            "scores": f("leave_one_out_scores.csv.gz"), "ranked_removal": f("ranked_removal_curves.csv.gz"), "shared": f("shared_dimensions.csv"), "outliers": f("outlier_removal.csv"),
            "pca": f("pca_curves.csv.gz"), "pca_ci": f("pca_ci.csv"), "ci": f("bootstrap_ci_k25_k50.csv")}


def rp_tables(res):
    base = res["baseline"]
    bm = base.groupby("model")["metric_full"].mean()
    # Truncation: RP per (model, method, seed, n)
    t1 = res["truncation"].groupby(["model", "method", "seed", "n"])["metric"].mean().reset_index()
    t1["RP"] = t1["metric"] / t1["model"].map(bm)
    rows = []
    for (m, method, n), g in t1.groupby(["model", "method", "n"]):
        v = g.sort_values("seed")["RP"].to_numpy()
        rows.append({"model": m, "method": method, "n": n, "K": 100 * n / N_DIMS, "RP_mean": v.mean(), "RP_std": v.std(ddof=1) if len(v) > 1 else 0.0,
                     "RP_std_first10": v[:10].std(ddof=1) if len(v) > 1 else 0.0, "band_lo": np.percentile(v, 2.5), "band_hi": np.percentile(v, 97.5),
                     "n_seeds": len(v)})
    t1s = pd.DataFrame(rows)
    # per-task ratio at K = 50 (random mean) for the per-task table
    b = base.set_index(["model", "task"])["metric_full"]
    pt = res["truncation"][(res["truncation"]["n"] == n_remove(50))].copy()
    pt["ratio"] = pt["metric"] / pt.apply(lambda r: b[(r["model"], r["task"])], axis=1)
    pt = pt.groupby(["model", "method", "task"])["ratio"].mean().reset_index()
    # Ranked removal: in_sample / held_out macro over the tasks present at each n
    t4 = res["ranked_removal"]
    agg = []
    ntask = base.groupby("model")["task"].nunique()
    for (m, p), d in t4.groupby(["model", "protocol"]):
        d = d if p == "in_sample" else d[d["n"].isin(N_GRID)]
        g = d.groupby(["curve", "seed", "n"])["metric"]
        mean, cnt = g.mean(), g.size()
        mean = mean[cnt == ntask[m]].reset_index()
        mean["RP"] = mean["metric"] / bm[m]
        mean["model"], mean["protocol"] = m, p
        agg.append(mean)
    t4a = pd.concat(agg)
    rows = []
    for (m, p, curve, n), g in t4a.groupby(["model", "protocol", "curve", "n"]):
        v = g["RP"].to_numpy()
        rows.append({"model": m, "protocol": p, "curve": curve, "n": n, "K": 100 * n / N_DIMS, "RP": v.mean(),
                     "band_lo": np.percentile(v, 2.5) if len(v) > 1 else np.nan, "band_hi": np.percentile(v, 97.5) if len(v) > 1 else np.nan})
    return t1s, pt, pd.DataFrame(rows)


def hypotheses(t1s, t4s, models):
    rows = []
    rand = t1s[t1s["method"] == "random"].set_index(["model", "n"])
    for m in models:
        h1 = rand.loc[(m, n_remove(50)), "RP_mean"]
        row = {"model": m, "group": "contrastive" if m in CONTRASTIVE else "raw", "random50_RP": h1,
               "random_removal_robust": bool(h1 >= 0.90) if m in CONTRASTIVE else None}
        for p in ("in_sample", "held_out"):
            s = t4s[(t4s["model"] == m) & (t4s["protocol"] == p)].set_index(["curve", "n"])
            above, below, ever = [], [], False
            for K in K_GRID[1:]:
                n = n_remove(K)
                hi, lo = rand.loc[(m, n), "band_hi"], rand.loc[(m, n), "band_lo"]
                bf, wf = s.loc[("best_first", n), "RP"], s.loc[("worst_first", n), "RP"]
                ever |= bool(bf > hi)
                if K in RANKING_POWER_K:
                    above.append(bool(bf > hi)); below.append(bool(wf < lo))
            row[f"{p}_best_above_band_all_K10_50"] = all(above)
            row[f"{p}_worst_below_band_all_K10_50"] = all(below)
            row[f"{p}_best_above_band_count"] = f"{sum(above)}/{len(above)}"
            row[f"{p}_worst_below_band_count"] = f"{sum(below)}/{len(below)}"
            row[f"{p}_best_ever_above_band"] = ever
            row[f"{p}_best_first_RP50"] = s.loc[("best_first", n_remove(50)), "RP"]
            row[f"{p}_worst_first_RP50"] = s.loc[("worst_first", n_remove(50)), "RP"]
        row["heldout_ranking_power"] = row["held_out_best_above_band_all_K10_50"] and row["held_out_worst_below_band_all_K10_50"]
        row["no_heldout_power"] = not row["held_out_best_ever_above_band"]
        row["in_sample_only_gain"] = row["in_sample_best_ever_above_band"] and not row["held_out_best_ever_above_band"]
        rows.append(row)
    return pd.DataFrame(rows)


def _grid(models, title, ylabel, path, draw, ncol=4, xlabel="K = % of dimensions removed"):
    model_grid(models, draw, title=title, xlabel=xlabel, ylabel=ylabel, path=path, ncol=ncol, title_fn=group_title)


def fig_t1(t1s, suite, models, path):
    def draw(ax, m):
        d = t1s[(t1s["model"] == m) & t1s["n"].isin(N_GRID)]
        r = d[d["method"] == "random"].sort_values("n")
        ax.fill_between(r["K"], r["band_lo"], r["band_hi"], color=BAND, alpha=0.6, lw=0, label="random-K 95% band")
        ax.plot(r["K"], r["RP_mean"], color=TEXT_2, lw=1.5, label="random-K mean")
        for meth, c in (("last", C["blue"]), ("first", C["orange"])):
            x = d[d["method"] == meth].sort_values("n")
            ax.plot(x["K"], x["RP_mean"], color=c, lw=2, label=f"{meth}-K")
        ax.axhline(0.9, color=TEXT_2, lw=0.8, ls=":")
    _grid(models, f"[{suite}] truncation curves (dotted = 0.90)", "relative performance (RP)", path, draw)


def fig_t4(t1s, t4s, protocol, suite, models, path):
    def draw(ax, m):
        r = t1s[(t1s["model"] == m) & (t1s["method"] == "random") & t1s["n"].isin(N_GRID)].sort_values("n")
        ax.fill_between(r["K"], r["band_lo"], r["band_hi"], color=BAND, alpha=0.6, lw=0, label="random-K 95% band")
        s = t4s[(t4s["model"] == m) & (t4s["protocol"] == protocol)]
        for cv, c, lbl in (("best_first", C["blue"], "best-first"), ("worst_first", C["orange"], "worst-first"),
                           ("degrading_only_ranked", C["aqua"], "degrading-only, ranked"), ("informative_only_ranked", C["yellow"], "informative-only, ranked")):
            x = s[s["curve"] == cv].sort_values("n")
            x = x[x["n"].isin(N_GRID)] if protocol == "held_out" else x
            ax.plot(x["K"], x["RP"], color=c, lw=2 if "first" in cv else 1.5, ls="-" if "first" in cv else "--", label=lbl)
    unit = "dataset" if suite == "ret" else "task"
    lab = {"in_sample": f"in_sample: ranked and evaluated on the same {unit}s (IN-SAMPLE, the paper's design)",
           "held_out": f"held_out: leave-one-{unit}-out (held-out)"}[protocol]
    _grid(models, f"[{suite}] ranked removal: {lab}", "relative performance (RP)", path, draw)


def fig_t2(res, suite, models, path):
    f = res["fidelity"].groupby(["model", "method", "seed", "n"])["fidelity"].mean().reset_index()

    def draw(ax, m):
        d = f[f["model"] == m]
        for meth, c in (("last", C["blue"]), ("first", C["orange"]), ("random", C["aqua"])):
            x = d[d["method"] == meth].groupby("n")["fidelity"].mean().reset_index()
            ax.plot(100 * x["n"] / N_DIMS, x["fidelity"], color=c, lw=2, label=f"{meth}-K")
    _grid(models, f"[{suite}] label-free fidelity to the full embedding", "fidelity", path, draw, ncol=3)


def fig_t5(res, suite, models, path):
    def draw(ax, m):
        d = res["shared"][(res["shared"]["model"] == m) & (res["shared"]["n_tasks"] >= 1)]
        ax.fill_between(d["n_tasks"], d["null_ge_lo"], d["null_ge_hi"], color=BAND, alpha=0.6, lw=0, label="random sets, 95%")
        ax.plot(d["n_tasks"], d["observed_ge"], color=C["blue"], lw=2, marker="o", ms=3, label="observed (IN-SAMPLE)")
        ax.set_yscale("log")
    _grid(models, f"[{suite}] dims degrading in at least n tasks", "# dims", path, draw, ncol=2,
          xlabel="n (number of datasets/tasks)")


def fig_t7(res, t1s, suite, models, path):
    def draw(ax, m):
        d = res["pca"][(res["pca"]["model"] == m) & (res["pca"]["method"] == "pca")]
        bm = res["baseline"][res["baseline"]["model"] == m]["metric_full"].mean()
        x = d.groupby("n")["metric"].mean() / bm
        ax.plot(100 * x.index / N_DIMS, x.values, color=C["blue"], lw=2, label="PCA (fit on NLI), RP vs raw baseline")
        r = t1s[(t1s["model"] == m) & (t1s["method"] == "random") & t1s["n"].isin(N_GRID)].sort_values("n")
        ax.fill_between(r["K"], r["band_lo"], r["band_hi"], color=BAND, alpha=0.6, lw=0, label="random-K 95% band")
        ax.plot(r["K"], r["RP_mean"], color=TEXT_2, lw=1.5, label="random-K mean")
    _grid(models, f"[{suite}] PCA truncation vs random truncation", "RP vs the raw full embedding", path, draw, ncol=3)


def raw_curves(suite: str) -> pd.DataFrame:
    """Curves of the untransformed embeddings with the same columns as the transforms run (same seeds): the main six plus the Matryoshka pair."""
    if suite == "sts":
        dfs = [pd.read_csv(d / "truncation_curves.csv.gz").rename(columns={"rho": "metric"}) for d in (STS_TRUNCATION, MRL_PAIR / "sts")]
    else:
        dfs = [pd.read_csv(d / SUITE_DIR[suite] / "truncation_curves.csv.gz") for d in (SUITE, MRL_PAIR)]
    return pd.concat(dfs, ignore_index=True).assign(transform="raw")


def transforms_summary(suite: str, root: Path) -> pd.DataFrame:
    """Per model and transform (raw included): full-width score, score and RP at K = 50 for random / last / first cuts, and the score
    at K = 50 relative to the raw full-width score. Macro-average over tasks; random = mean over seeds."""
    d = pd.concat([raw_curves(suite), pd.read_csv(root / SUITE_DIR[suite] / "truncation_curves.csv.gz")], ignore_index=True)
    d = d[d["n"].isin([0, n_remove(50)])]
    d = d[d["model"].isin(set(pd.read_csv(root / SUITE_DIR[suite] / "truncation_curves.csv.gz")["model"]))]
    m = d.groupby(["model", "transform", "method", "seed", "n"])["metric"].mean().reset_index()
    full = m[(m["method"] == "last") & (m["n"] == 0)].set_index(["model", "transform"])["metric"].rename("full")
    raw_full = full.xs("raw", level="transform").rename("raw_full")
    rows = []
    for (mod, tr, method), g in m[m["n"] == n_remove(50)].groupby(["model", "transform", "method"]):
        v = g["metric"].to_numpy()
        rows.append({"suite": suite, "model": mod, "transform": tr, "cut": method, "full": full[(mod, tr)], "score_K50": v.mean(),
                     "RP_K50": v.mean() / full[(mod, tr)], "RP_std": v.std(ddof=1) if len(v) > 1 else 0.0,
                     "vs_raw_full": v.mean() / raw_full[mod]})
    out = pd.DataFrame(rows)
    out["transform"] = pd.Categorical(out["transform"], ["raw"] + TRANSFORM_NAMES, ordered=True)
    return out.sort_values(["model", "transform", "cut"]).reset_index(drop=True)


def transforms_main(root: Path):
    dev = setup_device(False, "cpu")
    meta = run_metadata(dev, script="scripts/report_retrieval_classification.py --transforms", protocol_id="transformations", extra={"root": str(root)})
    md_parts = ["# Corrections before the cut\n",
                "Generated by `scripts/report_retrieval_classification.py --transforms` from `results/transformations/{sts,retrieval,classification}/` and the raw curves of "
                "`results/sts/truncation_and_ranked_removal`, `results/retrieval_classification`, `results/matryoshka`. `RP_K50` is against the transformed full width; `vs_raw_full` is the "
                "score at K = 50 over the raw full-width score.\n"]
    for suite in ("sts", "ret", "cls"):
        if not (root / SUITE_DIR[suite] / "truncation_curves.csv.gz").exists():
            md_parts.append(f"\n## {suite}\n\n(not run)\n")
            continue
        tab = transforms_summary(suite, root)
        tab.to_csv(root / SUITE_DIR[suite] / "summary.csv", index=False)
        write_metadata(root / SUITE_DIR[suite] / "summary.csv", meta)
        rnd = tab[tab["cut"] == "random"]
        md_parts.append(f"\n## {suite}\n\n### Random-50 % score relative to the raw full-width score\n\n"
                        f"{md(rnd.pivot(index='model', columns='transform', values='vs_raw_full').reset_index(), '{:.3f}')}\n")
        md_parts.append(f"\n### Full-width score after the transform\n\n"
                        f"{md(rnd.pivot(index='model', columns='transform', values='full').reset_index(), '{:.3f}')}\n")
        for cut in ("random", "last", "first"):
            wide = tab[tab["cut"] == cut].pivot(index="model", columns="transform", values="RP_K50")
            md_parts.append(f"\n### RP at K = 50 ({cut} cut), against the transformed full width\n\n{md(wide.reset_index(), '{:.3f}')}\n")
    (root / "report.md").write_text("\n".join(md_parts), encoding="utf-8")
    write_metadata(root / "report.md", meta)
    print(f"wrote {root / 'report.md'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=SUITE)
    p.add_argument("--fig-dir", type=Path, default=FIGURES)
    p.add_argument("--transforms", action="store_true", help="summarise results/transformations instead of the suite")
    args = p.parse_args()
    if args.transforms:
        return transforms_main(TRANSFORMS)
    dev = setup_device(False, "cpu")
    meta = run_metadata(dev, script="scripts/report_retrieval_classification.py", protocol_id="retrieval_classification", extra={"root": str(args.root)})
    md_parts = ["# Suite report: full classification + retrieval suite\n",
                "Generated by `scripts/report_retrieval_classification.py` from `results/retrieval_classification/{retrieval,classification}/`. in_sample is **in-sample** (the paper's design); held_out is "
                "leave-one-dataset/task-out.\n"]
    for suite, (title, mname) in SUITES.items():
        if not (args.root / SUITE_DIR[suite] / "truncation_curves.csv.gz").exists():
            md_parts.append(f"\n## {title}\n\n(not run)\n")
            continue
        res = load(args.root, suite)
        models = [m for m in MODELS if m in set(res["baseline"]["model"])]
        t1s, pt, t4s = rp_tables(res)
        hyp = hypotheses(t1s, t4s, models)
        for name, df in (("truncation_summary", t1s), ("truncation_per_task_k50", pt), ("ranked_removal_summary", t4s), ("criteria_check", hyp)):
            df.to_csv(args.root / SUITE_DIR[suite] / f"{name}.csv", index=False)
            write_metadata(args.root / SUITE_DIR[suite] / f"{name}.csv", meta)
        figs = {f"{suite}_truncation.png": lambda pth: fig_t1(t1s, suite, models, pth),
                f"{suite}_fidelity.png": lambda pth: fig_t2(res, suite, models, pth),
                f"{suite}_ranked_removal_in_sample.png": lambda pth: fig_t4(t1s, t4s, "in_sample", suite, models, pth),
                f"{suite}_ranked_removal_held_out.png": lambda pth: fig_t4(t1s, t4s, "held_out", suite, models, pth),
                f"{suite}_shared_dimensions.png": lambda pth: fig_t5(res, suite, models, pth),
                f"{suite}_pca.png": lambda pth: fig_t7(res, t1s, suite, models, pth)}
        for fn, fnc in figs.items():
            fnc(args.fig_dir / fn)
            write_metadata(args.fig_dir / fn, meta)
        base = res["baseline"].groupby("model")["metric_full"].mean().rename(f"full {mname} (macro)").reset_index()
        rand50 = t1s[(t1s["method"] == "random") & (t1s["n"] == n_remove(50))][["model", "RP_mean", "RP_std", "RP_std_first10", "band_lo", "band_hi"]]
        last50 = t1s[(t1s["method"] == "last") & (t1s["n"] == n_remove(50))][["model", "RP_mean"]].rename(columns={"RP_mean": "last-50 RP"})
        first50 = t1s[(t1s["method"] == "first") & (t1s["n"] == n_remove(50))][["model", "RP_mean"]].rename(columns={"RP_mean": "first-50 RP"})
        head = base.merge(rand50.rename(columns={"RP_mean": "random-50 RP"}), on="model").merge(last50, on="model").merge(first50, on="model")
        md_parts.append(f"\n## {title}\n\n### Truncation: relative performance at K = 50% (final layer)\n\n{md(head)}\n")
        md_parts.append(f"\n![truncation]({(Path('../../figures') / f'{suite}_truncation.png').as_posix()})\n")
        ci = res["ci"][res["ci"]["K"] == 50][["model", "quantity", "boot_mean", "ci_lo", "ci_hi"]]
        md_parts.append(f"\n### Paired bootstrap at K = 50% (95% CI over {'queries' if suite == 'ret' else 'test items'}, 1000 resamples)\n\n{md(ci)}\n")
        md_parts.append(f"\n### Hypotheses\n\n{md(hyp)}\n")
        md_parts.append(f"\n![ranked removal, in-sample]({(Path('../../figures') / f'{suite}_ranked_removal_in_sample.png').as_posix()})\n\n![ranked removal, held-out]({(Path('../../figures') / f'{suite}_ranked_removal_held_out.png').as_posix()})\n")
        s3 = res["attribution_summary"][res["attribution_summary"]["protocol"] == "in_sample"][["model", "n_degrading", "n_informative", "n_neutral", "degrading_frac_block_min", "degrading_frac_block_max",
                                                                         "lag1_sign_autocorr", "lag1_perm_p_two_sided"]]
        md_parts.append(f"\n### Per-dimension attribution (in-sample; the count of degrading dims is never evidence on its own)\n\n{md(s3)}\n")
        s5 = res["shared"][res["shared"]["n_tasks"] >= 2][["model", "n_tasks", "observed_ge", "null_ge_mean", "null_ge_hi", "p_ge_upper"]]
        md_parts.append(f"\n### Shared degrading dims (in-sample, descriptive)\n\n{md(s5)}\n")
        s6 = res["outliers"].groupby(["model", "set"]).agg(n_removed=("n_removed", "first"), RP=("RP", "mean")).reset_index()
        md_parts.append(f"\n### Outlier dimensions\n\n{md(s6)}\n")
        s7 = res["pca_ci"][["model", "quantity", "boot_mean", "ci_lo", "ci_hi"]]
        md_parts.append(f"\n### PCA-384 vs random 50%\n\n{md(s7)}\n\n![PCA]({(Path('../../figures') / f'{suite}_pca.png').as_posix()})\n")
    lt = args.root / "best_layer_table.csv"
    if lt.exists():
        tab = pd.read_csv(lt)
        cols = ["model", "final_layer", "l_star_sts", "l_star_ret", "l_star_cls", "sts_is_final", "ret_is_final", "cls_is_final"]
        md_parts.append(f"\n## l* per task type (dev-selected)\n\n{md(tab[cols])}\n\n![l*]({Path('../../figures/retrieval_classification_best_layer.png').as_posix()})\n")
    (args.root / "report.md").write_text("\n".join(md_parts), encoding="utf-8")
    write_metadata(args.root / "report.md", meta)
    print(f"wrote {args.root / 'report.md'}")


if __name__ == "__main__":
    main()
