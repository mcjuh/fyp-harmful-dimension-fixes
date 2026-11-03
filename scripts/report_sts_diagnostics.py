"""Render results/sts/diagnostics/report.md from the diagnostics CSVs (no recomputation).

Usage: python scripts/report_sts_diagnostics.py [--dir results/sts/diagnostics]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fyp.corpora import E_KEYS, R_KEYS
from fyp.models import MODEL_KEYS
from fyp.paths import STS_DIAGNOSTICS
from fyp.protocol import N50, N_DIMS


def rng3(v, d=3, sign=False):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(np.mean(v))} [{f.format(np.percentile(v, 2.5))}, {f.format(np.percentile(v, 97.5))}]"


def short(k):
    return k.replace("_test", "").replace("_train", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=STS_DIAGNOSTICS, help="directory with the sts_diagnostics.py CSVs; report.md is written there")
    D = ap.parse_args().dir
    meta = json.loads((D / "reliability_split_half.csv.meta.json").read_text())
    rd = lambda n: pd.read_csv(D / n)
    a, b, c, dd = rd("reliability_split_half.csv"), rd("ranking_set_agreement.csv"), rd("leave_one_out_bootstrap_summary.csv"), rd("rotation_null.csv")
    r6, r6s, r6p = rd("shared_dimensions.csv"), rd("shared_dimensions_task_sizes.csv"), rd("shared_dimensions_pairwise.csv")
    r7, r7d = rd("outlier_removal.csv"), rd("outlier_dimensions.csv")
    r8, r8b, r8v, r8ci = rd("pca_curves.csv.gz"), rd("pca_baselines.csv"), rd("pca_variance.csv.gz"), rd("pca_ci.csv")
    models = [m for m in MODEL_KEYS if m in set(a["model"])]
    rk = R_KEYS[0]

    L = ["# STS dimension analysis: diagnostics results", "",
         f"Script git `{meta['git_commit']}` (dirty={meta['git_dirty']}), device {meta['device']} ({meta['device_name']}). "
         "Final layer, float64. Supplementary items are marked *supp*. "
         "Δ_t(d) = ρ_t({d}) − ρ_t(∅), so positive means that removing d helps.", "",
         "## split-half reliability: split-half reliability of the train_ranked score (100 random halvings of R)", "",
         "| model | Spearman, mean [2.5, 97.5 pct] | Pearson, mean | sign agreement, mean | Spearman–Brown (full length) |",
         "|---|---|---|---|---|"]
    for m in models:
        x = a[a["model"] == m]
        L.append(f"| {m} | {rng3(x['spearman'])} | {x['pearson'].mean():.3f} | {x['sign_agreement'].mean():.3f} | {x['spearman_brown'].mean():.3f} |")

    L += ["", "## ranking-set agreement: Δ vectors of R (STS-B train, filtered, 3,567 pairs) vs AUX (SICK-R train, 4,439 pairs)", "",
          "Expected |D| overlap under independence = |D_R|·|D_AUX|/768.", "",
          "| model | Pearson | Spearman | sign agreement | \\|D_R\\| | \\|D_AUX\\| | \\|D_R ∩ D_AUX\\| (expected if independent) |",
          "|---|---|---|---|---|---|---|"]
    for m in models:
        x = b[b["model"] == m].iloc[0]
        L.append(f"| {m} | {x['pearson']:.3f} | {x['spearman']:.3f} | {x['sign_agreement']:.3f} | {x['n_degrading_R']} | {x['n_degrading_AUX']} | "
                 f"{x['n_degrading_both']} ({x['n_degrading_R'] * x['n_degrading_AUX'] / N_DIMS:.0f}) |")

    L += ["", "## bootstrap of per-dimension scores: 1,000-resample pair bootstrap of every Δ_t(d): dims whose 95% CI excludes 0", "",
          "Cells: CI entirely above 0 / entirely below 0 (the reference if all Δ were null is ≈ 38 in total, 5% of 768). "
          "Above 0 = removal reliably helps on that task.", "",
          "| model | " + " | ".join([short(rk) + " (R)"] + [short(k) for k in E_KEYS]) + " | in_sample score (*supp*) |",
          "|---|" + "---|" * (len(E_KEYS) + 2)]
    for m in models:
        x = c[c["model"] == m].set_index("unit")
        cells = [f"{x.loc[u, 'n_ci_above_0']} / {x.loc[u, 'n_ci_below_0']}" for u in [rk] + list(E_KEYS) + ["in_sample score (macro over E, supp)"]]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "Median bootstrap SE of Δ vs median |Δ| (ratio > 1: a typical Δ is smaller than its own SE):", "",
          "| model | " + " | ".join([short(rk) + " (R)"] + [short(k) for k in E_KEYS]) + " |", "|---|" + "---|" * (len(E_KEYS) + 1)]
    for m in models:
        x = c[c["model"] == m].set_index("unit")
        L.append(f"| {m} | " + " | ".join(f"{x.loc[u, 'median_boot_se'] / x.loc[u, 'median_abs_delta']:.2f}" for u in [rk] + list(E_KEYS)) + " |")

    L += ["", "## rotation null: random-rotation null (20 Haar rotations; cosines, and so ρ(∅), unchanged)", "",
          "Δ is computed per coordinate in a rotated basis and compared with the real axes. Rotated values are the mean "
          "[min, max] over the 20 rotations. Scores: train_ranked = Δ on R; in_sample = mean of the 7 E-task Δ (in-sample).", "",
          "| model | unit | std(Δ) ×1e5: real | rotated | fraction Δ>0: real | rotated | IQR(Δ) ×1e5 (*supp*): real | rotated | max\\|Δ\\| ×1e4: real | rotated |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    def rot_cells(dm, u, col, scale, d):
        real = dm[(dm["basis"] == "real") & (dm["unit"] == u)][col].iloc[0] * scale
        r = dm[(dm["basis"] == "rotated") & (dm["unit"] == u)][col] * scale
        return f"{real:.{d}f} | {r.mean():.{d}f} [{r.min():.{d}f}, {r.max():.{d}f}]"
    for m in models:
        dm = dd[dd["model"] == m]
        for u in ["train_ranked score", "in_sample score"]:
            L.append(f"| {m} | {u} | {rot_cells(dm, u, 'std', 1e5, 2)} | {rot_cells(dm, u, 'frac_pos', 1, 3)} | "
                     f"{rot_cells(dm, u, 'iqr_supp', 1e5, 2)} | {rot_cells(dm, u, 'max_abs', 1e4, 2)} |")
    L += ["", "Per task (real / rotated mean): std ratio rotated÷real, and fraction Δ>0 real → rotated.", "",
          "| model | " + " | ".join([short(rk) + " (R)"] + [short(k) for k in E_KEYS]) + " |", "|---|" + "---|" * (len(E_KEYS) + 1)]
    for m in models:
        dm = dd[dd["model"] == m]
        cells = []
        for u in [rk] + list(E_KEYS):
            re_ = dm[(dm["basis"] == "real") & (dm["unit"] == u)].iloc[0]
            ro = dm[(dm["basis"] == "rotated") & (dm["unit"] == u)]
            cells.append(f"{ro['std'].mean() / re_['std']:.2f}; {re_['frac_pos']:.2f}→{ro['frac_pos'].mean():.2f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    L += ["", "## Shared degrading dims across the 7 E tasks (IN-SAMPLE, descriptive)", "",
          "D_t = {d : Δ_t(d) > 0}. Observed number of dims degrading in ≥ n tasks vs a null of independent random sets of "
          "size |D_t| (1,000 draws): mean [2.5, 97.5 pct], one-sided p.", "",
          "| model | \\|D_t\\| range | ≥4 tasks | ≥5 | ≥6 | all 7 | 0 tasks (never degrading) |", "|---|---|---|---|---|---|---|"]
    for m in models:
        x = r6[r6["model"] == m].set_index("n_tasks")
        s = r6s[r6s["model"] == m]["n_degrading_t"]
        cell = lambda n: (f"**{int(x.loc[n, 'observed_ge'])}** vs {x.loc[n, 'null_ge_mean']:.1f} "
                          f"[{x.loc[n, 'null_ge_lo']:.0f}, {x.loc[n, 'null_ge_hi']:.0f}] (p={x.loc[n, 'p_ge_upper']:.3f})")
        zero = f"**{int(x.loc[0, 'observed'])}** vs {x.loc[0, 'null_mean']:.1f} [{x.loc[0, 'null_lo']:.0f}, {x.loc[0, 'null_hi']:.0f}]"
        L.append(f"| {m} | {s.min()}–{s.max()} | {cell(4)} | {cell(5)} | {cell(6)} | {cell(7)} | {zero} |")
    L += ["", "*Supp:* agreement of Δ_t between pairs of E tasks (21 pairs).", "",
          "| model | pairwise Spearman: mean [min, max] | pairwise sign agreement: mean [min, max] |", "|---|---|---|"]
    for m in models:
        x = r6p[r6p["model"] == m]
        L.append(f"| {m} | {x['spearman'].mean():.3f} [{x['spearman'].min():.3f}, {x['spearman'].max():.3f}] | "
                 f"{x['sign_agreement'].mean():.3f} [{x['sign_agreement'].min():.3f}, {x['sign_agreement'].max():.3f}] |")

    L += ["", "## Outlier dims (|z| > 3 on the mean final-layer embedding over R's sentences)", "",
          "| model | outlier dims (z) | train_ranked score of each ×1e4 | RP on E, outliers removed | RP, 10 random non-outlier sets: mean [min, max] |",
          "|---|---|---|---|---|"]
    for m in models:
        od = r7d[r7d["model"] == m]
        x = r7[r7["model"] == m]
        if not len(od):
            L.append(f"| {m} | none | – | – (nothing to remove) | – |")
            continue
        rnd = x[x["set"] == "random"]["RP"]
        L.append(f"| {m} | " + ", ".join(f"{int(r['dim'])} ({r['z']:+.1f})" for _, r in od.iterrows()) + " | "
                 + ", ".join(f"{1e4 * v:+.2f}" for v in od["train_ranked_score"]) + f" | {x[x['set'] == 'outliers']['RP'].iloc[0]:.4f} | "
                 f"{rnd.mean():.4f} [{rnd.min():.4f}, {rnd.max():.4f}] |")
    L += ["", "Per task (ratio to the task baseline), outliers removed:", "",
          "| model | " + " | ".join(short(k) for k in E_KEYS) + " |", "|---|" + "---|" * len(E_KEYS)]
    for m in models:
        x = r7[(r7["model"] == m) & (r7["set"] == "outliers")]
        if x["n_removed"].iloc[0] == 0:
            continue
        L.append(f"| {m} | " + " | ".join(f"{x[f'ratio_{k}'].iloc[0]:.4f}" for k in E_KEYS) + " |")

    L += ["", "## PCA (fit on 20k NLI sentences, never on E) keeping 384 components vs random 50% truncation", "",
          "PCA centres the data, so its K=0 point is the centred embedding (asserted). RP is given against the raw "
          "baseline (the paper's comparison) and the centred one. *Supp:* random 50% truncation of the centred embeddings "
          "(RP vs the centred baseline) separates the effect of centring from PCA's choice of directions.", "",
          "| model | variance kept by 384 PCs | centred full / raw full | PCA-384 RP (raw base) | PCA-384 RP (centred base) | random-50% RP (raw, 100 seeds) | random-50% on centred (*supp*) | PCA − random, paired bootstrap 95% CI (*supp*) |",
          "|---|---|---|---|---|---|---|---|"]
    for m in models:
        bb = r8b[r8b["model"] == m]
        bm, bc = bb["rho_raw"].mean(), bb["rho_centred"].mean()
        x = r8[r8["model"] == m]
        pca = x[(x["method"] == "pca") & (x["n"] == N50)]["rho"].mean()
        rnd = x[x["method"] == "random"].groupby("seed")["rho"].mean().mean()
        rc = x[x["method"] == "random_centred_supp"].groupby("seed")["rho"].mean().mean()
        v = r8v[(r8v["model"] == m) & (r8v["n_components"] == N_DIMS - N50)]["cum_explained_variance"].iloc[0]
        ci = r8ci[(r8ci["model"] == m)].set_index("quantity").loc["pca50_RP_raw_minus_random50"]
        L.append(f"| {m} | {v:.3f} | {bc / bm:.4f} | {pca / bm:.4f} | {pca / bc:.4f} | {rnd / bm:.4f} | {rc / bc:.4f} | "
                 f"{ci['boot_mean']:+.4f} [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}] |")

    L += ["", "Figures: `figures/sts_reliability.png`, `sts_rotation_null.png`, "
          "`sts_shared_dimensions.png`, `sts_outliers.png`, `sts_pca.png`. "
          "Raw tables: `results/sts/diagnostics/*.csv(.gz)`."]
    (D / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {D / 'report.md'}")


if __name__ == "__main__":
    main()
