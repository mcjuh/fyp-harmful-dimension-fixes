"""Render results/sts/truncation_and_ranked_removal/report.md from the sts_truncation_and_ranked_removal.py CSVs (no recomputation).

Usage: python scripts/report_sts.py [--dir results/sts/truncation_and_ranked_removal]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from fyp.models import ALL_MODEL_KEYS as MODEL_KEYS
from fyp.paths import STS_TRUNCATION
from fyp.protocol import CONTRASTIVE_MODELS as CONTRASTIVE, n_remove


def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=STS_TRUNCATION, help="directory with the sts_truncation_and_ranked_removal.py CSVs; report.md is written there")
    S4 = ap.parse_args().dir
    meta = json.loads((S4 / "criteria_check.csv.meta.json").read_text())
    base = pd.read_csv(S4 / "full_embedding_scores.csv")
    r1 = pd.read_csv(S4 / "truncation_summary.csv")
    r2 = pd.read_csv(S4 / "fidelity.csv.gz")
    r3 = pd.read_csv(S4 / "leave_one_out_summary.csv")
    r4 = pd.read_csv(S4 / "ranked_removal_summary.csv")
    dpt = pd.read_csv(S4 / "degrading_only_removal.csv")
    inter = pd.read_csv(S4 / "interaction_check.csv")
    ci = pd.read_csv(S4 / "bootstrap_ci_k25_k50.csv")
    hyp = pd.read_csv(S4 / "criteria_check.csv").set_index("model")
    models = [m for m in MODEL_KEYS if m in hyp.index]
    n10, n25, n50 = n_remove(10), n_remove(25), n_remove(50)

    L = ["# STS ablation: truncation and ranked removal results", "",
         f"Script git `{meta['git_commit']}` (dirty={meta['git_dirty']}); raw parts computed at "
         f"{sorted(set(meta.get('raw_parts_computed_at_commit', {}).values()))}. Final layer, float64 closed form, "
         "all numbers on **E** unless stated. RP = relative performance = mean_E ρ(S_K) / mean_E ρ(∅). "
         "Random band = 2.5–97.5 percentile of RP over 100 random-K draws. "
         "**in_sample is in-sample (ranked and evaluated on E) and optimistic by construction.**", "",
         "## Baselines (final layer, ρ ×100)", "",
         "| model | " + " | ".join(base["task"].unique()) + " | mean |", "|---|" + "---|" * (base["task"].nunique() + 1)]
    for m in models:
        b = base[base["model"] == m]
        L.append(f"| {m} | " + " | ".join(f"{100 * v:.2f}" for v in b["rho_full"]) + f" | {100 * b['rho_full'].mean():.2f} |")

    # ---------------- truncation
    L += ["", "## Truncation (label-free) and robustness", "",
          "| model | group | random-50% RP mean | std (100 seeds) | std (first 10) | 95% band | last-50% RP | first-50% RP | random-25% RP | robust (≥ 0.90) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        g = r1[(r1["model"] == m)].set_index(["method", "n"])
        rnd = g.loc[("random", n50)]
        h1 = hyp.loc[m, "random_removal_robust"]
        verdict = ("**supported**" if h1 is True or h1 == "True" else "**not supported**") if m in CONTRASTIVE else "n/a (raw)"
        L.append(f"| {m} | {hyp.loc[m, 'group']} | {f3(rnd['RP_mean'])} | {rnd['RP_std100']:.4f} | {rnd['RP_std_first10']:.4f} | "
                 f"[{f3(rnd['band_lo'])}, {f3(rnd['band_hi'])}] | {f3(g.loc[('last', n50), 'RP_mean'])} | "
                 f"{f3(g.loc[('first', n50), 'RP_mean'])} | {f3(g.loc[('random', n25), 'RP_mean'])} | {verdict} |")

    # ---------------- fidelity
    L += ["", "## Label-free fidelity, Spearman(full cos, truncated cos), mean over E tasks", "",
          "| model | random K=25% | random K=50% | random K=90% | last K=50% | first K=50% |", "|---|---|---|---|---|---|"]
    for m in models:
        d = r2[r2["model"] == m]
        fr = lambda meth, K: d[(d["method"] == meth) & (d["K"] == K)]["fidelity"].mean()
        L.append(f"| {m} | {f3(fr('random', 25))} | {f3(fr('random', 50))} | {f3(fr('random', 90))} | {f3(fr('last', 50))} | {f3(fr('first', 50))} |")

    # ---------------- attribution
    L += ["", "## Per-dimension attribution", "",
          "Counts are shown for completeness only: pure noise also splits roughly half and half. "
          "Uniformity: fraction of D per block of 64 indices (min–max over 12 blocks); lag-1 autocorrelation of sign(score) along the index vs a 1,000-permutation null.", "",
          "| model | protocol | ranking pairs | \\|D\\| | \\|I\\| | neutral | score std ×1e5 | D-frac per block (min–max) | lag-1 sign autocorr (p) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        for p in ("in_sample", "train_ranked"):
            r = r3[(r3["model"] == m) & (r3["protocol"] == p)].iloc[0]
            L.append(f"| {m} | {p}{' (in-sample)' if p == 'in_sample' else ''} | {r['ranking_pairs']:,} | {r['n_degrading']} | {r['n_informative']} | {r['n_neutral']} | "
                     f"{1e5 * r['score_std']:.2f} | {r['degrading_frac_block_min']:.2f}–{r['degrading_frac_block_max']:.2f} | "
                     f"{r['lag1_sign_autocorr']:+.3f} ({r['lag1_perm_p_two_sided']:.3f}) |")
        r = r3[(r3["model"] == m) & (r3["protocol"] == "held_out")]
        L.append(f"| {m} | held_out (7 folds) | {r['ranking_pairs'].min():,}–{r['ranking_pairs'].max():,} | {r['n_degrading'].min()}–{r['n_degrading'].max()} | "
                 f"{r['n_informative'].min()}–{r['n_informative'].max()} | {r['n_neutral'].max()} | {1e5 * r['score_std'].mean():.2f} (mean) | "
                 f"{r['degrading_frac_block_min'].min():.2f}–{r['degrading_frac_block_max'].max():.2f} | min p = {r['lag1_perm_p_two_sided'].min():.3f} |")

    # ---------------- ranked removal
    L += ["", "## Ranked removal curves against the random band", "",
          "Best-first = remove the highest-score (most 'degrading') dims first; worst-first = lowest first. "
          "'↑' / '↓' = above / below the truncation random 95% band at that K. The ranking-power criterion needs best-first ↑ **and** worst-first ↓ at every K ∈ {10,…,50} under train_ranked and held_out.", "",
          "| model | protocol | best-first RP @10/25/50% | worst-first RP @10/25/50% | random mean @10/25/50% | best ↑ (K10–50) | worst ↓ (K10–50) | best ever ↑ (any K) |",
          "|---|---|---|---|---|---|---|---|"]
    rnd = r1[r1["method"] == "random"].set_index(["model", "n"])
    for m in models:
        for p in ("in_sample", "train_ranked", "held_out"):
            s = r4[(r4["model"] == m) & (r4["protocol"] == p)].set_index(["curve", "n"])

            def cell(curve):
                out = []
                for n in (n10, n25, n50):
                    v = s.loc[(curve, n), "RP"]
                    mark = "↑" if v > rnd.loc[(m, n), "band_hi"] else ("↓" if v < rnd.loc[(m, n), "band_lo"] else "")
                    out.append(f"{v:.3f}{mark}")
                return " / ".join(out)
            L.append(f"| {m} | {p}{' (in-sample)' if p == 'in_sample' else ''} | {cell('best_first')} | {cell('worst_first')} | "
                     + " / ".join(f"{rnd.loc[(m, n), 'RP_mean']:.3f}" for n in (n10, n25, n50))
                     + f" | {hyp.loc[m, f'{p}_best_above_band_count_K10_50']} | {hyp.loc[m, f'{p}_worst_below_band_count_K10_50']} | "
                       f"{'yes' if hyp.loc[m, f'{p}_best_ever_above_band'] else 'no'} |")

    L += ["", "### degrading-only / informative-only removal", "",
          "At K = \\|D\\| the degrading-only curve has removed all of D, so ranked and random order agree by construction. "
          "Whether the *order within D* matters is read at K = 10% (77 dims, which is below every \\|D\\|): ranked vs the mean and 95% band over 100 random orders of D.", "",
          "| model | protocol | \\|D\\| | RP, all of D removed | degrading-only @10%: ranked | degrading-only @10%: random-order mean [band] | RP, all of I removed |",
          "|---|---|---|---|---|---|---|"]
    for m in models:
        for p in ("in_sample", "train_ranked", "held_out"):
            s = r4[(r4["model"] == m) & (r4["protocol"] == p)]
            get = lambda c, n: s[(s["curve"] == c) & (s["n"] == n)]
            r10, q10 = get("degrading_only_ranked", n10), get("degrading_only_random", n10)
            rand10 = (f"{q10['RP'].iloc[0]:.3f} [{q10['band_lo'].iloc[0]:.3f}, {q10['band_hi'].iloc[0]:.3f}]" if len(q10) else "–")
            if p == "held_out":
                d = dpt[(dpt["model"] == m) & (dpt["protocol"] == "held_out")]
                L.append(f"| {m} | held_out | {d['n_degrading'].min()}–{d['n_degrading'].max()} (per fold) | {f3(d['rho_mean_at_n_degrading'].mean() / d['base_mean'].mean())} | "
                         f"{f3(r10['RP'].mean()) if len(r10) else '–'} | {rand10} | (per fold) |")
                continue
            nD = int(r3[(r3["model"] == m) & (r3["protocol"] == p)]["n_degrading"].iloc[0])
            nI = int(r3[(r3["model"] == m) & (r3["protocol"] == p)]["n_informative"].iloc[0])
            L.append(f"| {m} | {p} | {nD} | {f3(get('degrading_only_ranked', nD)['RP'].mean())} | {f3(r10['RP'].mean())} | {rand10} | "
                     f"{f3(get('informative_only_ranked', nI)['RP'].mean())} |")

    # ---------------- Interaction
    L += ["", "### Interaction check: Σ single-dim score vs actual joint change, removing the whole of D best-first", "",
          "| model | protocol | \\|D\\| | predicted Δ (ranking data) | actual Δ (ranking data) | predicted Δ (E, in-sample Δs) | actual Δ (E) |",
          "|---|---|---|---|---|---|---|"]
    for m in models:
        for p in ("in_sample", "train_ranked"):
            d = inter[(inter["model"] == m) & (inter["protocol"] == p)]
            r = d[d["n"] == d["n"].max()].iloc[0]
            L.append(f"| {m} | {p} | {int(r['n'])} | {r['predicted_ranking']:+.4f} | {r['actual_ranking']:+.4f} | {r['predicted_E']:+.4f} | {r['actual_E']:+.4f} |")

    # ---------------- Headline CIs
    L += ["", "## Headline paired-bootstrap 95% CIs (1,000 pair-resamples per task)", "",
          "| model | K | random mean RP | last-K RP | train_ranked best − random | train_ranked worst − random | held_out best − random | held_out worst − random |",
          "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for K in (25, 50):
            c = ci[(ci["model"] == m) & (ci["K"] == K)].set_index("quantity")
            fmt = lambda q, d=3: f"{c.loc[q, 'boot_mean']:+.{d}f} [{c.loc[q, 'ci_lo']:+.{d}f}, {c.loc[q, 'ci_hi']:+.{d}f}]"
            L.append(f"| {m} | {K}% | {c.loc['random_mean', 'boot_mean']:.3f} [{c.loc['random_mean', 'ci_lo']:.3f}, {c.loc['random_mean', 'ci_hi']:.3f}] | "
                     f"{c.loc['last', 'boot_mean']:.3f} [{c.loc['last', 'ci_lo']:.3f}, {c.loc['last', 'ci_hi']:.3f}] | "
                     f"{fmt('train_ranked_best_minus_random_mean')} | {fmt('train_ranked_worst_minus_random_mean')} | {fmt('held_out_best_minus_random_mean')} | {fmt('held_out_worst_minus_random_mean')} |")

    # ---------------- Pre-registered readings
    L += ["", "## Pre-registered readings", "",
          "| model | random-removal robust | ranking power (train_ranked & held_out) | no held-out power | in-sample-only gain pattern (in_sample ↑, train_ranked/held_out never ↑) | degrading-only RP at the number of degrading dims, train_ranked / held_out |",
          "|---|---|---|---|---|---|"]
    for m in models:
        h = hyp.loc[m]
        h1 = ("supported" if str(h["random_removal_robust"]) == "True" else "not supported") if m in CONTRASTIVE else "n/a"
        L.append(f"| {m} | {h1} | {'supported' if h['heldout_ranking_power'] else 'not supported'} | "
                 f"{'**TRIGGERED**' if h['no_heldout_power'] else 'not triggered'} | {'yes' if h['in_sample_only_gain'] else 'no'} | "
                 f"{h['train_ranked_degrading_only_RP_at_n_degrading']:.3f} / {h['held_out_degrading_only_RP_at_n_degrading']:.3f} |")
    L += ["", "Figures: `figures/sts_truncation.png`, `sts_leave_one_out_scores.png`, `sts_ranked_removal_in_sample.png`, "
          "`sts_ranked_removal_train_ranked.png`, `sts_ranked_removal_held_out.png`, `sts_ranked_removal_per_task.png`. Raw tables: `results/sts/truncation_and_ranked_removal/*.csv(.gz)`."]
    (S4 / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {S4 / 'report.md'}")


if __name__ == "__main__":
    main()
