"""STS per-layer baselines, l*, and the published-value sanity check.

For each model x layer: Spearman (no deletion) on stsb_dev, sickr_dev and each E task.
l* = argmax over layers of mean(rho_stsb_dev, rho_sickr_dev). E numbers are descriptive only.
Sanity: final-layer 7-task E average (x100) vs SimCSE Table 5; a gap > 5 points
for any contrastive model is a STOP.

Outputs: results/sts/baselines/{layer_spearman,best_layer,sanity}.csv, report.md, figures/sts_baselines_layer_curves.png
(each with .meta.json). `--out DIR` writes everything to DIR instead (development / verification).

Usage: python scripts/sts_baselines.py [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fyp.corpora import DEV_KEYS, E_KEYS
from fyp.evaluation import load_embeddings, load_pairs_index, pair_cosine, spearman
from fyp.models import MODEL_KEYS, MODEL_SPECS
from fyp.paths import FIGURES, STS_BASELINES
from fyp.plotting import C, TEXT_2, finish_figure, style_axis
from fyp.protocol import CONTRASTIVE_MODELS as CONTRASTIVE
from fyp.runtime import add_device_args, run_metadata, save_table, setup_device, write_metadata

PROTOCOL_ID = "sts"
SANITY_THRESHOLD = 5.0  # points

# SimCSE paper (arXiv 2104.08821) Table 5, Spearman x100, "all" setting.
# Published pooling: sup = [CLS]+MLP, unsup = [CLS] w/o MLP, SBERT = mean; BERT row = first-last avg.
PUBLISHED = {
    "sup-simcse":   [75.30, 84.67, 80.19, 85.40, 80.82, 84.25, 80.39, 81.57],
    "unsup-simcse": [68.40, 82.41, 74.38, 80.91, 78.56, 76.85, 72.23, 76.25],
    "sbert-nli":    [70.97, 76.53, 73.19, 79.09, 74.30, 77.03, 72.91, 74.89],
    "bert-raw":     [39.70, 59.38, 49.67, 66.03, 66.19, 53.87, 62.06, 56.70],  # first-last avg, not comparable
}



def layer_table() -> pd.DataFrame:
    rows = []
    for model in MODEL_KEYS:
        for source in DEV_KEYS + E_KEYS:
            emb = np.asarray(load_embeddings(model, source, mmap=False))
            idx = load_pairs_index(source)
            for layer in range(emb.shape[1]):
                x = emb[:, layer, :].astype(np.float64)
                rho = spearman(pair_cosine(x[idx["idx1"]], x[idx["idx2"]]), idx["score"])
                rows.append({"model": model, "layer": layer, "source": source,
                             "role": "DEV" if source in DEV_KEYS else "E", "spearman": rho})
        print(f"  {model}: done")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None, help="write results and the figure here instead of results/sts/baselines and figures/")
    add_device_args(p)
    args = p.parse_args()
    dev = setup_device(args.require_gpu, args.device)  # arithmetic here is numpy float64 on CPU
    out_dir, fig_dir = (args.out, args.out) if args.out else (STS_BASELINES, FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(dev, script="scripts/sts_baselines.py", protocol_id=PROTOCOL_ID,
                        extra={"note": "all Spearman arithmetic is float64 numpy on CPU; device field records torch device selection only"})

    df = layer_table()
    save_table(df, out_dir / "layer_spearman.csv", meta)

    wide = df.pivot_table(index=["model", "layer"], columns="source", values="spearman").reset_index()
    wide["dev_mean"] = wide[DEV_KEYS].mean(axis=1)
    wide["E_avg"] = wide[E_KEYS].mean(axis=1)

    lstar_rows, sanity_rows = [], []
    for model in MODEL_KEYS:
        w = wide[wide["model"] == model].sort_values("layer")
        final = MODEL_SPECS[model].final_layer
        lstar = int(w.loc[w["dev_mean"].idxmax(), "layer"])
        at = lambda L, c: float(w.loc[w["layer"] == L, c].iloc[0])
        lstar_rows.append({
            "model": model, "final_layer": final, "l_star": lstar, "l_star_is_final": lstar == final,
            "dev_mean_at_l_star": at(lstar, "dev_mean"), "dev_mean_at_final": at(final, "dev_mean"),
            "stsb_dev_at_final": at(final, "stsb_dev"), "sickr_dev_at_final": at(final, "sickr_dev"),
            "E_avg_at_l_star": at(lstar, "E_avg"), "E_avg_at_final": at(final, "E_avg"),
        })
        ours = [100 * at(final, k) for k in E_KEYS] + [100 * at(final, "E_avg")]
        row = {"model": model, **{f"ours_{k}": v for k, v in zip(E_KEYS + ["avg"], ours)}}
        if model in PUBLISHED:
            row.update({f"pub_{k}": v for k, v in zip(E_KEYS + ["avg"], PUBLISHED[model])})
            row["gap_avg"] = ours[-1] - PUBLISHED[model][-1]
            row["comparable"] = model in CONTRASTIVE
            row["sanity_pass"] = (abs(row["gap_avg"]) <= SANITY_THRESHOLD) if model in CONTRASTIVE else None
        sanity_rows.append(row)

    lstar_df = pd.DataFrame(lstar_rows)
    sanity_df = pd.DataFrame(sanity_rows)
    for name, d in (("best_layer.csv", lstar_df), ("sanity.csv", sanity_df)):
        save_table(d, out_dir / name, meta)

    # ---- figure: small multiples, one panel per model ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for ax, model in zip(axes.flat, MODEL_KEYS):
        w = wide[wide["model"] == model].sort_values("layer")
        ls = int(lstar_df.loc[lstar_df["model"] == model, "l_star"].iloc[0])
        ax.plot(w["layer"], w["dev_mean"], color=C["blue"], lw=2, marker="o", ms=4, label="DEV mean (selects l*)")
        ax.plot(w["layer"], w["E_avg"], color=C["orange"], lw=2, marker="o", ms=4, label="E 7-task avg (descriptive)")
        ax.axvline(ls, color=TEXT_2, lw=1, ls="--")
        ax.text(ls, 0.415, f" l*={ls}", color=TEXT_2, fontsize=8, va="bottom",
                ha="right" if ls == w["layer"].max() else "left")
        ax.set_title(model, fontsize=10)
        ax.set_xticks(w["layer"])
        style_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("hidden state (0 = embedding output)", fontsize=9, color=TEXT_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Spearman ρ (mean-pooled, cosine)", fontsize=9, color=TEXT_2)
    axes[0, 0].set_ylim(0.4, 0.9)  # line chart: zero baseline not required; shared across panels
    finish_figure(fig, f"STS per-layer baseline performance (no deletion)", fig_dir / "sts_baselines_layer_curves.png",
                  axes[0, 0], ncol=2)
    write_metadata(fig_dir / "sts_baselines_layer_curves.png", meta)

    # ---- markdown report ----
    L = [f"# STS baselines and l\\*", "",
         f"git `{meta['git_commit']}` (dirty={meta['git_dirty']}) · generated by `scripts/sts_baselines.py`. "
         "Mean pooling, cosine, Spearman ×100 unless noted. E numbers are **descriptive only**.", "",
         "## l\\* per model (argmax over layers of mean(STS-B dev, SICK-R dev))", "",
         "| Model | final layer | l\\* | l\\* = final? | DEV mean @ l\\* | DEV mean @ final | E avg @ l\\* | E avg @ final |",
         "|---|---|---|---|---|---|---|---|"]
    for r in lstar_rows:
        L.append(f"| {r['model']} | {r['final_layer']} | {r['l_star']} | {'yes' if r['l_star_is_final'] else '**no**'} | "
                 f"{100 * r['dev_mean_at_l_star']:.2f} | {100 * r['dev_mean_at_final']:.2f} | "
                 f"{100 * r['E_avg_at_l_star']:.2f} | {100 * r['E_avg_at_final']:.2f} |")
    L += ["", "Ablation always uses the final layer; l\\* is reported only.", "",
          "## Sanity check: final layer vs SimCSE Table 5 (\"all\" setting)", "",
          "| Model | " + " | ".join(E_KEYS) + " | **avg** | published avg | gap | verdict |",
          "|---|" + "---|" * (len(E_KEYS) + 4)]
    for r in sanity_rows:
        cells = " | ".join(f"{r[f'ours_{k}']:.2f}" for k in E_KEYS)
        if "pub_avg" in r:
            verdict = ("PASS" if r["sanity_pass"] else "**FAIL**") if r["comparable"] else "n/a (published row is first-last avg)"
            L.append(f"| {r['model']} | {cells} | **{r['ours_avg']:.2f}** | {r['pub_avg']:.2f} | {r['gap_avg']:+.2f} | {verdict} |")
        else:
            L.append(f"| {r['model']} | {cells} | **{r['ours_avg']:.2f}** | – | – | no published reference |")
    L += ["", f"Threshold: |gap| ≤ {SANITY_THRESHOLD:.0f} points for contrastive models. Published pooling differs "
          "(sup-SimCSE [CLS]+MLP, unsup-SimCSE [CLS]); SICK-R test here is 4,906 vs 4,927 pairs.", "",
          "![per-layer curves](../../figures/sts_baselines_layer_curves.png)"]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    failed = [r["model"] for r in sanity_rows if r.get("sanity_pass") is False]
    if failed:
        sys.exit(f"STOP: sanity check failed for {failed} (gap to the published average above {SANITY_THRESHOLD:.0f} points)")


if __name__ == "__main__":
    main()
