"""l* per task type: the best layer for STS, retrieval and classification, per model.

  STS  l*  = scripts/sts_baselines.py (argmax of the macro-average of STS-B dev and SICK-R dev Spearman).
  ret  l*  = argmax over layers of macro-average DEV-query nDCG@10 over the 13 NanoBEIR datasets (cosine).
  cls  l*  = argmax over layers of macro-average DEV accuracy over the 12 MTEB tasks (probe on all 768 dims,
             the same 10 few-shot draws, solved to convergence on the GPU).
Test-split per-layer numbers are computed too and reported as DESCRIPTIVE only (they never select a layer).
No truncation / attribution is run at l*.

Outputs: results/retrieval_classification/best_layer_per_layer.csv, best_layer_table.csv, figures/retrieval_classification_best_layer.png (+ .meta.json sidecars); per-model parts in
results/retrieval_classification/_parts/ are reused if present. `--out DIR` writes everything to DIR instead.
Usage: python scripts/select_layer.py [--models ...] [--out DIR] [--require-gpu]
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fyp.models import MODEL_KEYS, MODEL_SPECS
from fyp.paths import FIGURES, STS_BASELINES, SUITE
from fyp.plotting import C as PALETTE
from fyp.runtime import add_device_args, run_metadata, setup_device, write_metadata
from fyp.paths import CLS_ALL_LAYERS_CACHE
from fyp.suite import load_cls_tasks, load_ret_tasks

PROTOCOL_ID = "retrieval_classification"


def per_layer(model, device):
    rows = []
    L = MODEL_SPECS[model].num_hidden_states
    for layer in range(L):
        for split in ("dev", "test"):
            ret = load_ret_tasks(model, split, layer=layer, device=device)
            vals = {k: t.base() for k, t in ret.items()}
            rows.append({"model": model, "suite": "ret", "layer": layer, "split": split, "metric": float(np.mean(list(vals.values()))),
                         **{f"task_{k}": v for k, v in vals.items()}})
            del ret
            cls = load_cls_tasks(model, split, layer=layer, device=device, cache=CLS_ALL_LAYERS_CACHE)
            vals = {k: t.base() for k, t in cls.items()}
            rows.append({"model": model, "suite": "cls", "layer": layer, "split": split, "metric": float(np.mean(list(vals.values()))),
                         **{f"task_{k}": v for k, v in vals.items()}})
            del cls
        print(f"  {model} layer {layer}/{L - 1}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    p.add_argument("--out", type=Path, default=None, help="write tables, parts and the figure here instead of results/retrieval_classification and figures/")
    add_device_args(p)
    args = p.parse_args()
    dev = setup_device(args.require_gpu, args.device)
    OUT, fig_dir = (args.out, args.out) if args.out else (SUITE, FIGURES)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    part = OUT / "_parts"
    part.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        f = part / f"best_layer_{model}.csv"
        if f.exists():
            rows += pd.read_csv(f).to_dict("records")
            continue
        r = per_layer(model, dev["device"])
        pd.DataFrame(r).to_csv(f, index=False)
        rows += r
    df = pd.DataFrame(rows)
    sts = pd.read_csv(STS_BASELINES / "layer_spearman.csv")
    dev_sts = sts[sts["role"] == "DEV"].groupby(["model", "layer"])["spearman"].mean().reset_index()
    e_sts = sts[sts["role"] != "DEV"].groupby(["model", "layer"])["spearman"].mean().reset_index()
    sts_rows = pd.concat([dev_sts.assign(split="dev"), e_sts.assign(split="test")]).rename(columns={"spearman": "metric"})
    sts_rows["suite"] = "sts"
    allrows = pd.concat([df[["model", "suite", "layer", "split", "metric"]], sts_rows[["model", "suite", "layer", "split", "metric"]]],
                        ignore_index=True)
    allrows = allrows[allrows["model"].isin(args.models)]
    allrows.to_csv(OUT / "best_layer_per_layer.csv", index=False)
    tab = []
    for model in args.models:
        final = MODEL_SPECS[model].final_layer
        row = {"model": model, "final_layer": final}
        for suite in ("sts", "ret", "cls"):
            d = allrows[(allrows["model"] == model) & (allrows["suite"] == suite) & (allrows["split"] == "dev")].sort_values("layer")
            ls = int(d.loc[d["metric"].idxmax(), "layer"])
            t = allrows[(allrows["model"] == model) & (allrows["suite"] == suite) & (allrows["split"] == "test")].set_index("layer")["metric"]
            dv = d.set_index("layer")["metric"]
            row.update({f"l_star_{suite}": ls, f"{suite}_is_final": ls == final, f"{suite}_dev_at_l_star": float(dv[ls]),
                        f"{suite}_dev_at_final": float(dv[final]), f"{suite}_test_at_l_star_descriptive": float(t[ls]),
                        f"{suite}_test_at_final_descriptive": float(t[final]), f"{suite}_test_argmax_layer_descriptive": int(t.idxmax())})
        tab.append(row)
    tab = pd.DataFrame(tab)
    tab.to_csv(OUT / "best_layer_table.csv", index=False)
    meta = run_metadata(dev, script="scripts/select_layer.py", protocol_id=PROTOCOL_ID, seeds={"fewshot_seed": 32000},
                        extra={"models": args.models, "selection": "argmax over layers of the macro-average dev metric; sts from sts_baselines.py",
                               "seconds": round(time.time() - t0, 1)})
    write_metadata(OUT / "best_layer_per_layer.csv", meta)
    write_metadata(OUT / "best_layer_table.csv", meta)
    # figure: per-layer dev curves, min-max normalised per model and suite so layers are comparable
    C = {"sts": PALETTE["blue"], "ret": PALETTE["orange"], "cls": PALETTE["aqua"]}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for ax, model in zip(axes.flat, args.models):
        for suite, lbl in (("sts", "STS"), ("ret", "retrieval"), ("cls", "classification")):
            d = allrows[(allrows["model"] == model) & (allrows["suite"] == suite) & (allrows["split"] == "dev")].sort_values("layer")
            v = d["metric"].to_numpy()
            ax.plot(d["layer"], (v - v.min()) / (v.max() - v.min() + 1e-12), color=C[suite], lw=2, label=lbl)
            i = int(np.argmax(v))
            ax.plot(d["layer"].iloc[i], 1.0, "o", color=C[suite], ms=7)
        ax.set_title(model, fontsize=10)
        ax.grid(axis="y", color="#e4e3df")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("layer (hidden state index)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("dev metric, min-max per curve", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("l* per task type: dev metric by layer (dot = l*)", y=1.0, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "retrieval_classification_best_layer.png", dpi=150, bbox_inches="tight")
    write_metadata(fig_dir / "retrieval_classification_best_layer.png", meta)
    print(tab[["model", "final_layer", "l_star_sts", "l_star_ret", "l_star_cls"]].to_string(index=False))


if __name__ == "__main__":
    main()
