"""Two summary figures for the CA report, all eight models in one panel set per suite (STS, retrieval, classification).

Reads the final-run tables under results/{sts,retrieval_classification,matryoshka}/ and recomputes nothing.
  report_random_truncation.png : RP of random-K removal against K, one line per model.
  report_gain_in_vs_held_out.png : gain of ranked (best-first) removal over the random mean at K = 50, in-sample against held-out.
  report_transformations_full_width.png : full-width score after each transformation relative to the raw embedding, heat map per suite.
Usage: python scripts/figures_summary.py [--fig-dir figures]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from fyp.models import ALL_MODEL_KEYS as MODELS
from fyp.paths import FIGURES, MRL_PAIR, STS_TRUNCATION, SUITE, TRANSFORMS, SUITE_DIR
from fyp.plotting import C, TEXT_2, finish_figure, style_axis
from fyp.protocol import CONTRASTIVE_MODELS, K_GRID, N_GRID, n_remove
from fyp.runtime import run_metadata, setup_device, write_metadata
from fyp.transforms import TRANSFORMS as TRANSFORM_NAMES
from report_retrieval_classification import load, rp_tables

N50 = n_remove(50)
SUITE_TITLES = {"sts": "STS", "ret": "Retrieval (NanoBEIR)", "cls": "Classification (MTEB)"}
HELD_OUT = {"sts": "held_out", "ret": "held_out", "cls": "held_out"}
COLOURS = dict(zip(CONTRASTIVE_MODELS, (C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"])))
RAW_STYLE = {"bert-raw": (C["purple"], "o"), "roberta-raw": (C["brown"], "s"), "distilbert-raw": (C["slate"], "^")}   # colour and marker of the raw encoders (dashed lines)


def sts_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(random-removal summary, best-first summary) for the eight models; STS is read from the summary tables as written."""
    r1 = pd.concat([pd.read_csv(d / "truncation_summary.csv") for d in (STS_TRUNCATION, MRL_PAIR / "sts")])
    r4 = pd.concat([pd.read_csv(d / "ranked_removal_summary.csv") for d in (STS_TRUNCATION, MRL_PAIR / "sts")])
    return r1, r4


def suite_tables(suite: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t1, t4 = [], []
    for root in (SUITE, MRL_PAIR):
        a, _, c = rp_tables(load(root, suite))
        t1.append(a)
        t4.append(c)
    return pd.concat(t1), pd.concat(t4)


def tables() -> dict:
    return {"sts": sts_tables(), "ret": suite_tables("ret"), "cls": suite_tables("cls")}


def fig_random(tabs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, (suite, (t1, _)) in zip(axes, tabs.items()):
        for m in MODELS:
            d = t1[(t1["model"] == m) & (t1["method"] == "random") & t1["n"].isin(N_GRID)].sort_values("n")
            raw = m not in CONTRASTIVE_MODELS
            ax.plot(d["K"], d["RP_mean"], color=RAW_STYLE[m][0] if raw else COLOURS[m], lw=1.6, ls="--" if raw else "-",
                    marker=RAW_STYLE[m][1] if raw else None, markevery=3, ms=4, label=m)
        ax.set_title(SUITE_TITLES[suite], fontsize=10)
        ax.set_xlabel("K = % of dimensions removed", fontsize=9, color=TEXT_2)
        style_axis(ax)
    axes[0].set_ylabel("relative performance (RP)", fontsize=9, color=TEXT_2)
    finish_figure(fig, "Random removal of dimensions: RP against K", path, axes[0], ncol=8)


def fig_gain(tabs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    ypos = {m: i for i, m in enumerate(reversed(MODELS))}
    for ax, (suite, (t1, t4)) in zip(axes, tabs.items()):
        rand = t1[(t1["method"] == "random") & (t1["n"] == N50)].set_index("model")["RP_mean"]
        best = t4[(t4["curve"] == "best_first") & (t4["n"] == N50)]
        for m in MODELS:
            g = {p: float(best[(best["model"] == m) & (best["protocol"] == p)]["RP"].iloc[0]) - rand[m]
                 for p in ("in_sample", HELD_OUT[suite])}
            y = ypos[m]
            ax.plot([g[HELD_OUT[suite]], g["in_sample"]], [y, y], color=TEXT_2, lw=1, zorder=1)
            ax.scatter(g["in_sample"], y, s=36, facecolor="white", edgecolor=C["blue"], lw=1.5, zorder=2, label="ranking built on the evaluation data")
            ax.scatter(g[HELD_OUT[suite]], y, s=36, color=C["blue"], zorder=2, label="ranking built on other data")
        ax.axvline(0, color=TEXT_2, lw=0.8)
        ax.set_title(SUITE_TITLES[suite], fontsize=10)
        ax.set_xlabel("gain in RP over random removal (K = 50)", fontsize=9, color=TEXT_2)
        ax.set_yticks(list(ypos.values()), list(ypos.keys()))
        style_axis(ax)
        ax.grid(axis="x", color="#e4e3df", lw=0.8)
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    keep = dict(zip(l, h))
    axes[0].legend(keep.values(), keep.keys(), loc="lower right", frameon=False, fontsize=7.5)
    finish_figure(fig, "Gain of ranked removal over random removal at K = 50", path, None)


def fig_transforms(tabs: dict, path: Path) -> None:
    """Heat map per suite: score of the full-width embedding after each transformation divided by the raw full-width score (colour clipped to 0.85-1.20, cells carry the value)."""
    labels = {"centre": "centre", "standardise": "standardise", "mean_r1": "mean R1", "mean_r2": "mean R2", "abtt": "all-but-the-top", "whiten": "whiten"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=0.85, vmax=1.20)
    for ax, suite in zip(axes, ("sts", "ret", "cls")):
        d = pd.read_csv(TRANSFORMS / SUITE_DIR[suite] / "summary.csv")
        full = d[d["cut"] == "random"].pivot(index="model", columns="transform", values="full")
        ratio = full[TRANSFORM_NAMES].div(full["raw"], axis=0).loc[MODELS]
        im = ax.imshow(ratio.to_numpy(), cmap="RdBu", norm=norm, aspect="auto")
        for i in range(ratio.shape[0]):
            for j in range(ratio.shape[1]):
                v = ratio.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v - 1) > 0.12 else "#201f1c")
        ax.set_xticks(range(len(TRANSFORM_NAMES)), [labels[t] for t in TRANSFORM_NAMES], rotation=35, ha="right", fontsize=8, color=TEXT_2)
        ax.set_yticks(range(len(MODELS)), MODELS, fontsize=8, color=TEXT_2)
        ax.set_title(SUITE_TITLES[suite], fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="full-width score / raw full-width score")
    fig.suptitle("Full-width score after each transformation, relative to the raw embedding", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig-dir", type=Path, default=FIGURES)
    args = p.parse_args()
    dev = setup_device(False, "cpu")
    meta = run_metadata(dev, script="scripts/figures_summary.py", protocol_id="summary_figures")
    tabs = tables()
    for fn, fnc in (("report_random_truncation.png", fig_random), ("report_gain_in_vs_held_out.png", fig_gain),
                    ("report_transformations_full_width.png", fig_transforms)):
        fnc(tabs, args.fig_dir / fn)
        write_metadata(args.fig_dir / fn, meta)
        print("wrote", args.fig_dir / fn)


if __name__ == "__main__":
    main()
