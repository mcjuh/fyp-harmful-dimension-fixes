"""Figure style and the small-multiples layout shared by every report figure (reference palette, light mode)."""

from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fyp.protocol import CONTRASTIVE_MODELS  # noqa: E402

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "purple": "#7b5ea7", "brown": "#8c5a2b", "slate": "#34495e"}   # the last three mark the raw encoders in the summary figures
BAND, TEXT_2, GRID = "#c9c8c2", "#52514e", "#e4e3df"


def style_axis(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=TEXT_2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def group_title(model: str) -> str:
    return model + (" (contrastive)" if model in CONTRASTIVE_MODELS else " (raw)")


def finish_figure(fig, title: str, path, legend_ax=None, ncol: int = 4) -> None:
    """Suptitle, shared legend taken from `legend_ax`, tight layout, save at 150 dpi, close."""
    fig.suptitle(title, y=0.995, fontsize=11)
    if legend_ax is not None:
        h, l = legend_ax.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=ncol, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def model_grid(models: Sequence[str], draw: Callable, *, title: str, xlabel: str, ylabel: str, path, ncol: int = 4,
               title_fn: Callable[[str], str] = lambda m: m, legend_from: int = 0) -> None:
    """2 x 3 panels, one per model: `draw(ax, model)` fills a panel (it may reset the panel title)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, m in zip(axes.flat, models):
        ax.set_title(title_fn(m), fontsize=10)
        draw(ax, m)
        style_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_2)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_2)
    finish_figure(fig, title, path, axes.flat[legend_from], ncol)
