"""Predictions for the Matryoshka pair (random-removal robustness, prefix against random cuts), from results/matryoshka/{sts,retrieval,classification}/.

Reads the CSVs written by sts_truncation_and_ranked_removal.py and retrieval_classification_truncation_and_ranked_removal.py (final test run) and recomputes nothing.
Writes predictions.csv (one row per prediction, model and suite) and rp_at_k.csv (RP of random-, last- and first-K at K = 25, 50, 75).
Usage: python scripts/report_matryoshka.py [--root results/matryoshka]
"""

import argparse
from pathlib import Path

import pandas as pd

from fyp.models import MRL_MODEL_KEYS
from fyp.paths import MRL_PAIR
from fyp.protocol import ROBUSTNESS_THRESHOLD, n_remove
from fyp.runtime import run_metadata, save_table, setup_device
from report_retrieval_classification import load, rp_tables

MRL, NO_MRL = MRL_MODEL_KEYS
N50 = n_remove(50)


def curves(root: Path):
    """{suite: summary table with model, method, n, K, RP_mean, band_lo, band_hi} for sts, ret, cls."""
    out = {"sts": pd.read_csv(root / "sts" / "truncation_summary.csv")}
    for suite in ("ret", "cls"):
        out[suite] = rp_tables(load(root, suite))[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=MRL_PAIR)
    args = ap.parse_args()
    dev = setup_device(False, "cpu")
    cur = curves(args.root)
    at = lambda suite, model, method, n=N50, col="RP_mean": float(
        cur[suite][(cur[suite]["model"] == model) & (cur[suite]["method"] == method) & (cur[suite]["n"] == n)][col].iloc[0])
    rows = []
    for suite in ("sts", "ret", "cls"):
        for m in (MRL, NO_MRL):
            v = at(suite, m, "random")
            rows.append({"prediction": "random_removal_robust", "suite": suite, "model": m, "value": v, "threshold": ROBUSTNESS_THRESHOLD, "supported": v >= ROBUSTNESS_THRESHOLD})
    for suite in ("sts", "ret"):
        r, hi, last = at(suite, MRL, "random"), at(suite, MRL, "random", col="band_hi"), at(suite, MRL, "last")
        rows.append({"prediction": "mrl_prefix_above_random", "suite": suite, "model": MRL, "value": last, "random_mean": r, "random_band_hi": hi,
                     "supported": last > r and last > hi})
        r, hi, last = at(suite, NO_MRL, "random"), at(suite, NO_MRL, "random", col="band_hi"), at(suite, NO_MRL, "last")
        rows.append({"prediction": "no_mrl_prefix_not_above_random", "suite": suite, "model": NO_MRL, "value": last, "random_mean": r, "random_band_hi": hi,
                     "supported": not last > hi})
        a, b = at(suite, MRL, "random"), at(suite, NO_MRL, "random")
        rows.append({"prediction": "mrl_random_below_no_mrl", "suite": suite, "model": f"{MRL} < {NO_MRL}", "value": a, "other": b, "supported": a < b})
    pred = pd.DataFrame(rows)
    grid = pd.DataFrame([{"suite": s, "model": m, "method": meth, "K": k, "RP": at(s, m, meth, n_remove(k)),
                          "band_lo": at(s, m, meth, n_remove(k), "band_lo"), "band_hi": at(s, m, meth, n_remove(k), "band_hi")}
                         for s in ("sts", "ret", "cls") for m in (MRL, NO_MRL) for meth in ("random", "last", "first") for k in (25, 50, 75)])
    meta = run_metadata(dev, script="scripts/report_matryoshka.py", protocol_id="matryoshka", extra={"root": str(args.root)})
    save_table(pred, args.root / "predictions.csv", meta)
    save_table(grid, args.root / "rp_at_k.csv", meta)
    print(pred.to_string(index=False))


if __name__ == "__main__":
    main()
