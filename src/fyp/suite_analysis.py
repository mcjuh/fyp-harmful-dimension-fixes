"""The paper's analyses (the analyses + headline CIs) written once over the `fyp.suite` evaluator interface.

Every function takes `tasks: {name: evaluator}` for ONE model (final layer) and returns long DataFrames.
Conventions (as the STS work): K grid 0..90 by 5; n = 768 - int(768 * (100 - K)/100) (the authors' keep count); ties in a score are broken by a
seeded random permutation, never by dimension index; relative performance (RP) is the macro-average over tasks of the
metric divided by the macro-average of the full-embedding metric. The evaluator-independent statistics live in
`fyp.dimension_stats` and are shared with the STS pipeline.
"""

import numpy as np
import pandas as pd

from fyp.ablation import ranked_order
from fyp.bootstrap import bootstrap_means
from fyp.dimension_stats import estimates_table, headline_estimates, outlier_sets, removal_curve_specs, shared_dims_table
from fyp.protocol import K_GRID, N25, N50, N_BOOT, N_DIMS, N_FID_RANDOM, N_GRID, SEED_BOOT, SEED_TIE

RET_RANDOM, CLS_RANDOM = 100, 10   # random-K / random-order seeds per suite (cls: one per few-shot experiment)


def keep_from_order(order, ns) -> np.ndarray:
    """(len(ns), 768) bool KEEP masks: delete the first n entries of `order`, for each n (nested)."""
    keep = np.ones((len(ns), N_DIMS), dtype=bool)
    for i, n in enumerate(ns):
        keep[i, np.asarray(order)[:n]] = False
    return keep


def paired_experiments(task, seed, B):
    """Classification pairs random-K seed s with few-shot experiment s (A1); retrieval has no experiments."""
    return np.full(B, seed % task.E) if task.kind == "cls" else None


def random_seeds(task_kind: str, base: int = 0) -> range:
    return range(base, base + (RET_RANDOM if task_kind == "ret" else CLS_RANDOM))


# ------------------------------------------------------------------ truncation and fidelity
def run_truncation(tasks: dict, model: str) -> tuple:
    """Last-K / first-K / random-K curves and label-free fidelity for every task."""
    kind = next(iter(tasks.values())).kind
    orders = {("last", -1): np.arange(N_DIMS - 1, -1, -1), ("first", -1): np.arange(N_DIMS)}
    for s in random_seeds(kind):
        orders[("random", s)] = np.random.default_rng(s).permutation(N_DIMS)
    r1, r2 = [], []
    for name, task in tasks.items():
        base = task.base()
        for (method, seed), order in orders.items():
            keep = keep_from_order(order, N_GRID)
            exps = paired_experiments(task, seed, len(N_GRID)) if method == "random" else None
            if method != "random" or seed < N_FID_RANDOM:
                m, f = task.both(keep, exps)
                r2 += [{"model": model, "method": method, "seed": seed, "task": name, "n": n, "K": k, "fidelity": float(x)}
                       for n, k, x in zip(N_GRID, K_GRID, f)]
            else:
                m = task.metric(keep, exps)
            assert abs(m[0] - base) < 1e-9 or (method == "random" and kind == "cls"), (name, method, seed, m[0], base)
            r1 += [{"model": model, "method": method, "seed": seed, "task": name, "n": n, "metric": float(x)}
                   for n, x in zip(N_GRID, m)]
    return pd.DataFrame(r1), pd.DataFrame(r2)


# ------------------------------------------------------------------ attribution
def make_units(deltas: dict) -> dict:
    """{('in_sample','all'): (score, eval_tasks), ('held_out', t): (score, [t])}. in_sample = in-sample (the paper); held_out ranks on the other tasks."""
    names = list(deltas)
    units = {("in_sample", "all"): (np.mean([deltas[k] for k in names], 0), names)}
    for t in names:
        units[("held_out", t)] = (np.mean([deltas[k] for k in names if k != t], 0), [t])
    return units


# ------------------------------------------------------------------ ranked removal
def run_removal_curves(tasks: dict, units: dict, model: str) -> pd.DataFrame:
    """best-first / worst-first through all 768, degrading-only / informative-only (ranked and random orders), per unit and eval task."""
    kind = next(iter(tasks.values())).kind
    seeds_d, seeds_i = random_seeds(kind, 1000), random_seeds(kind, 2000)
    rows = []
    bases = {k: t.base() for k, t in tasks.items()}
    for (proto, fold), (score, eval_tasks) in units.items():
        curves, _, _, nD, nI = removal_curve_specs(score, seeds_d, seeds_i)
        for (cname, seed), (order, ns) in curves.items():
            keep = keep_from_order(order, ns)
            for k in eval_tasks:
                task = tasks[k]
                exps = paired_experiments(task, seed - (1000 if cname == "degrading_only_random" else 2000), len(ns)) if seed >= 0 else None
                m = task.metric(keep, exps)
                if ns[0] == 0:
                    assert abs(m[0] - bases[k]) < 1e-9 or (seed >= 0 and kind == "cls"), (k, cname, m[0], bases[k])
                rows += [{"model": model, "protocol": proto, "fold": fold, "curve": cname, "seed": seed, "task": k, "n": n,
                          "metric": float(x), "n_degrading": nD, "n_informative": nI} for n, x in zip(ns, m)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ shared dimensions
def run_shared(deltas: dict, model: str) -> pd.DataFrame:
    return shared_dims_table(np.stack([deltas[k] > 0 for k in deltas]), model)


# ------------------------------------------------------------------ outliers
def run_outliers(tasks: dict, mu: np.ndarray, model: str) -> tuple:
    """Outlier coordinates of the mean embedding `mu` (|z| > 3 over its 768 values; paper Sec 3.3), removed vs the same
    number of random non-outlier coordinates (10 sets). RP = macro metric / macro full metric."""
    z, outl, sets = outlier_sets(mu)
    zdf = pd.DataFrame({"model": model, "dim": np.arange(N_DIMS), "z": z})
    if not sets:
        return pd.DataFrame([{"model": model, "set": "outliers", "seed": -1, "n_removed": 0, "RP": 1.0}]), zdf
    base_mean = np.mean([t.base() for t in tasks.values()])
    keep = np.ones((len(sets), N_DIMS), dtype=bool)
    for i, S in enumerate(sets.values()):
        keep[i, S] = False
    per_task = {k: t.metric(keep) for k, t in tasks.items()}
    macro = np.mean([per_task[k] for k in tasks], 0)
    rows = [{"model": model, "set": nm, "seed": sd, "n_removed": len(outl), "RP": float(macro[i] / base_mean),
             **{f"ratio_{k}": float(per_task[k][i] / tasks[k].base()) for k in tasks}} for i, (nm, sd) in enumerate(sets)]
    return pd.DataFrame(rows), zdf


# ------------------------------------------------------------------ headline paired bootstrap
def run_headline_ci(tasks: dict, units: dict, model: str) -> pd.DataFrame:
    """Paired bootstrap (1000 resamples of queries / test items, same indices for every method) of RP at K = 25, 50:
    random-K mean, last-K, and best-/worst-first under in_sample (in-sample) and held_out."""
    kind = next(iter(tasks.values())).kind
    names = list(tasks)
    seeds = list(random_seeds(kind))
    boot = {}
    for ti, k in enumerate(names):
        task = tasks[k]
        cols, labels = [], []

        def add(label, keep, exps=None):
            cols.append(task.per_sample(keep, exps)); labels.append(label)

        add("full", np.ones((1, N_DIMS), bool))
        last = np.arange(N_DIMS - 1, -1, -1)
        add("last_25", keep_from_order(last, [N25])); add("last_50", keep_from_order(last, [N50]))
        for s in seeds:
            order = np.random.default_rng(s).permutation(N_DIMS)
            for K, n in ((25, N25), (50, N50)):
                add(f"random{s}_{K}", keep_from_order(order, [n]), paired_experiments(task, s, 1))
        for proto in ("in_sample", "held_out"):
            sc = units[("in_sample", "all")][0] if proto == "in_sample" else units[("held_out", k)][0]
            for cn, desc in (("best", True), ("worst", False)):
                order = ranked_order(sc, desc, SEED_TIE)
                add(f"{proto}_{cn}_25", keep_from_order(order, [N25])); add(f"{proto}_{cn}_50", keep_from_order(order, [N50]))
        ps = np.concatenate(cols, 0)                                   # (n_methods, N samples)
        boot[k] = dict(zip(labels, bootstrap_means(ps, N_BOOT, SEED_BOOT + ti).T))   # each (n_boot,)
    full_mean = np.mean([boot[k]["full"] for k in names], 0)
    rp = lambda nm: np.mean([boot[k][nm] for k in names], 0) / full_mean
    rows = []
    for K in (25, 50):
        est = headline_estimates(rp, K, [f"random{s}" for s in seeds], ("in_sample", "held_out"))
        rows += estimates_table(est, {"model": model, "K": K})
    return pd.DataFrame(rows)
