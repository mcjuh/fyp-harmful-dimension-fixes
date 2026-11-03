"""Corrections before the cut: transform the embeddings, then cut.

For each transform in fyp.transforms (fitted on the 20k all-nli sample, never on evaluation data) the usual truncation curves are
computed on the transformed space: random-K (100 draws; classification 10), last-K and first-K, K = 0..90 in steps of 5.
On whitened coordinates the last-K cut is "keep the first k components". Raw (no transform) is not re-run: its curves are in
results/retrieval_classification and results/matryoshka and results/sts/truncation_and_ranked_removal with the same seeds.

Usage:
  python scripts/transformations.py --suite sts --split dev --out <scratch> --models bert-raw sup-simcse
  python scripts/transformations.py --suite ret --split dev --out <scratch> --models bert-raw --tasks arguana scifact --quick
  python scripts/transformations.py --suite sts --split test            # final, all eight models
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fyp.suite_analysis as A
from fyp.ablation import TaskData
from fyp.classification import ALL_CLS_KEYS
from fyp.corpora import DEV_KEYS, E_KEYS, NLI_KEY
from fyp.evaluation import load_layer_pairs
from fyp.models import ALL_MODEL_KEYS, cache_dir, final_layer
from fyp.paths import CACHE, TRANSFORMS as FINAL_OUT, SUITE_DIR
from fyp.protocol import K_GRID, N_DIMS, N_GRID, SEEDS_RANDOM_K
from fyp.retrieval import RET_KEYS
from fyp.runtime import add_device_args, run_metadata, save_table, setup_device
from fyp.suite import load_tasks
from fyp.transforms import TRANSFORMS, fit

PROTOCOL_ID = "transformations"


def sts_truncation(model, split, layer, f):
    """Truncation curves on the transformed STS pairs: one rho per task, method, seed and n."""
    keys = E_KEYS if split == "test" else DEV_KEYS
    orders = {("last", -1): np.arange(N_DIMS - 1, -1, -1), ("first", -1): np.arange(N_DIMS)}
    orders.update({("random", s): np.random.default_rng(s).permutation(N_DIMS) for s in SEEDS_RANDOM_K})
    rows = []
    for k in keys:
        a, b, gold = load_layer_pairs(model, k, layer)
        t = TaskData.from_pairs(k, f(a), f(b), gold)
        for (method, seed), order in orders.items():
            rho = t.rho_nested(order, N_GRID)
            rows += [{"model": model, "method": method, "seed": seed, "task": k, "n": n, "metric": float(r)} for n, r in zip(N_GRID, rho)]
    return pd.DataFrame(rows)


def run_model(suite, model, split, device, tasks_keys):
    t0 = time.time()
    layer = final_layer(model)
    X = np.asarray(np.load(cache_dir(CACHE, model) / f"{NLI_KEY}.emb.npy", mmap_mode="r")[:, layer, :], dtype=np.float64)
    out = []
    for name in TRANSFORMS:
        f = fit(name, X)
        if suite == "sts":
            t1 = sts_truncation(model, split, layer, f)
        else:
            kw = {"datasets": tasks_keys} if suite == "ret" else {"tasks": tasks_keys}
            tasks = load_tasks(suite, model, split, device=device, transform=f, **kw)
            t1, _ = A.run_truncation(tasks, model)
        out.append(t1.assign(transform=name))
        print(f"  {model}: {name} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, choices=["sts", "ret", "cls"])
    p.add_argument("--split", choices=("dev", "test"), default="test")
    p.add_argument("--models", nargs="+", default=ALL_MODEL_KEYS, choices=ALL_MODEL_KEYS)
    p.add_argument("--tasks", nargs="+", default=None, help="subset of datasets/tasks (development only; ret/cls)")
    p.add_argument("--out", type=Path, default=None, help="output directory (required for --split dev); default results/transformations/<suite>")
    p.add_argument("--quick", action="store_true", help="development: 10 random-K seeds instead of 100")
    p.add_argument("--reuse-parts", action="store_true", help="skip models whose part file exists; summarise all parts")
    add_device_args(p)
    args = p.parse_args()
    out_root = (args.out or (FINAL_OUT / SUITE_DIR[args.suite])).resolve()
    if args.split == "dev" and (args.out is None or out_root.is_relative_to(FINAL_OUT.resolve())):
        raise SystemExit("dev runs must not write under results/transformations (eval lock): pass --out <scratch dir>")
    if args.split == "test" and (args.quick or args.tasks):
        raise SystemExit("the final (test) run uses all tasks and all seeds")
    if args.quick:
        A.RET_RANDOM = 10
    dev = setup_device(args.require_gpu, args.device)
    keys = args.tasks or (RET_KEYS if args.suite == "ret" else ALL_CLS_KEYS)
    parts = out_root / "_parts"
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for model in args.models:
        path = parts / f"{args.suite}_{model}.pkl"
        if args.reuse_parts and path.exists():
            continue
        path.write_bytes(pickle.dumps(run_model(args.suite, model, args.split, dev["device"], keys)))
    present = [m for m in ALL_MODEL_KEYS if (parts / f"{args.suite}_{m}.pkl").exists() and (m in args.models or args.reuse_parts)]
    df = pd.concat([pickle.loads((parts / f"{args.suite}_{m}.pkl").read_bytes()) for m in present], ignore_index=True)
    meta = run_metadata(dev, script="scripts/transformations.py", protocol_id=PROTOCOL_ID, seeds={
        "random_K": "0..99 (sts, ret) / 0..9 paired with few-shot experiments (cls)"}, extra={
        "suite": args.suite, "split": args.split, "models": present, "tasks": keys if args.suite != "sts" else "E (test) / DEV (dev)",
        "transforms": TRANSFORMS, "fit_sample": f"{NLI_KEY}, final layer", "K_grid": K_GRID, "quick": args.quick,
        "seconds": round(time.time() - t0, 1)})
    save_table(df, out_root / "truncation_curves.csv.gz", meta)
    print(f"wrote {out_root} ({len(present)} models) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
