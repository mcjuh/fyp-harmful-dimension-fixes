"""Timing of the primitives behind each cut.

For every model, on the cached final-layer embeddings: the one-time cost of choosing or fitting a cut (leave-one-out
attribution per task type, PCA fit, mean estimate) and the repeat costs of using it (centring or PCA projection of a vector,
re-encoding a sentence). Once, model-independent: brute-force search at deployment-like corpus sizes for several kept widths,
on the GPU and the CPU, and the cost of building the narrowed index (prefix copy or random-column gather).
Nothing here computes an evaluation metric that is written out: attribution deltas are timed and discarded, retrieval and
classification use the DEV split, and STS uses the filtered STS-B train set (R), so the evaluation lock is not touched.
Implementations are the repository's own (closed-form float64 attribution; fp32 search); precision changes are out of scope.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from fyp.ablation import TaskData
from fyp.classification import ALL_CLS_KEYS
from fyp.corpora import NLI_KEY, R_KEYS
from fyp.evaluation import load_embeddings, load_layer_pairs
from fyp.models import ALL_MODEL_KEYS, MODEL_KEYS, final_layer, load_encoder
from fyp.paths import CACHE, OVERLAP_MASKS, TIMING
from fyp.protocol import N_DIMS, n_remove
from fyp.runtime import add_device_args, run_metadata, save_table, setup_device
from fyp.suite import load_tasks

PROTOCOL_ID = "timing"
RET_SAMPLE = ["scifact", "nfcorpus", "fiqa2018"]    # NanoBEIR corpora of 2,919, 2,953 and 4,598 documents (dev queries only)
KEPT = [(0, N_DIMS), (25, N_DIMS - n_remove(25)), (50, N_DIMS - n_remove(50)), (75, N_DIMS - n_remove(75))]


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(fn, dev, repeats, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        sync(dev)
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t)
    return ts


class Rows:
    def __init__(self, base):
        self.base, self.rows = base, []

    def add(self, primitive, role, ts, device, **extra):
        q = np.percentile(ts, [25, 50, 75])
        self.rows.append({**self.base, "primitive": primitive, "role": role, "device": device, "repeats": len(ts),
                          "median_s": q[1], "q25_s": q[0], "q75_s": q[2], **extra})
        print(f"  {primitive:34s} {role:9s} {device:4s} median {q[1]:.4g}s (n={extra.get('n_items', '')})", flush=True)


def model_rows(model, dev, smoke):
    layer = final_layer(model)
    R = Rows({"model": model})
    nli = np.asarray(load_embeddings(model, NLI_KEY)[:, layer, :], dtype=np.float64)
    X = nli[:2000] if smoke else nli
    reps = 2 if smoke else 5

    # one-time fits on the 20k all-nli sample (never on E)
    R.add("mean_fit", "one_time", bench(lambda: X.mean(0), "cpu", reps), "cpu", n_items=len(X), d_in=N_DIMS)
    mu = X.mean(0)
    R.add("centre_apply", "repeat", bench(lambda: X - mu, "cpu", reps), "cpu", n_items=len(X), d_in=N_DIMS, d_out=N_DIMS)
    pca_holder = {}
    def fit():
        pca_holder["p"] = PCA(n_components=N_DIMS, svd_solver="full").fit(X)
    R.add("pca_fit_full", "one_time", bench(fit, "cpu", 2 if smoke else 3, warmup=0), "cpu", n_items=len(X), d_in=N_DIMS, d_out=N_DIMS)
    pca = pca_holder["p"]
    V = X[:10000]
    for K in (50,):
        d_out = N_DIMS - n_remove(K)
        comps = pca.components_[:d_out]
        R.add(f"pca_project_to_{d_out}", "repeat", bench(lambda: (V - pca.mean_) @ comps.T, "cpu", reps), "cpu",
              n_items=len(V), d_in=N_DIMS, d_out=d_out)

    # one-time leave-one-out attribution, per task type (the selection cost of the label-ranked cut)
    masks = np.load(OVERLAP_MASKS)
    for k in R_KEYS:
        t = TaskData.from_pairs(k, *load_layer_pairs(model, k, layer)).subset(masks[f"R__{k}"])
        R.add("attribution_sts", "one_time", bench(t.single_deltas, "cpu", 2 if smoke else 3), "cpu", n_items=t.n, d_in=N_DIMS, task=k)
    keys = RET_SAMPLE[:1] if smoke else RET_SAMPLE
    for key, t in load_tasks("ret", model, "dev", device=dev, datasets=keys).items():
        R.add("attribution_ret", "one_time", bench(t.single_deltas, dev, 2, warmup=0), dev, n_items=int(t.nq), d_in=N_DIMS, task=key)
    ckeys = ALL_CLS_KEYS[:1] if smoke else ALL_CLS_KEYS[:2]
    for key, t in load_tasks("cls", model, "dev", device=dev, tasks=ckeys).items():
        R.add("attribution_cls", "one_time", bench(t.single_deltas, dev, 1, warmup=0), dev, n_items=int(t.m), d_in=N_DIMS, task=key,
              n_fits=int(getattr(t, "n_fits", -1)))

    # repeat: encoding a sentence (GPU, every hidden state pooled and copied back, as in the extraction code)
    sents = json.loads((CACHE / "sentences" / f"{NLI_KEY}.json").read_text())["sentences"][: (256 if smoke else 2000)]
    enc = load_encoder(model, dev)
    R.add("encode_sentences", "repeat", bench(lambda: enc.encode(sents), dev, 2 if smoke else 3), dev, n_items=len(sents), d_in=N_DIMS)
    del enc
    if dev == "cuda":
        torch.cuda.empty_cache()
    return R.rows


def search_rows(dev, smoke):
    R = Rows({"model": "synthetic"})
    rng = np.random.default_rng(0)
    nq, topk = 64, 10
    sizes = [10_000] if smoke else [100_000, 1_000_000]
    for N in sizes:
        C = rng.standard_normal((N, N_DIMS), dtype=np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        Q = rng.standard_normal((nq, N_DIMS), dtype=np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        for device in ([dev, "cpu"] if dev == "cuda" else ["cpu"]):
            if device == "cuda":
                Cd, Qd = torch.from_numpy(C).to("cuda"), torch.from_numpy(Q).to("cuda")
            for K, d in KEPT:
                idx = np.sort(rng.permutation(N_DIMS)[:d])
                common = {"n_items": N, "d_in": N_DIMS, "d_out": d, "K": K, "bytes_per_vector": 4 * d}
                reps = 3 if smoke else 5
                if device == "cuda":
                    idt = torch.from_numpy(idx).to("cuda")
                    R.add("index_build_prefix_copy", "one_time", bench(lambda: Cd[:, :d].contiguous(), "cuda", reps), "cuda", **common)
                    R.add("index_build_random_gather", "one_time", bench(lambda: Cd.index_select(1, idt), "cuda", reps), "cuda", **common)
                    Cp, Qp = Cd[:, :d].contiguous(), Qd[:, :d].contiguous()
                    R.add("search_top10", "repeat", bench(lambda: torch.topk(Qp @ Cp.T, topk, dim=1), "cuda", reps), "cuda", **common)
                    del Cp, Qp
                else:
                    R.add("index_build_prefix_copy", "one_time", bench(lambda: np.ascontiguousarray(C[:, :d]), "cpu", reps), "cpu", **common)
                    R.add("index_build_random_gather", "one_time", bench(lambda: C[:, idx], "cpu", reps), "cpu", **common)
                    Cp, Qp = np.ascontiguousarray(C[:, :d]), np.ascontiguousarray(Q[:, :d])
                    def search():
                        s = Qp @ Cp.T
                        np.argpartition(-s, topk, axis=1)[:, :topk]
                    R.add("search_top10", "repeat", bench(search, "cpu", reps), "cpu", **common)
                    del Cp, Qp
            if device == "cuda":
                del Cd, Qd
                torch.cuda.empty_cache()
        del C
    return R.rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ALL_MODEL_KEYS, choices=ALL_MODEL_KEYS)
    p.add_argument("--smoke", action="store_true", help="development: small sizes, few repeats; requires --out")
    p.add_argument("--skip-search", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="default results/timing")
    add_device_args(p)
    args = p.parse_args()
    if args.smoke and args.out is None:
        raise SystemExit("--smoke must not write to results/timing: pass --out <scratch dir>")
    dev = setup_device(args.require_gpu, args.device)
    out = (args.out or TIMING).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.skip_search:
        print("synthetic brute-force search", flush=True)
        rows += search_rows(dev["device"], args.smoke)
    for m in args.models:
        print(m, flush=True)
        rows += model_rows(m, dev["device"], args.smoke)
    meta = run_metadata(dev, script="scripts/timing_benchmark.py", protocol_id=PROTOCOL_ID, extra={
        "smoke": args.smoke, "cpu": platform.processor(), "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "note": "median/IQR over repeats after warm-up, CUDA synchronised; attribution is the repository's closed-form float64 "
                "code; search is fp32 matmul + top-10 on random unit vectors (64 queries); nothing else ran on the GPU"})
    save_table(pd.DataFrame(rows), out / "timing_benchmark.csv", meta)
    print(f"wrote {out / 'timing_benchmark.csv'}")


if __name__ == "__main__":
    main()
