"""Suite extraction: ALL hidden-state layers for the NanoBEIR retrieval corpora/queries and for the
12 MTEB classification tasks' texts (dev, test and the few-shot draws), for the six frozen encoders.

Cache (gitignored):
  results/cache/ret/{model}/{dataset}.corpus.npy|.queries.npy     float32 (n, n_hidden_states, 768), max_length 512
  results/cache/cls_all/{model}/{task}.emb.npy                    float32 (n_unique_texts, n_hidden_states, 768), max_length 128
  results/cache/ret/sentences/, results/cache/cls_all/sentences/  ids, texts, qrels, splits, few-shot draws
Same controlled config as the STS extraction (fp32, TF32 off, attention-mask mean pooling, revision-pinned models);
retrieval uses max_length 512 and length-sorted batches. The classification final layer is checked against
the sklearn-probe cache (results/cache/cls) for the 11 tasks both cover.

With --cls-full only the classification suite is extracted, with the FULL Hub test splits and the final layer only:
  results/cache/cls_full/{model}/{task}.emb.npy      float32 (n_unique_texts, 768), max_length 128, length-sorted batches
  results/cache/cls_full/sentences/                  same layout as cls_all; dev, pool and few-shot draws identical to cls_all
Existing caches are never touched; every text also present in cls_all is checked against its final layer (max abs diff < 1e-3).
Resumable: a finished task file is skipped.

Usage: python scripts/extract_retrieval_classification_embeddings.py [--suite ret cls] [--models ...] [--ret-datasets ...] [--cls-tasks ...] [--require-gpu]
       python scripts/extract_retrieval_classification_embeddings.py --cls-full [--models ...] [--cls-tasks ...] [--require-gpu]
"""

import argparse
import json
import time

import numpy as np

from fyp.classification import ALL_CLS_KEYS, CLS_KEYS, build_sentence_table
from fyp.models import MODEL_KEYS, ALL_MODEL_KEYS, cache_dir, load_encoder
from fyp.retrieval import RET_KEYS, RET_MAX_LENGTH, build_ret_table
from fyp.paths import CLS_ALL_LAYERS_CACHE as CLS_ALL, CLS_CACHE as CLS_FULL, CLS_FINAL_CACHE as CLS_A1, RET_CACHE as RET
from fyp.runtime import add_device_args, run_metadata, setup_device

RET_BATCH = 64
CLS_BATCH = 128
PROTOCOL = "retrieval_classification"


def extract_full_cls(args, dev):
    """Final layer, full test splits, new cache; validated against the old all-layer cache by text."""
    tabs = {k: build_sentence_table(k, CLS_FULL, full_test=True) for k in args.cls_tasks}
    for k, t in tabs.items():
        i = t["info"]
        print(f"  cls {k:22s} labels={i['n_labels']:3d} pool={i['n_pool']:6d} dev={i['n_dev']:5d} test={i['n_test']:6d} "
              f"(hub {i['n_test_hub']}) unique texts={i['n_unique_texts']:6d} fewshot_in_test={i['n_fewshot_texts_also_in_test']}", flush=True)
    for model_key in args.models:
        enc = load_encoder(model_key, dev["device"])
        layer = enc.spec.final_layer
        t0, per = time.time(), {}
        cache_dir(CLS_FULL, model_key).mkdir(parents=True, exist_ok=True)
        for key, tab in tabs.items():
            path = cache_dir(CLS_FULL, model_key) / f"{key}.emb.npy"
            texts = tab["texts"]
            try:                      # a half-written file fails to load or has the wrong shape and is redone
                done = path.exists() and np.load(path, mmap_mode="r").shape == (len(texts), 768)
            except Exception:
                done = False
            if done:
                print(f"  {model_key:15s} cls {key:22s} exists, skipped", flush=True)
                continue
            emb = enc.encode(texts, batch_size=CLS_BATCH, sort_by_length=True, pool_on_device=True, layers=[layer])[:, 0, :]
            assert emb.shape == (len(texts), 768) and np.isfinite(emb).all(), (model_key, key)
            old_path = cache_dir(CLS_ALL, model_key) / f"{key}.emb.npy"
            check = None
            if old_path.exists() and enc.spec.pooling == "mean":   # the comparison is meaningful for mean pooling only
                old_texts = json.loads((CLS_ALL / "sentences" / f"{key}.json").read_text())["texts"]
                new_row = {t: i for i, t in enumerate(texts)}
                pairs = [(i, new_row[t]) for i, t in enumerate(old_texts) if t in new_row]
                old = np.load(old_path, mmap_mode="r")
                oi, ni = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
                order = np.argsort(oi)
                check = float(np.abs(np.asarray(old[oi[order], layer, :]) - emb[ni[order]]).max())
                assert check < 1e-3, (model_key, key, check)
            np.save(path, emb)
            per[key] = {"n_texts": len(texts), "n_truncated_at_128": enc.count_truncated(texts),
                        "max_abs_diff_vs_cls_all_final_layer": check, "n_texts_compared": None if check is None else len(pairs)}
            print(f"  {model_key:15s} cls {key:22s} {emb.shape} maxdiff-vs-old={check} ({time.time() - t0:.0f}s)", flush=True)
        meta = run_metadata(dev, script="scripts/extract_retrieval_classification_embeddings.py --cls-full", protocol_id="retrieval_classification", extra={
            "model": enc.spec.__dict__, "dtype": "float32", "pooling": enc.spec.pooling, "cache_name": enc.spec.cache_dir_name, "layers": "final layer only",
            "cls_max_length": 128, "full_test_splits": True, "per_set": per, "seconds": round(time.time() - t0, 1)})
        (cache_dir(CLS_FULL, model_key) / "full_extract.meta.json").write_text(json.dumps(meta, indent=2, default=str))
        del enc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", nargs="+", default=["ret", "cls"], choices=["ret", "cls"])
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=ALL_MODEL_KEYS)
    p.add_argument("--ret-datasets", nargs="+", default=RET_KEYS, choices=RET_KEYS)
    p.add_argument("--cls-tasks", nargs="+", default=ALL_CLS_KEYS, choices=ALL_CLS_KEYS)
    p.add_argument("--cls-full", action="store_true", help="classification only, full test splits, final layer, new cache")
    add_device_args(p)
    args = p.parse_args()
    dev = setup_device(args.require_gpu, args.device)
    if args.cls_full:
        return extract_full_cls(args, dev)

    ret_tabs = {k: build_ret_table(k, RET) for k in args.ret_datasets} if "ret" in args.suite else {}
    cls_tabs = {k: build_sentence_table(k, CLS_ALL) for k in args.cls_tasks} if "cls" in args.suite else {}
    for k, t in ret_tabs.items():
        i = t["info"]
        print(f"  ret {k:15s} docs={i['n_docs']:5d} queries={i['n_queries']:3d} scored={i['n_scored_queries']:3d} "
              f"dev={i['n_dev']:2d} test={i['n_test']:3d} identical id/text pairs={i['n_identical_id_or_text_pairs']}")
    for k, t in cls_tabs.items():
        i = t["info"]
        print(f"  cls {k:22s} labels={i['n_labels']:3d} pool={i['n_pool']:6d} dev={i['n_dev']:5d} test={i['n_test']:5d} "
              f"unique texts={i['n_unique_texts']:6d} fewshot_in_test={i['n_fewshot_texts_also_in_test']}")

    for model_key in args.models:
        enc = load_encoder(model_key, dev["device"])
        t0, per = time.time(), {"ret": {}, "cls": {}}
        if ret_tabs:
            cache_dir(RET, model_key).mkdir(parents=True, exist_ok=True)
        if cls_tabs:
            cache_dir(CLS_ALL, model_key).mkdir(parents=True, exist_ok=True)
        for key, tab in ret_tabs.items():
            for part, texts in (("corpus", tab["corpus_texts"]), ("queries", tab["query_texts"])):
                n_trunc = enc.count_truncated(texts, RET_MAX_LENGTH)
                emb = enc.encode(texts, batch_size=RET_BATCH, max_length=RET_MAX_LENGTH, sort_by_length=True, pool_on_device=True)
                assert emb.shape == (len(texts), enc.num_layers, 768) and np.isfinite(emb).all(), (model_key, key, part)
                np.save(cache_dir(RET, model_key) / f"{key}.{part}.npy", emb)
                per["ret"][f"{key}.{part}"] = {"n": len(texts), "n_truncated_at_512": n_trunc}
            print(f"  {model_key:15s} ret {key:15s} done ({time.time() - t0:.0f}s)", flush=True)
        for key, tab in cls_tabs.items():
            texts = tab["texts"]
            n_trunc = enc.count_truncated(texts)
            emb = enc.encode(texts, batch_size=CLS_BATCH, pool_on_device=True)
            assert emb.shape == (len(texts), enc.num_layers, 768) and np.isfinite(emb).all(), (model_key, key)
            np.save(cache_dir(CLS_ALL, model_key) / f"{key}.emb.npy", emb)
            check = None
            if key in CLS_KEYS and enc.spec.pooling == "mean" and (CLS_A1 / model_key / f"{key}.emb.npy").exists():
                a1 = np.load(CLS_A1 / model_key / f"{key}.emb.npy")
                check = float(np.abs(a1 - emb[:, -1, :]).max())
                assert a1.shape == emb[:, -1, :].shape and check < 1e-3, (model_key, key, check)
            per["cls"][key] = {"n_texts": len(texts), "n_truncated_at_128": n_trunc, "max_abs_diff_final_layer_vs_A1": check}
            print(f"  {model_key:15s} cls {key:22s} {emb.shape} trunc={n_trunc} vs-A1 maxdiff={check} ({time.time() - t0:.0f}s)", flush=True)
        meta = run_metadata(dev, script="scripts/extract_retrieval_classification_embeddings.py", protocol_id=PROTOCOL, extra={
            "model": enc.spec.__dict__, "dtype": "float32", "pooling": enc.spec.pooling + " (all hidden states)", "cache_name": enc.spec.cache_dir_name,
            "ret_max_length": RET_MAX_LENGTH, "cls_max_length": 128, "per_set": per, "seconds": round(time.time() - t0, 1)})
        (cache_dir(RET, model_key) if ret_tabs else cache_dir(CLS_ALL, model_key)).joinpath("a2_extract.meta.json").write_text(
            json.dumps(meta, indent=2, default=str))
        del enc


if __name__ == "__main__":
    main()
