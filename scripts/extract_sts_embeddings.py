"""STS cache: controlled extraction of every model x every pair source (+ the NLI sample), all hidden states.

Cache layout (results/cache/, gitignored):
  sentences/{key}.json          unique-sentence table for one source (model-independent);
                                row r of every model's {key}.emb.npy encodes sentences[r]
  sentences/{key}.pairs.npz     idx1, idx2 (int64 rows into the table), score (float64);
                                pair i = (sentences[idx1[i]], sentences[idx2[i]], score[i])
  {model}/{key}.emb.npy         float32, shape (n_unique_sentences, n_hidden_states, 768)
  {model}/extract.meta.json     run metadata + per-source truncation counts
The NLI sample (key "nli20k") has a sentence table but no pairs file.

Usage: python scripts/extract_sts_embeddings.py [--models m1 m2 ...] [--require-gpu]
"""

import argparse
import json
import time

import numpy as np

from fyp.corpora import NLI_KEY, NLI_REV, NLI_SAMPLE_SEED, PAIR_SOURCES, dedupe_pairs, load_nli_sample, load_pairs
from fyp.models import BATCH_SIZE, MAX_LENGTH, MODEL_KEYS, ALL_MODEL_KEYS, cache_dir, load_encoder
from fyp.paths import CACHE
from fyp.runtime import add_device_args, run_metadata, setup_device


def build_sentence_tables() -> dict:
    """Write (once) and return {key: sentence list} for all pair sources + the NLI sample."""
    out_dir = CACHE / "sentences"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for key, src in PAIR_SOURCES.items():
        path = out_dir / f"{key}.json"
        if not path.exists():
            pairs = load_pairs(key)
            sents, idx1, idx2 = dedupe_pairs(pairs["sentence1"], pairs["sentence2"])
            path.write_text(json.dumps({"source": src.__dict__, "n_pairs": len(idx1), "sentences": sents}))
            np.savez(out_dir / f"{key}.pairs.npz", idx1=idx1, idx2=idx2, score=pairs["score"])
        tables[key] = json.loads(path.read_text())["sentences"]
    path = out_dir / f"{NLI_KEY}.json"
    if not path.exists():
        path.write_text(json.dumps({"source": {"hf_name": "sentence-transformers/all-nli", "config": "pair-class",
                                               "split": "train", "revision": NLI_REV, "seed": NLI_SAMPLE_SEED},
                                    "sentences": load_nli_sample()}))
    tables[NLI_KEY] = json.loads(path.read_text())["sentences"]
    return tables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=ALL_MODEL_KEYS)
    add_device_args(p)
    args = p.parse_args()
    dev = setup_device(args.require_gpu, args.device)

    tables = build_sentence_tables()
    for key, sents in tables.items():
        print(f"  source {key:12s} unique sentences = {len(sents)}")

    for model_key in args.models:
        enc = load_encoder(model_key, dev["device"])
        out_dir = cache_dir(CACHE, model_key)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_source = {}
        t0 = time.time()
        for key, sents in tables.items():
            emb = enc.encode(sents)
            assert emb.shape == (len(sents), enc.num_layers, 768), emb.shape
            assert np.isfinite(emb).all(), f"{model_key}/{key}: non-finite values"
            np.save(out_dir / f"{key}.emb.npy", emb)
            n_trunc = enc.count_truncated(sents)
            per_source[key] = {"n_sentences": len(sents), "n_truncated_at_max_length": n_trunc}
            print(f"  {model_key:15s} {key:12s} {emb.shape}  truncated>{MAX_LENGTH}: {n_trunc}")
        meta = run_metadata(dev, script="scripts/extract_sts_embeddings.py", protocol_id=None, extra={
            "model": enc.spec.__dict__, "dtype": "float32", "pooling": enc.spec.pooling + (" (mean: attention-mask mean) every hidden state"), "cache_name": enc.spec.cache_dir_name,
            "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "batch_policy": "input order, no length sorting",
            "inputs_passed": enc.input_names, "sources": per_source, "seconds": round(time.time() - t0, 1),
        })
        (out_dir / "extract.meta.json").write_text(json.dumps(meta, indent=2, default=str))
        del enc


if __name__ == "__main__":
    main()
