"""Verify the STS embedding cache (results/cache/).

Per model:
  - every source file: shape (n_sentences, n_hidden_states, 768), float32, all finite
  - fresh GPU re-extraction of 200 random cached sentences vs the cache  (max |diff| < 1e-4)
  - CPU extraction of the same 200 sentences vs GPU                       (reported)
sbert-nli only: layer-final mean pooling vs sentence-transformers encode(), cosine > 0.9999.
Writes results/audit/cache_verify.json and results/audit/cache_manifest.json (SHA256 of every STS cache file: the model
directories and sentences/, not the classification/retrieval caches cls/, cls_all/, ret/).

Usage: python scripts/check_cache.py
"""

import argparse
import hashlib
import json

import numpy as np

from fyp.models import MODEL_KEYS, MODEL_SPECS, load_encoder
from fyp.models import cache_dir
from fyp.paths import AUDIT, CACHE
from fyp.protocol import SEED_BASE_SAMPLE as SEED
from fyp.runtime import add_device_args, run_metadata, setup_device, write_metadata

N_SAMPLE = 200
TOL = 1e-4
SUITE_CACHE_DIRS = ("cls", "cls_all", "ret")    # other pipelines' caches under results/cache/


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    add_device_args(p)
    args = p.parse_args()
    dev = setup_device(args.require_gpu, args.device)
    assert dev["device"] == "cuda", "GPU-vs-CPU comparison needs the GPU"

    tables = {f.stem: json.loads(f.read_text())["sentences"] for f in (CACHE / "sentences").glob("*.json")}
    rng = np.random.default_rng(SEED)
    keys = sorted(tables)
    sample = []
    for _ in range(N_SAMPLE):
        k = keys[rng.integers(len(keys))]
        sample.append((k, int(rng.integers(len(tables[k])))))
    sample_sents = [tables[k][r] for k, r in sample]

    report = {"n_sample": N_SAMPLE, "seed": SEED, "models": {}}
    for model_key in MODEL_KEYS:
        spec = MODEL_SPECS[model_key]
        entry = {"files": {}}
        for k in keys:
            arr = np.load(cache_dir(CACHE, model_key) / f"{k}.emb.npy", mmap_mode="r")
            ok_shape = arr.shape == (len(tables[k]), spec.num_hidden_states, 768)
            entry["files"][k] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "shape_ok": ok_shape,
                                 "all_finite": bool(np.isfinite(np.asarray(arr)).all())}
        cached = np.stack([np.load(cache_dir(CACHE, model_key) / f"{k}.emb.npy", mmap_mode="r")[r] for k, r in sample])

        gpu = load_encoder(model_key, "cuda").encode(sample_sents)
        cpu = load_encoder(model_key, "cpu").encode(sample_sents)
        d_fresh = np.abs(gpu.astype(np.float64) - cached).max()
        d_cpu = np.abs(gpu.astype(np.float64) - cpu).max(axis=(0, 2))
        entry["reextract_gpu_vs_cache_max_abs_diff"] = float(d_fresh)
        entry["reextract_pass"] = bool(d_fresh < TOL)
        entry["gpu_vs_cpu_max_abs_diff"] = float(d_cpu.max())
        entry["gpu_vs_cpu_max_abs_diff_per_layer"] = d_cpu.tolist()
        entry["max_abs_value"] = float(np.abs(cached).max())

        if model_key == "sbert-nli":
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(spec.hf_name, revision=spec.revision, device="cuda")
            st_emb = st.encode(sample_sents, batch_size=128, convert_to_numpy=True).astype(np.float64)
            ours = gpu[:, -1].astype(np.float64)
            cos = (st_emb * ours).sum(1) / (np.linalg.norm(st_emb, axis=1) * np.linalg.norm(ours, axis=1))
            entry["sbert_encode_vs_ours"] = {"st_max_seq_length": st.max_seq_length,
                                             "st_modules": [type(m).__name__ for m in st],
                                             "min_cosine": float(cos.min()), "pass_0.9999": bool(cos.min() > 0.9999)}
        report["models"][model_key] = entry
        print(f"{model_key:15s} files_ok={all(f['shape_ok'] and f['all_finite'] for f in entry['files'].values())} "
              f"fresh-vs-cache={d_fresh:.2e} gpu-vs-cpu={d_cpu.max():.2e} "
              + (f"sbert min cos={entry['sbert_encode_vs_ours']['min_cosine']:.7f}" if "sbert_encode_vs_ours" in entry else ""))

    out = AUDIT / "cache_verify.json"
    out.write_text(json.dumps(report, indent=2))
    write_metadata(out, run_metadata(dev, script="scripts/check_cache.py", protocol_id=None, seeds={"sample": SEED}))

    manifest = {str(f.relative_to(CACHE)).replace("\\", "/"): {"sha256": sha256(f), "bytes": f.stat().st_size}
                for f in sorted(CACHE.rglob("*")) if f.is_file() and f.relative_to(CACHE).parts[0] not in SUITE_CACHE_DIRS}
    (AUDIT / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {out} and {AUDIT / 'cache_manifest.json'} ({len(manifest)} files, "
          f"{sum(v['bytes'] for v in manifest.values()) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
