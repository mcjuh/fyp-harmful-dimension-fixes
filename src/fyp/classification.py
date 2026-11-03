"""MTEB classification tasks and the linear probe.

Protocol = MTEB's own classification protocol, so the numbers are comparable to the paper:
  - probe: L2 logistic regression on frozen final-layer mean-pooled embeddings
    (sklearn defaults, C=1, lbfgs, max_iter=100, no feature scaling)
  - training set: `N_EXPERIMENTS` (10) few-shot draws of `SAMPLES_PER_LABEL` (8) examples per label
    from the train pool; the same test set is scored after every draw
  - metrics: accuracy (main) and macro-F1

Dimension deletion for a linear probe: the kept columns are selected *before* fitting, so a
deleted coordinate never reaches the probe (identical to zeroing it, because L2 drives the
weights of an all-zero feature to exactly 0). Any centring/scaling would have to come first;
none is applied.

Splits per task (all model-independent, fixed by seeds below):
  test  <= TEST_CAP rows (seeded subsample of the Hub test split); the whole split with full_test=True
  dev   <= DEV_CAP rows: the Hub validation split if there is one, else a seeded carve of the
        train split. The carve is removed from the train pool in every mode. Dev is only for
        development (`--eval-split dev`); the eval lock applies to test.
  pool  the rest of the train split, from which the few-shot draws are taken
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

N_EXPERIMENTS = 10
SAMPLES_PER_LABEL = 8
TEST_CAP = 5000
DEV_CAP = 1000
MAX_ITER = 100
SPLIT_SEED = 31000      # test / dev subsampling and the dev carve (mixed with a task index)
FEWSHOT_SEED = 32000    # few-shot draw e uses default_rng([FEWSHOT_SEED, e])


@dataclass(frozen=True)
class ClsTask:
    key: str
    hf_name: str
    revision: str
    config: Optional[str]      # Hub config name, if any
    loader: str                # "hub" = load_dataset; "files" = fetch one file per split at the revision
    file_fmt: Optional[str]    # loader == "files": path template with {config} and {split}
    shards: Optional[dict] = None    # loader == "shards": {split: (parquet paths at the revision)}
    train_cap: Optional[int] = None  # seeded cap on the train rows read; None = all


CLS_TASKS: Dict[str, ClsTask] = {t.key: t for t in [
    ClsTask("amazon_counterfactual", "mteb/amazon_counterfactual", "b9867e02b01002698f60c07fb8efdce8c5020ee8", "en", "hub", None),
    ClsTask("amazon_reviews", "mteb/amazon_reviews_multi", "c379a6705fec24a2493fa68e011692605f44e119", "en", "files", "{config}/{split}.jsonl"),
    ClsTask("banking77", "mteb/banking77", "18072d2685ea682290f7b8924d94c62acc19c0b2", None, "hub", None),
    ClsTask("emotion", "mteb/emotion", "13535ec7ed83ac3920c40db3c3fd4133af55cc06", None, "hub", None),
    ClsTask("imdb", "mteb/imdb", "d05f0155defa7991dad75bc68c5ccb6774b1fdc5", None, "hub", None),
    ClsTask("massive_intent", "mteb/massive_intent", "feaaf04a19b242736c2d040393e7932a22a0435e", "en", "hub", None),
    ClsTask("massive_scenario", "mteb/massive_scenario", "380a57d5ddfb28f2d860450e99221b22d8111627", "en", "hub", None),
    ClsTask("mtop_domain", "mteb/mtop_domain", "b8451f34e224fe9b3b82ac712dba88dcdc9eb58f", "en", "files", "{config}/{split}-00000-of-00001.parquet"),
    ClsTask("mtop_intent", "mteb/mtop_intent", "9a200bf94b01afa0438a39c54f6a3aceaf78856f", "en", "files", "{config}/{split}-00000-of-00001.parquet"),
    ClsTask("toxic_conversations", "mteb/toxic_conversations_50k", "3969dd5d574dfe3a2427e3a544b64ee8dcaf9936", None, "hub", None),
    ClsTask("tweet_sentiment", "mteb/tweet_sentiment_extraction", "7261898ee3b9a739595e8dbf41df6b2332f429bb", None, "hub", None),
    # A2: the paper's 12th task. 3.6M-row train split -> seeded 100k sample; test capped at TEST_CAP as the others.
    ClsTask("amazon_polarity", "mteb/amazon_polarity", "ec149c1fe36043668a50804214d4597804001f6f", None, "shards", None,
            shards={"train": tuple(f"data/train-0000{i}-of-00004.parquet" for i in range(4)),
                    "test": ("data/test-00000-of-00001.parquet",)}, train_cap=100_000),
]}
ALL_CLS_KEYS = list(CLS_TASKS)       # the paper's 12 tasks (A2)
CLS_KEYS = ALL_CLS_KEYS[:11]         # the 11 A1 tasks: the defaults of the A1 scripts, which must not change
POOL_SEED = 34000                    # train sample for tasks with a train_cap


def _load_split(task: ClsTask, split: str):
    """(texts, raw_labels) for one Hub split, or None if the task has no such split."""
    if task.loader == "hub":
        from datasets import load_dataset
        try:
            ds = (load_dataset(task.hf_name, task.config, split=split, revision=task.revision) if task.config
                  else load_dataset(task.hf_name, split=split, revision=task.revision))
        except ValueError:      # split not present
            return None
        return list(ds["text"]), list(ds["label"])
    import pandas as pd
    from huggingface_hub import hf_hub_download
    if task.loader == "shards":
        if split not in task.shards:
            return None
        df = pd.concat([pd.read_parquet(hf_hub_download(task.hf_name, f, repo_type="dataset", revision=task.revision),
                                        columns=["text", "label"]) for f in task.shards[split]], ignore_index=True)
        if split == "train" and task.train_cap and len(df) > task.train_cap:
            rows = np.sort(np.random.default_rng([POOL_SEED, 0]).choice(len(df), size=task.train_cap, replace=False))
            df = df.iloc[rows]
        return list(df["text"]), list(df["label"])
    path = hf_hub_download(task.hf_name, task.file_fmt.format(config=task.config, split=split),
                           repo_type="dataset", revision=task.revision)
    df = pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_parquet(path)
    return list(df["text"]), list(df["label"])


def load_cls_task(key: str, full_test: bool = False) -> dict:
    """Deterministic split construction for one task. `full_test=True` keeps the whole Hub test split instead of the
    TEST_CAP subsample; the subsample is still drawn, so the random stream, and with it the dev split, the carve and the pool, do not depend on the cap. Returns
    {'test': (texts, y), 'dev': (texts, y), 'pool': (texts, y), 'labels': sorted raw labels, 'info': {...}}
    with integer labels y (index into 'labels')."""
    task = CLS_TASKS[key]
    tidx = ALL_CLS_KEYS.index(key)
    rng = np.random.default_rng([SPLIT_SEED, tidx])
    raw = {s: _load_split(task, s) for s in ("train", "validation", "test")}
    assert raw["train"] is not None and raw["test"] is not None, key
    labels = sorted({l for s in raw.values() if s is not None for l in s[1]}, key=str)
    to_int = {l: i for i, l in enumerate(labels)}

    def take(split_data, rows):
        texts, y = split_data
        return [texts[i] for i in rows], np.array([to_int[y[i]] for i in rows], dtype=np.int64)

    def cap(n, c):
        return np.sort(rng.choice(n, size=c, replace=False)) if n > c else np.arange(n)

    capped_rows = cap(len(raw["test"][0]), TEST_CAP)
    test = take(raw["test"], np.arange(len(raw["test"][0])) if full_test else capped_rows)
    n_train = len(raw["train"][0])
    if raw["validation"] is not None:
        dev = take(raw["validation"], cap(len(raw["validation"][0]), DEV_CAP))
        pool_rows = np.arange(n_train)
        dev_source = "validation split"
    else:
        carve = np.sort(rng.choice(n_train, size=DEV_CAP, replace=False))
        dev = take(raw["train"], carve)
        pool_rows = np.setdiff1d(np.arange(n_train), carve)
        dev_source = f"{DEV_CAP}-row seeded carve of train (removed from the pool)"
    pool = take(raw["train"], pool_rows)
    info = {"task": key, "hf_name": task.hf_name, "revision": task.revision, "config": task.config,
            "n_labels": len(labels), "n_test_hub": len(raw["test"][0]), "n_test": len(test[0]),
            "n_dev": len(dev[0]), "dev_source": dev_source, "n_pool": len(pool[0])}
    return {"test": test, "dev": dev, "pool": pool, "labels": [str(l) for l in labels], "info": info}


def few_shot_rows(y: np.ndarray, experiment: int, samples_per_label: int = SAMPLES_PER_LABEL) -> np.ndarray:
    """Rows of `y` for few-shot draw `experiment`: up to `samples_per_label` random rows per label
    (all of them if a label has fewer), in label order. Depends only on (y, experiment)."""
    rng = np.random.default_rng([FEWSHOT_SEED, experiment])
    rows: List[np.ndarray] = []
    for lab in np.unique(y):
        cand = np.flatnonzero(y == lab)
        rows.append(rng.permutation(cand)[:samples_per_label])
    return np.concatenate(rows)


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray,
              keep: Optional[np.ndarray] = None, max_iter: int = MAX_ITER) -> dict:
    """Fit the probe on the kept coordinates only and score on the eval set.
    `keep` = indices of coordinates that are NOT deleted (None = all)."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if keep is not None:
        x_train, x_eval = x_train[:, keep], x_eval[:, keep]
    clf = LogisticRegression(max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(np.asarray(x_train, dtype=np.float64), y_train)
    pred = clf.predict(np.asarray(x_eval, dtype=np.float64))
    return {"acc": float(accuracy_score(y_eval, pred)), "f1_macro": float(f1_score(y_eval, pred, average="macro")),
            "converged": bool(int(np.max(clf.n_iter_)) < max_iter)}


def kept_indices(method: str, n_remove: int, experiment: int, dim: int = 768) -> np.ndarray:
    """Coordinates that survive deleting `n_remove` of `dim` by `method`:
    'last' deletes the highest indices, 'first' the lowest, 'random' the first n of a seeded
    permutation (default_rng(experiment).permutation(dim), the truncation convention, so nested in K)."""
    if method == "last":
        return np.arange(dim - n_remove)
    if method == "first":
        return np.arange(n_remove, dim)
    if method == "random":
        return np.sort(np.random.default_rng(experiment).permutation(dim)[n_remove:])
    raise ValueError(method)


def build_sentence_table(key: str, cache_dir, full_test: bool = False) -> dict:
    """Unique-text table for one task (dev, test and the few-shot draws), written to
    `cache_dir/sentences/{key}.json|.npz`; returns {'info', 'texts'}. Idempotent (a written table is reused).
    Used by scripts/extract_retrieval_classification_embeddings.py."""
    from pathlib import Path
    out = Path(cache_dir) / "sentences"
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{key}.json").exists():
        return json.loads((out / f"{key}.json").read_text())
    d = load_cls_task(key, full_test)
    pool_texts, pool_y = d["pool"]
    draws = [few_shot_rows(pool_y, e) for e in range(N_EXPERIMENTS)]
    table: dict = {}

    def rows_of(texts):
        return np.array([table.setdefault(t, len(table)) for t in texts], dtype=np.int64)

    dev_rows, test_rows = rows_of(d["dev"][0]), rows_of(d["test"][0])
    train_rows = np.stack([rows_of([pool_texts[i] for i in r]) for r in draws])
    train_y = np.stack([pool_y[r] for r in draws])
    texts = list(table)
    test_set = set(d["test"][0])
    overlap = sorted({t for r in draws for t in (pool_texts[i] for i in r)} & test_set)
    info = {**d["info"], "n_unique_texts": len(texts), "few_shot_rows_per_experiment": int(train_rows.shape[1]),
            "n_experiments": N_EXPERIMENTS, "samples_per_label": SAMPLES_PER_LABEL, "labels": d["labels"],
            "test_class_counts": np.bincount(d["test"][1], minlength=len(d["labels"])).tolist(),
            "n_fewshot_texts_also_in_test": len(overlap)}
    np.savez(out / f"{key}.npz", dev_rows=dev_rows, dev_y=d["dev"][1], test_rows=test_rows, test_y=d["test"][1],
             train_rows=train_rows, train_y=train_y)
    (out / f"{key}.json").write_text(json.dumps({"info": info, "texts": texts}))
    return {"info": info, "texts": texts}


def load_table(cache_dir, key: str) -> dict:
    """Read the model-independent cache for one task written by scripts/extract_retrieval_classification_embeddings.py."""
    meta = json.loads((cache_dir / "sentences" / f"{key}.json").read_text())
    arr = np.load(cache_dir / "sentences" / f"{key}.npz")
    return {"texts": meta["texts"], "info": meta["info"], **{k: arr[k] for k in arr.files}}
