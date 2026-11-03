"""NanoBEIR retrieval.

13 datasets from `sentence-transformers/NanoBEIR-en` at a pinned revision. Per dataset: ~50 queries, 2.2k-5.7k
documents (title is not a separate field). Binary relevance. Metric: nDCG@10 with cosine similarity, per
query, macro-averaged over queries; BEIR's `ignore_identical_ids` convention (a document whose id equals the
query id is never retrieved; matters for Quora/ArguAna).

Dev/test split (eval lock): per dataset a seeded 20% of queries is dev, the rest test. Corpora are shared.

Cache (results/cache/ret/, gitignored):
  sentences/{key}.json   {"info", "corpus_ids", "corpus_texts", "query_ids", "query_texts"}
  sentences/{key}.npz    qrel_q, qrel_d (row pairs), excl_q, excl_d (identical-id pairs), dev_q, test_q (query rows)
  {model}/{key}.corpus.npy, {key}.queries.npy   float32 (n, n_hidden_states, 768)
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

RET_REPO = "sentence-transformers/NanoBEIR-en"
RET_REVISION = "beb106fbcfaa599c508c667041bf8c85fd78736b"
RET_KEYS = ["arguana", "climatefever", "dbpedia", "fever", "fiqa2018", "hotpotqa", "msmarco", "nfcorpus", "nq",
            "quoraretrieval", "scidocs", "scifact", "touche2020"]
_HUB_NAMES = {"arguana": "ArguAna", "climatefever": "ClimateFEVER", "dbpedia": "DBPedia", "fever": "FEVER",
              "fiqa2018": "FiQA2018", "hotpotqa": "HotpotQA", "msmarco": "MSMARCO", "nfcorpus": "NFCorpus", "nq": "NQ",
              "quoraretrieval": "QuoraRetrieval", "scidocs": "SCIDOCS", "scifact": "SciFact", "touche2020": "Touche2020"}
RET_MAX_LENGTH = 512
DEV_FRACTION = 0.2
SPLIT_SEED = 33000          # default_rng([SPLIT_SEED, dataset_index])
TOPK = 10


def _read(key: str, part: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(RET_REPO, f"{part}/Nano{_HUB_NAMES[key]}-00000-of-00001.parquet", repo_type="dataset",
                           revision=RET_REVISION)
    return pd.read_parquet(path)


def build_ret_table(key: str, cache_dir) -> dict:
    """Load one dataset, build ids/texts/qrels/split arrays, write them under cache_dir/sentences, return the meta."""
    out = Path(cache_dir) / "sentences"
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{key}.json").exists():
        return json.loads((out / f"{key}.json").read_text())
    corpus, queries, qrels = _read(key, "corpus"), _read(key, "queries"), _read(key, "qrels")
    cid, qid = [str(x) for x in corpus["_id"]], [str(x) for x in queries["_id"]]
    c_row, q_row = {c: i for i, c in enumerate(cid)}, {q: i for i, q in enumerate(qid)}
    pairs = [(q_row[str(q)], c_row[str(c)]) for q, c in zip(qrels["query-id"], qrels["corpus-id"])
             if str(q) in q_row and str(c) in c_row]
    qrel_q, qrel_d = (np.array(v, dtype=np.int64) for v in zip(*pairs))
    # BEIR's ignore_identical_ids, extended to identical TEXT: NanoBEIR's ids do not line up with BEIR's, and e.g.
    # 21 of 50 ArguAna queries sit verbatim in the corpus (non-relevant) and would rank first for every model.
    norm = lambda t: " ".join(str(t).casefold().split())
    by_text: Dict[str, List[int]] = {}
    for i, t in enumerate(corpus["text"]):
        by_text.setdefault(norm(t), []).append(i)
    ident_set = {(q_row[q], c_row[q]) for q in qid if q in c_row}
    for qi, t in enumerate(queries["text"]):
        ident_set.update((qi, ci) for ci in by_text.get(norm(t), []))
    ident = sorted(ident_set)
    excl_q, excl_d = (np.array(v, dtype=np.int64) for v in zip(*ident)) if ident else (np.zeros(0, np.int64),) * 2
    # queries with at least one retrievable relevant document (BEIR scores only those)
    excluded = set(zip(excl_q.tolist(), excl_d.tolist()))
    has_rel = np.zeros(len(qid), dtype=bool)
    for q, d in zip(qrel_q.tolist(), qrel_d.tolist()):
        if (q, d) not in excluded:
            has_rel[q] = True
    valid = np.flatnonzero(has_rel)
    rng = np.random.default_rng([SPLIT_SEED, RET_KEYS.index(key)])
    n_dev = int(round(DEV_FRACTION * len(valid)))
    dev_q = np.sort(rng.choice(valid, size=n_dev, replace=False))
    test_q = np.setdiff1d(valid, dev_q)
    np.savez(out / f"{key}.npz", qrel_q=qrel_q, qrel_d=qrel_d, excl_q=excl_q, excl_d=excl_d, dev_q=dev_q, test_q=test_q)
    info = {"key": key, "hub_name": _HUB_NAMES[key], "revision": RET_REVISION, "n_docs": len(cid), "n_queries": len(qid),
            "n_scored_queries": int(len(valid)), "n_dev": int(len(dev_q)), "n_test": int(len(test_q)),
            "n_qrels": int(len(qrel_q)), "n_identical_id_or_text_pairs": int(len(excl_q))}
    meta = {"info": info, "corpus_ids": cid, "corpus_texts": [str(t) for t in corpus["text"]],
            "query_ids": qid, "query_texts": [str(t) for t in queries["text"]]}
    (out / f"{key}.json").write_text(json.dumps(meta))
    return meta


def load_ret_table(cache_dir, key: str) -> dict:
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "sentences" / f"{key}.json").read_text())
    arr = np.load(cache_dir / "sentences" / f"{key}.npz")
    return {**meta, **{k: arr[k] for k in arr.files}}


class RetrievalEval:
    """nDCG@10 of cosine retrieval on a set of queries, under arbitrary coordinate-deletion masks (float64, GPU).

    Q (nq, 768) and D (nd, 768) are float arrays; `rel` (nq, nd) bool; `excl` (nq, nd) bool or None.
    Queries with no retrievable relevant document must already be removed. A deletion mask is the boolean
    KEEP vector (True = kept)."""

    def __init__(self, Q, D, rel, excl=None, device: str = "cuda", chunk: int = 96):
        self.device = device
        self.Q = torch.as_tensor(np.asarray(Q), dtype=torch.float64, device=device)
        self.D = torch.as_tensor(np.asarray(D), dtype=torch.float64, device=device)
        self.rel = torch.as_tensor(np.asarray(rel), dtype=torch.float64, device=device)
        self.excl = None if excl is None or not np.any(excl) else torch.as_tensor(np.asarray(excl), device=device)
        self.nq, self.nd = self.Q.shape[0], self.D.shape[0]
        self.Q2, self.D2 = self.Q ** 2, self.D ** 2
        n_rel = (self.rel if self.excl is None else self.rel * (~self.excl)).sum(1).long()
        assert bool((n_rel > 0).all()), "queries without a retrievable relevant document must be dropped"
        disc = 1.0 / torch.log2(torch.arange(TOPK, device=device, dtype=torch.float64) + 2.0)
        self.disc = disc
        self.idcg = torch.stack([disc[: min(TOPK, int(n))].sum() for n in n_rel])
        self.chunk = chunk
        self.n_dims = self.Q.shape[1]

    def _cos(self, keep):
        """(b, nq, nd) cosine under each keep mask, dot / sqrt(|q|^2 |d|^2)."""
        m = keep.to(torch.float64)
        dot = torch.matmul(self.Q[None] * m[:, None, :], self.D.T)
        qn, dn = (self.Q2 @ m.T).T, (self.D2 @ m.T).T                       # (b, nq), (b, nd)
        return dot / torch.sqrt((qn[:, :, None] * dn[:, None, :]).clamp_min(1e-300))

    def _ndcg_from_scores(self, cos):
        if self.excl is not None:
            cos = cos.masked_fill(self.excl[None], float("-inf"))
        top = cos.topk(TOPK, dim=-1).indices                                 # (b, nq, 10)
        gains = torch.gather(self.rel[None].expand(cos.shape[0], -1, -1), 2, top)
        return (gains * self.disc).sum(-1) / self.idcg                       # (b, nq)

    def per_query(self, keep) -> np.ndarray:
        """(B, nq) per-query nDCG@10 for B keep masks."""
        keep = torch.as_tensor(np.asarray(keep), device=self.device)
        out = [self._ndcg_from_scores(self._cos(keep[s:s + self.chunk])) for s in range(0, len(keep), self.chunk)]
        return torch.cat(out).cpu().numpy()

    def metric(self, keep) -> np.ndarray:
        return self.per_query(keep).mean(1)

    def base(self) -> float:
        return float(self.metric(np.ones((1, self.n_dims), dtype=bool))[0])

    def single_deltas(self) -> np.ndarray:
        """(768,) nDCG@10 without coordinate d minus the full-embedding nDCG@10."""
        keep = ~np.eye(self.n_dims, dtype=bool)
        return self.metric(keep) - self.base()

    def fidelity(self, keep) -> np.ndarray:
        """(B,) mean over queries of the Spearman correlation between the full-embedding document scores and the
        truncated-embedding document scores (paper App. A.3)."""
        keep = torch.as_tensor(np.asarray(keep), device=self.device)
        ones = torch.ones(1, self.n_dims, dtype=torch.bool, device=self.device)

        def ranks(x):                                                        # (..., nd) -> centred unit-norm ranks
            r = x.argsort(-1).argsort(-1).to(torch.float64)
            r = r - r.mean(-1, keepdim=True)
            return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-300)

        r_full = ranks(self._cos(ones))[0]                                   # (nq, nd)
        out = []
        for s in range(0, len(keep), max(1, self.chunk // 4)):
            r = ranks(self._cos(keep[s:s + max(1, self.chunk // 4)]))
            out.append((r * r_full[None]).sum(-1).mean(-1))
        return torch.cat(out).cpu().numpy()


def make_eval(table: dict, q_emb: np.ndarray, d_emb: np.ndarray, split: str, device: str = "cuda") -> RetrievalEval:
    """RetrievalEval for the 'dev' or 'test' queries of one dataset. q_emb/d_emb: (n, 768) at the chosen layer."""
    rows = table[f"{split}_q"]
    pos = {r: i for i, r in enumerate(rows.tolist())}
    nq, nd = len(rows), d_emb.shape[0]
    rel = np.zeros((nq, nd), dtype=bool)
    for q, d in zip(table["qrel_q"].tolist(), table["qrel_d"].tolist()):
        if q in pos:
            rel[pos[q], d] = True
    excl = np.zeros((nq, nd), dtype=bool)
    for q, d in zip(table["excl_q"].tolist(), table["excl_d"].tolist()):
        if q in pos:
            excl[pos[q], d] = True
    return RetrievalEval(q_emb[rows], d_emb, rel, excl, device=device)
