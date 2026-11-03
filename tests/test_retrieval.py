"""RetrievalEval: masked cosine == brute force, nDCG@10 == a plain-python reference, ids excluded, closed-form deltas."""

import math

import numpy as np
import torch

from fyp.retrieval import RetrievalEval

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _toy(seed=0, nq=12, nd=60, d=24):
    rng = np.random.default_rng(seed)
    Q, D = rng.normal(size=(nq, d)) + 0.5, rng.normal(size=(nd, d)) + 0.5
    rel = np.zeros((nq, nd), dtype=bool)
    for q in range(nq):
        rel[q, rng.choice(nd, size=rng.integers(1, 5), replace=False)] = True
    excl = np.zeros((nq, nd), dtype=bool)
    excl[0, np.flatnonzero(rel[0])[0]] = True            # query 0's first relevant doc is "the query itself"
    if rel[0].sum() == 1:                                # keep at least one retrievable relevant doc
        rel[0, 59 if not rel[0, 59] else 58] = True
    return Q, D, rel, excl


def _ref_ndcg(Q, D, rel, excl, keep):
    out = []
    for q in range(Q.shape[0]):
        a, B = Q[q][keep], D[:, keep]
        cos = (B @ a) / np.sqrt((a @ a) * (B * B).sum(1))
        cos[excl[q]] = -np.inf
        top = np.argsort(-cos, kind="stable")[:10]
        dcg = sum(rel[q, t] / math.log2(i + 2) for i, t in enumerate(top))
        n_rel = int((rel[q] & ~excl[q]).sum())
        idcg = sum(1 / math.log2(i + 2) for i in range(min(10, n_rel)))
        out.append(dcg / idcg)
    return np.array(out)


def test_per_query_matches_reference_for_random_masks():
    Q, D, rel, excl = _toy()
    ev = RetrievalEval(Q, D, rel, excl, device=DEV)
    rng = np.random.default_rng(1)
    keeps = np.stack([rng.random(24) > p for p in (0.0, 0.3, 0.5, 0.7)])
    got = ev.per_query(keeps)
    for i, k in enumerate(keeps):
        np.testing.assert_allclose(got[i], _ref_ndcg(Q, D, rel, excl, k), atol=1e-12)


def test_excluded_docs_are_never_retrieved():
    Q, D, rel, excl = _toy()
    ev_ex = RetrievalEval(Q, D, rel, excl, device=DEV)
    full = np.ones((1, 24), dtype=bool)
    np.testing.assert_allclose(ev_ex.per_query(full)[0], _ref_ndcg(Q, D, rel, excl, np.ones(24, bool)), atol=1e-12)


def test_single_deltas_match_brute_force_and_k0_anchor():
    Q, D, rel, excl = _toy(3)
    ev = RetrievalEval(Q, D, rel, excl, device=DEV)
    base = ev.base()
    assert abs(base - _ref_ndcg(Q, D, rel, excl, np.ones(24, bool)).mean()) < 1e-12
    deltas = ev.single_deltas()
    for dim in (0, 5, 23):
        keep = np.ones(24, bool)
        keep[dim] = False
        assert abs(deltas[dim] - (_ref_ndcg(Q, D, rel, excl, keep).mean() - base)) < 1e-12


def test_fidelity_is_one_for_full_embedding_and_below_one_when_truncated():
    Q, D, rel, excl = _toy(4)
    ev = RetrievalEval(Q, D, rel, excl, device=DEV)
    f = ev.fidelity(np.stack([np.ones(24, bool), np.arange(24) < 12]))
    assert abs(f[0] - 1.0) < 1e-12 and 0.0 < f[1] < 1.0


def test_batching_does_not_change_results():
    Q, D, rel, excl = _toy(5)
    keeps = np.random.default_rng(2).random((10, 24)) > 0.4
    a = RetrievalEval(Q, D, rel, excl, device=DEV, chunk=3).per_query(keeps)
    b = RetrievalEval(Q, D, rel, excl, device=DEV, chunk=64).per_query(keeps)
    np.testing.assert_allclose(a, b, atol=1e-12)
