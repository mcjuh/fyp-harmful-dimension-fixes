"""Suite analysis plumbing on tiny synthetic tasks: anchors, nesting, held_out excludes the held-out task, shapes."""

import numpy as np
import pytest
import torch

import fyp.suite_analysis as A
from fyp.suite import ClsTask, RetTask

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ret_task(seed):
    rng = np.random.default_rng(seed)
    Q, D = rng.normal(size=(8, 768)) + 0.3, rng.normal(size=(40, 768)) + 0.3
    rel = np.zeros((8, 40), bool)
    for q in range(8):
        rel[q, rng.choice(40, 2, replace=False)] = True
    return RetTask(Q, D, rel, None, device=DEV)


def _cls_task(seed, K=3, E=2):
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(K), 4)
    cen = rng.normal(size=(K, 768))
    Xtr = np.stack([cen[y] + rng.normal(size=(len(y), 768)) * 3 for _ in range(E)])
    ytr = np.stack([y] * E)
    ye = np.tile(np.arange(K), 10)
    Xe = cen[ye] + rng.normal(size=(len(ye), 768)) * 3
    return ClsTask(Xtr, ytr, Xe, ye, device=DEV)


def test_keep_from_order_is_nested_and_counts():
    order = np.random.default_rng(0).permutation(768)
    keep = A.keep_from_order(order, [0, 10, 384])
    assert keep.shape == (3, 768) and keep.sum(1).tolist() == [768, 758, 384]
    assert (keep[2] <= keep[1]).all() and (keep[1] <= keep[0]).all()


@pytest.mark.parametrize("make", [_ret_task, _cls_task])
def test_truncation_anchor_and_shapes(make):
    old = (A.RET_RANDOM, A.CLS_RANDOM)
    A.RET_RANDOM, A.CLS_RANDOM = 3, 3
    try:
        tasks = {"a": make(1), "b": make(2)}
        t1, t2 = A.run_truncation(tasks, "m")
        assert set(t1["method"]) == {"last", "first", "random"}
        for name, t in tasks.items():
            k0 = t1[(t1["task"] == name) & (t1["n"] == 0) & (t1["method"] == "last")]["metric"].iloc[0]
            assert abs(k0 - t.base()) < 1e-9
        assert t2["fidelity"].between(-1 - 1e-9, 1 + 1e-9).all()
        f0 = t2[(t2["n"] == 0)]["fidelity"]
        assert np.allclose(f0, 1.0)                      # full embedding agrees with itself
    finally:
        A.RET_RANDOM, A.CLS_RANDOM = old


def test_units_lodo_excludes_heldout_and_curves_anchor():
    old = (A.RET_RANDOM, A.CLS_RANDOM)
    A.RET_RANDOM, A.CLS_RANDOM = 2, 2
    try:
        tasks = {"a": _ret_task(1), "b": _ret_task(2), "c": _ret_task(3)}
        deltas = {k: t.single_deltas() for k, t in tasks.items()}
        units = A.make_units(deltas)
        np.testing.assert_allclose(units[("in_sample", "all")][0], np.mean([deltas[k] for k in tasks], 0))
        np.testing.assert_allclose(units[("held_out", "a")][0], (deltas["b"] + deltas["c"]) / 2)
        assert units[("held_out", "a")][1] == ["a"]
        t4 = A.run_removal_curves(tasks, units, "m")
        bf = t4[(t4["curve"] == "best_first") & (t4["n"] == 0) & (t4["protocol"] == "in_sample")]
        for _, r in bf.iterrows():
            assert abs(r["metric"] - tasks[r["task"]].base()) < 1e-9
        degrading_only = t4[(t4["curve"] == "degrading_only_ranked") & (t4["protocol"] == "in_sample")]
        assert degrading_only["n"].max() == int((units[("in_sample", "all")][0] > 0).sum())      # the curve stops at |D|
        ci = A.run_headline_ci(tasks, units, "m")
        assert set(ci["K"]) == {25, 50} and (ci["ci_lo"] <= ci["ci_hi"]).all()
    finally:
        A.RET_RANDOM, A.CLS_RANDOM = old


def test_cls_task_metric_matches_direct_fit_and_paired_exps():
    t = _cls_task(5)
    keep = np.random.default_rng(0).random((3, 768)) > 0.5
    m_all = t.metric(keep)                               # mean over the 2 experiments
    m0, m1 = t.metric(keep, np.zeros(3, int)), t.metric(keep, np.ones(3, int))
    np.testing.assert_allclose(m_all, (m0 + m1) / 2)
    from fyp.probe_gpu import LogRegGPU
    p = LogRegGPU(t.probes[0].X.cpu().numpy() + t.probes[0].mu.cpu().numpy(), t.probes[0].classes[t.probes[0].y_int], DEV)
    st = p.fit(keep[:1], tol=1e-6)
    acc = p.correct(st, t.Xev, t.yev.cpu().numpy()).double().mean().item()
    assert abs(acc - m0[0]) < 0.02                       # sweep tol 1e-4 vs exact: at most a couple of flipped items
