"""Closed-form deletion vs brute force, K=0 anchors, bootstrap exactness."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from fyp.ablation import TaskData, brute_force_rho as _brute_rho, ranked_order
from fyp.bootstrap import bootstrap_spearman, bootstrap_spearman_torch, bootstrap_weights
from fyp.protocol import N_DIMS, n_remove


@pytest.fixture
def task():
    rng = np.random.default_rng(0)
    n, d = 300, 32
    a = rng.normal(size=(n, d)) + 0.5
    b = a + rng.normal(scale=1.5, size=(n, d))
    gold = np.round(rng.uniform(0, 5, size=n), 1)  # coarse -> many ties, like STS gold
    return a, b, gold, TaskData.from_pairs("t", a, b, gold)


def test_full_rho_matches_scipy(task):
    a, b, gold, t = task
    assert abs(t.rho_full() - _brute_rho(a, b, gold, [])) < 1e-12


def test_single_deltas_match_brute_force(task):
    a, b, gold, t = task
    base = _brute_rho(a, b, gold, [])
    deltas = t.single_deltas()
    for d in range(a.shape[1]):
        assert abs(deltas[d] - (_brute_rho(a, b, gold, [d]) - base)) < 1e-12


def test_nested_curve_matches_brute_force_and_k0_is_baseline(task):
    a, b, gold, t = task
    order = np.random.default_rng(1).permutation(a.shape[1])
    # Not 31: with a single coordinate left every cosine is exactly +-1 under brute force but
    # +-(1-eps) in closed form, so massive ties rank differently. The protocol's deepest point
    # (K=90%) leaves 77 coordinates; sts_truncation_and_ranked_removal.py also brute-force checks real data there.
    ns = [0, 1, 5, 16, 28]
    got = t.rho_nested(order, ns)
    for n, r in zip(ns, got):
        assert abs(r - _brute_rho(a, b, gold, order[:n])) < 1e-12
    assert got[0] == pytest.approx(t.rho_full(), abs=1e-15)


def test_subset_equals_recomputed(task):
    a, b, gold, t = task
    mask = np.arange(len(gold)) % 3 != 0
    assert abs(t.subset(mask).rho_full() - _brute_rho(a[mask], b[mask], gold[mask], [])) < 1e-12


def test_fidelity_is_one_at_k0(task):
    _, _, _, t = task
    assert t.fidelity_nested(np.arange(32), [0])[0] == pytest.approx(1.0, abs=1e-12)


def test_n_remove_grid():
    assert [n_remove(k) for k in (0, 5, 25, 50, 90)] == [0, 39, 192, 384, 692]
    assert n_remove(100) == N_DIMS


def test_n_remove_is_the_authors_keep_rule():
    """Kept = int(768 * scale) with scale the kept fraction. Kept counts at the eight grid points where the old round() rule differed
    (old / authors'): K = 5: 730 / 729, 15: 653 / 652, 30: 538 / 537, 40: 461 / 460, 55: 346 / 345, 65: 269 / 268, 80: 154 / 153, 90: 77 / 76."""
    authors = {5: 729, 15: 652, 30: 537, 40: 460, 55: 345, 65: 268, 80: 153, 90: 76}
    assert {k: N_DIMS - n_remove(k) for k in authors} == authors
    for k in range(0, 100, 5):
        assert N_DIMS - n_remove(k) == int(N_DIMS * ((100 - k) / 100))      # their expression, for every grid point
        assert n_remove(k) <= n_remove(k + 5)                              # monotone
    assert N_DIMS - n_remove(50) == int(768 * 0.5) == 384


def test_ranked_order_ties_not_by_index():
    score = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
    o = ranked_order(score, descending=True, tie_seed=7)
    assert o[0] == 1 and o[-1] == 4
    assert set(o[1:5]) == {0, 2, 3, 5}
    assert list(o[1:5]) != [0, 2, 3, 5] or list(ranked_order(score, True, 8)[1:5]) != [0, 2, 3, 5]
    assert list(ranked_order(score, descending=False, tie_seed=7))[0] == 4


def test_identical_sentence_pairs_are_exact_ties(task):
    """a == b (identical-sentence pairs, e.g. 79 in STS12) must give cos exactly 1.0 after any
    deletion, so they tie under average ranks."""
    a, _, gold, _ = task
    t = TaskData.from_pairs("same", a, a.copy(), gold)
    assert np.all(t.cos_full() == 1.0)
    assert np.all(t.cos_single_deletions() == 1.0)
    assert np.all(t.cos_nested(np.random.default_rng(3).permutation(32), [1, 10, 28]) == 1.0)


def test_bootstrap_weights_equal_explicit_resampling(task):
    a, b, gold, t = task
    cos = np.stack([t.cos_full(), t.cos_nested(np.arange(32), [10])[:, 0]], axis=1)
    W = bootstrap_weights(len(gold), 20, seed=9000)
    got = bootstrap_spearman(cos, gold, W)
    rng = np.random.default_rng(9000)
    idx = rng.integers(0, len(gold), size=(20, len(gold)))
    for bi in range(20):
        for j in range(2):
            assert abs(got[bi, j] - spearmanr(cos[idx[bi], j], gold[idx[bi]]).statistic) < 1e-12


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if __import__("torch").cuda.is_available() else []))
def test_bootstrap_torch_equals_numpy(task, device):
    """Batched torch path (bootstrap of per-dimension scores, PCA CI) == numpy weighted-rank path, including tied predictions."""
    a, b, gold, t = task
    cos = np.concatenate([t.cos_full()[:, None], t.cos_single_deletions()], axis=1)
    cos[:40, 3] = np.round(cos[:40, 3], 1)  # force tie groups inside one prediction column
    W = bootstrap_weights(len(gold), 50, seed=5000)
    ref = bootstrap_spearman(cos, gold, W)
    got = bootstrap_spearman_torch(cos, gold, W, device=device, chunk=7)
    assert np.max(np.abs(got - ref)) < 1e-12


def test_cluster_weights_reduce_to_pair_bootstrap_without_sharing():
    from fyp.bootstrap import cluster_bootstrap_weights
    assert np.array_equal(cluster_bootstrap_weights(np.arange(57), 30, seed=11000), bootstrap_weights(57, 30, seed=11000))


def test_pair_components_and_cluster_weights_tie_shared_pairs():
    from fyp.bootstrap import cluster_bootstrap_weights, pair_components
    # sentences 0-1-2 chained by pairs 0,1; pair 2 alone; pairs 3,4 share sentence 5
    u, v = np.array([0, 1, 3, 5, 6]), np.array([1, 2, 4, 7, 5])
    comp = pair_components(u, v)
    assert comp[0] == comp[1] and comp[3] == comp[4] and len({comp[0], comp[2], comp[3]}) == 3
    W = cluster_bootstrap_weights(comp, 200, seed=1)
    assert np.array_equal(W[:, 0], W[:, 1]) and np.array_equal(W[:, 3], W[:, 4])
    assert np.array_equal(W.sum(1) - W[:, 1] - W[:, 4], np.full(200, 3.0))  # 3 clusters drawn per replicate


def test_rotation_preserves_cosine_and_is_orthogonal():
    from fyp.ablation import haar_orthogonal
    Q = haar_orthogonal(32, seed=6000)
    assert np.max(np.abs(Q.T @ Q - np.eye(32))) < 1e-12
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(50, 32)), rng.normal(size=(50, 32))
    t, tr = TaskData.from_pairs("t", a, b, rng.uniform(size=50)), TaskData.from_pairs("r", a @ Q, b @ Q, rng.uniform(size=50))
    assert np.max(np.abs(t.cos_full() - tr.cos_full())) < 1e-12
    assert not np.allclose(haar_orthogonal(32, seed=6001), Q)
