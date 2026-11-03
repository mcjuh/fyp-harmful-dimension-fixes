"""The evaluator-independent analysis helpers shared by the STS pipeline and the retrieval/classification suite."""

import numpy as np

from fyp.ablation import ranked_order
from fyp.bootstrap import bootstrap_weights, resample_counts
from fyp.dimension_stats import (estimates_table, headline_estimates, lag1_sign_autocorr, outlier_sets, pca_estimates,
                                 removal_curve_specs, score_summary, shared_dims_table)
from fyp.protocol import N_DIMS, N_GRID, SEED_TIE, SEEDS_OUTLIER


def test_resample_counts_is_the_multinomial_stream_the_suite_ci_were_computed_with():
    n = 57
    old = np.random.default_rng(9003).multinomial(n, np.full(n, 1.0 / n), size=20)     # the expression before the refactor
    got = resample_counts(n, 20, 9003)
    assert np.array_equal(got, old) and (got.sum(1) == n).all()
    # same distribution as the pair bootstrap's weights (each row sums to n), but not the same random stream
    assert (bootstrap_weights(n, 20, 9003).sum(1) == n).all()
    assert not np.array_equal(got, bootstrap_weights(n, 20, 9003))


def test_lag1_sign_autocorr_extremes():
    assert lag1_sign_autocorr(np.tile([1.0, -1.0], 384)) < -0.99          # alternating signs along the index
    assert lag1_sign_autocorr(np.r_[np.ones(384), -np.ones(384)]) > 0.99  # two long runs


def test_score_summary_counts_blocks_and_neutral():
    rng = np.random.default_rng(0)
    score = rng.normal(size=N_DIMS)
    score[:5] = 0.0
    s = score_summary(score)
    assert s["n_degrading"] == int((score > 0).sum()) and s["n_informative"] == int((score < 0).sum()) and s["n_neutral"] == 5
    assert s["n_degrading"] + s["n_informative"] + s["n_neutral"] == N_DIMS
    assert s["degrading_frac_block_min"] <= s["degrading_frac_block_max"] and 0 < s["lag1_perm_p_two_sided"] <= 1
    assert score_summary(score) == s                                       # seeded: deterministic


def test_shared_dims_table_observed_histogram_and_tail():
    deg = np.zeros((3, N_DIMS), dtype=bool)
    deg[0, :100] = deg[1, :60] = deg[2, :10] = True                       # dims 0-9 in 3 tasks, 10-59 in 2, 60-99 in 1
    t = shared_dims_table(deg, "m")
    assert t["observed"].tolist() == [N_DIMS - 100, 40, 50, 10]
    assert t["observed_ge"].tolist() == [N_DIMS, 100, 60, 10]
    assert (t["null_ge_lo"] <= t["null_ge_hi"]).all() and t["p_ge_upper"].between(0, 1).all()
    assert (t["model"] == "m").all()


def test_outlier_sets_flags_injected_outlier_and_matches_sizes():
    mu = np.random.default_rng(1).normal(size=N_DIMS) * 0.1
    mu[308] = 5.0
    z, outl, sets = outlier_sets(mu)
    assert outl.tolist() == [308] and abs(z[308]) > 3
    assert set(sets) == {("outliers", -1)} | {("random", s) for s in SEEDS_OUTLIER}
    for (name, _), dims in sets.items():
        assert len(dims) == 1 and (name == "outliers" or dims[0] != 308)
    assert outlier_sets(np.arange(N_DIMS, dtype=float))[2] == {}           # a linear ramp has max |z| = sqrt(3): nothing to remove


def test_removal_curve_specs_structure_and_anchors():
    score = np.random.default_rng(3).normal(size=N_DIMS)
    curves, best, worst, nD, nI = removal_curve_specs(score, range(1000, 1003), range(2000, 2002))
    assert nD == int((score > 0).sum()) and nI == int((score < 0).sum())
    assert np.array_equal(best, ranked_order(score, True, SEED_TIE)) and np.array_equal(worst, ranked_order(score, False, SEED_TIE))
    assert set(c for c, _ in curves) == {"best_first", "worst_first", "degrading_only_ranked", "informative_only_ranked", "degrading_only_random", "informative_only_random"}
    assert sum(c == "degrading_only_random" for c, _ in curves) == 3 and sum(c == "informative_only_random" for c, _ in curves) == 2
    order, ns = curves[("degrading_only_ranked", -1)]
    assert ns[0] == 0 and ns[-1] == nD and set(order) == set(np.flatnonzero(score > 0))            # the curve stops at |D|
    assert set(curves[("degrading_only_random", 1001)][0]) == set(order)                                    # random order within the same D
    assert set(N_GRID) <= set(curves[("best_first", -1)][1]) and nD in curves[("best_first", -1)][1]
    assert curves[("informative_only_ranked", -1)][1][-1] == nI


def test_headline_estimates_and_estimates_table_layout():
    rng = np.random.default_rng(4)
    names = ["last_25", "last_50"] + [f"{p}_{c}_{K}" for p in ("train_ranked", "held_out") for c in ("best", "worst") for K in (25, 50)] \
        + [f"random{s}_{K}" for s in range(3) for K in (25, 50)]
    boot = {nm: 1.0 + 0.01 * rng.normal(size=40) for nm in names}
    est = headline_estimates(lambda nm: boot[nm], 50, ["random0", "random1", "random2"], ("train_ranked", "held_out"))
    assert list(est)[:6] == ["random_mean", "last", "train_ranked_best", "train_ranked_worst", "held_out_best", "held_out_worst"]
    np.testing.assert_allclose(est["train_ranked_best_minus_random_mean"], boot["train_ranked_best_50"] - est["random_mean"])
    rows = estimates_table(est, {"model": "m", "K": 50}, {"scheme": "pair"})
    assert list(rows[0]) == ["model", "K", "quantity", "scheme", "boot_mean", "ci_lo", "ci_hi"]
    assert all(r["ci_lo"] <= r["boot_mean"] <= r["ci_hi"] for r in rows)


def test_pca_estimates_identities():
    full, cent, pca, rnd = (np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.9, 1.9]), np.array([0.95, 0.96]))
    est = pca_estimates(full, cent, pca, rnd)
    np.testing.assert_allclose(est["pca50_RP_raw"], pca / full)
    np.testing.assert_allclose(est["pca50_RP_centred"], pca / cent)
    np.testing.assert_allclose(est["centred_full_RP_raw"], cent / full)
    np.testing.assert_allclose(est["pca50_RP_raw_minus_random50"], pca / full - rnd)
