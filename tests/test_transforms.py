"""Transforms: fitted from the fit sample alone, and each one does what its definition says."""

import numpy as np
import pytest

from fyp import transforms as T
from fyp.ablation import TaskData


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(768, 768)) * np.linspace(2.0, 0.05, 768)       # unequal variances
    return rng.normal(size=(3000, 768)) @ A * 0.1 + rng.normal(size=768)  # shared non-zero mean


def test_fit_is_independent_of_the_evaluation_data(sample):
    ev = np.random.default_rng(1).normal(size=(50, 768))
    for name in T.TRANSFORMS:
        f = T.fit(name, sample)
        assert np.array_equal(f(ev), T.fit(name, sample)(ev))
        assert f(ev).shape == ev.shape


def test_centre_standardise_whiten_moments(sample):
    assert np.allclose(T.fit("centre", sample)(sample).mean(0), 0, atol=1e-10)
    S = T.fit("standardise", sample)(sample)
    assert np.allclose(S.mean(0), 0, atol=1e-10) and np.allclose(S.std(0), 1)
    W = T.fit("whiten", sample)(sample)
    assert np.allclose(np.cov(W, rowvar=False), np.eye(768), atol=1e-8)


def test_mean_corrections_unit_norm_and_direction(sample):
    mu = sample.mean(0)
    u = mu / np.linalg.norm(mu)
    Z = np.random.default_rng(2).normal(size=(40, 768)) + mu
    r1, r2 = T.fit("mean_r1", sample)(Z), T.fit("mean_r2", sample)(Z)
    assert np.allclose(np.linalg.norm(r1, axis=1), 1) and np.allclose(np.linalg.norm(r2, axis=1), 1)
    assert np.allclose(r2 @ u, 0, atol=1e-12)                      # R2 removes the mean direction
    assert not np.allclose(r1 @ u, 0, atol=1e-3)


def test_abtt_removes_top_components(sample):
    Y = T.fit("abtt", sample)(sample)
    Vt = np.linalg.svd(sample - sample.mean(0), full_matrices=False)[2]
    assert T.ABTT_M == 7
    assert np.allclose(Y @ Vt[:T.ABTT_M].T, 0, atol=1e-8)
    assert np.allclose(np.linalg.matrix_rank(Y), 768 - T.ABTT_M)


def test_zeroing_equals_deleting_after_a_transform(sample):
    """Zeroing a coordinate and deleting it give the same cosine, so the cut after a transform needs no zeroing variant."""
    rng = np.random.default_rng(3)
    f = T.fit("standardise", sample)
    a, b = f(rng.normal(size=(30, 768))), f(rng.normal(size=(30, 768)))
    order = rng.permutation(768)
    td = TaskData.from_pairs("t", a, b, rng.normal(size=30))
    drop = order[:300]
    keep = np.setdiff1d(np.arange(768), drop)
    cos_del = (a[:, keep] * b[:, keep]).sum(1) / np.sqrt((a[:, keep] ** 2).sum(1) * (b[:, keep] ** 2).sum(1))
    a0, b0 = a.copy(), b.copy()
    a0[:, drop] = 0
    b0[:, drop] = 0
    cos_zero = (a0 * b0).sum(1) / np.sqrt((a0 ** 2).sum(1) * (b0 ** 2).sum(1))
    assert np.allclose(cos_del, cos_zero, atol=1e-14)
    assert np.allclose(td.cos_nested(order, [300])[:, 0], cos_del, atol=1e-12)
