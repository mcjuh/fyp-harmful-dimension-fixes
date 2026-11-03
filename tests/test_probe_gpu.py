"""GPU probe == sklearn LogisticRegression(C=1) optimum; masks == dropped columns; warm start; batching."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from fyp.probe_gpu import LogRegGPU

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _data(n, d, K, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(K, d)) * 1.5
    y = np.tile(np.arange(K), n // K + 1)[:n]
    x = centers[y] + rng.normal(size=(n, d)) * 2.0
    return x * 3.0 + 5.0, y            # non-centred, non-unit scale, like raw embeddings


def _obj(x, y, W, b):
    """sklearn's objective (C=1): sum CE + 0.5 ||W||^2; W is (Kp, d); Kp == 1 is the binary sigmoid form."""
    from scipy.special import logsumexp
    z = x @ W.T + b
    if W.shape[0] == 1:
        z = z[:, 0]
        return float((np.logaddexp(0, z) - y * z).sum() + 0.5 * (W ** 2).sum())
    return float((logsumexp(z, 1) - z[np.arange(len(y)), y]).sum() + 0.5 * (W ** 2).sum())


def _sk(x, y):
    return LogisticRegression(C=1.0, max_iter=20000, tol=1e-12).fit(x, y)


@pytest.mark.parametrize("K,n,d", [(2, 16, 50), (3, 24, 50), (6, 48, 60), (11, 88, 40)])
def test_matches_sklearn_optimum(K, n, d):
    x, y = _data(n, d, K, K)
    p = LogRegGPU(x, y, DEV)
    st = p.fit()
    assert st.converged.all()
    sk = _sk(x, y)
    W = st.W[0].cpu().numpy().T                         # (Kp, d)
    b = st.b[0].cpu().numpy()
    skW, skb = (sk.coef_, sk.intercept_)                # binary: (1, d) and (1,), already the sigmoid form
    # same optimum: objective values agree, and parameters to ~1e-3 (the objective is nearly flat along
    # the intercept direction, so a gradient tolerance of 1e-6 leaves ~1e-4 parameter slack)
    assert abs(_obj(x, y, W, b) - _obj(x, y, skW, skb)) <= 1e-8 * max(1.0, abs(_obj(x, y, skW, skb)))
    np.testing.assert_allclose(W, skW, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(b, skb, rtol=2e-3, atol=2e-3)
    xe, ye = _data(300, d, K, 100 + K)
    pred = p.predict(st, xe)[0].cpu().numpy()
    assert (pred == sk.predict(xe)).mean() >= 0.995


def test_mask_equals_dropped_columns_and_batch_equals_single():
    K, n, d = 5, 40, 60
    x, y = _data(n, d, K, 3)
    rng = np.random.default_rng(0)
    masks = np.stack([rng.random(d) > 0.5 for _ in range(4)])
    p = LogRegGPU(x, y, DEV)
    st = p.fit(masks)                                   # one batch of 4
    assert st.converged.all()
    for i in range(4):
        keep = np.flatnonzero(masks[i])
        sk = _sk(x[:, keep], y)
        W = st.W[i].cpu().numpy().T
        np.testing.assert_allclose(W[:, keep], sk.coef_, rtol=2e-3, atol=2e-3)
        assert np.all(W[:, ~masks[i]] == 0)             # deleted coordinates carry exactly zero weight
        np.testing.assert_allclose(st.b[i].cpu().numpy(), sk.intercept_, rtol=2e-3, atol=2e-3)
        single = p.fit(masks[i:i + 1])
        np.testing.assert_allclose(single.W[0].cpu().numpy(), st.W[i].cpu().numpy(), rtol=2e-3, atol=2e-3)


def test_warm_start_reaches_same_optimum():
    K, n, d = 6, 48, 80
    x, y = _data(n, d, K, 9)
    p = LogRegGPU(x, y, DEV)
    base = p.fit()
    masks = np.ones((d, d), dtype=bool)
    masks[np.arange(d), np.arange(d)] = False           # all single-dimension deletions
    cold = p.fit(masks)
    warm = p.fit(masks, warm=base)
    assert cold.converged.all() and warm.converged.all()
    np.testing.assert_allclose(warm.W.cpu().numpy(), cold.W.cpu().numpy(), rtol=5e-3, atol=5e-3)


def test_correct_matches_predict():
    x, y = _data(30, 40, 3, 1)
    xe, ye = _data(100, 40, 3, 2)
    p = LogRegGPU(x, y, DEV)
    st = p.fit()
    c = p.correct(st, xe, ye)[0].cpu().numpy()
    assert (c == (p.predict(st, xe)[0].cpu().numpy() == ye)).all()
