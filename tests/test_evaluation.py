import numpy as np
from scipy.stats import rankdata

from fyp.evaluation import pair_cosine, spearman


def test_pair_cosine_matches_manual():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 8)), rng.normal(size=(5, 8))
    manual = [x @ y / np.linalg.norm(x) / np.linalg.norm(y) for x, y in zip(a, b)]
    np.testing.assert_allclose(pair_cosine(a, b), manual, rtol=1e-12)


def test_pair_cosine_is_float64_from_float32_input():
    a = np.ones((2, 3), dtype=np.float32)
    assert pair_cosine(a, a).dtype == np.float64


def test_spearman_uses_average_ranks_for_ties():
    pred = np.array([0.1, 0.2, 0.2, 0.9])
    gold = np.array([1.0, 3.0, 3.0, 4.0])
    r1, r2 = rankdata(pred, "average"), rankdata(gold, "average")
    expected = np.corrcoef(r1, r2)[0, 1]
    assert abs(spearman(pred, gold) - expected) < 1e-12
