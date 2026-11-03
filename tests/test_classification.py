import numpy as np

from fyp.protocol import n_remove
from fyp.classification import SAMPLES_PER_LABEL, few_shot_rows, fit_probe, kept_indices


def test_few_shot_rows_per_label_and_deterministic():
    y = np.repeat(np.arange(5), [50, 8, 7, 1, 20])
    r = few_shot_rows(y, 3)
    assert np.array_equal(r, few_shot_rows(y, 3))
    assert not np.array_equal(r, few_shot_rows(y, 4))
    assert len(set(r.tolist())) == len(r)
    counts = np.bincount(y[r], minlength=5)
    assert counts.tolist() == [SAMPLES_PER_LABEL, 8, 7, 1, SAMPLES_PER_LABEL]   # min(8, available)


def test_few_shot_rows_depend_only_on_labels_and_experiment():
    y = np.repeat(np.arange(3), 30)
    assert np.array_equal(few_shot_rows(y, 0), few_shot_rows(y.copy(), 0))


def test_kept_indices_semantics():
    n = n_remove(25)
    assert n == 192
    assert np.array_equal(kept_indices("last", n, 0), np.arange(576))
    assert np.array_equal(kept_indices("first", n, 0), np.arange(192, 768))
    r = kept_indices("random", n, 5)
    assert len(r) == 576 and len(set(r.tolist())) == 576 and r.min() >= 0 and r.max() < 768
    assert np.array_equal(r, kept_indices("random", n, 5))
    assert not np.array_equal(r, kept_indices("random", n, 6))
    # nested in K: deleting more only removes more
    assert set(kept_indices("random", n_remove(50), 5)) <= set(r)
    # K = 0 keeps everything for every method
    for m in ("last", "first", "random"):
        assert np.array_equal(kept_indices(m, 0, 2), np.arange(768))


def _toy(seed=0, n=60, d=768, classes=3):
    """Gaussian blobs; the class centres are fixed (seed 99), only the noise depends on `seed`."""
    centers = np.random.default_rng(99).normal(size=(classes, d))
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(classes), n // classes)
    return (centers[y] + 0.5 * rng.normal(size=(n, d))).astype(np.float32), y


def test_probe_separable_and_deletion_equals_zeroing():
    x, y = _toy(0)
    xe, ye = _toy(1)
    full = fit_probe(x, y, xe, ye)
    assert full["acc"] > 0.95
    assert np.isclose(full["acc"], fit_probe(x, y, xe, ye, keep=np.arange(768))["acc"])
    keep = kept_indices("random", n_remove(50), 1)
    deleted = fit_probe(x, y, xe, ye, keep=keep)
    zero = np.ones(768, dtype=bool)
    zero[keep] = False
    xz, xez = x.copy(), xe.copy()
    xz[:, zero], xez[:, zero] = 0.0, 0.0
    zeroed = fit_probe(xz, y, xez, ye)
    assert deleted["acc"] == zeroed["acc"] and np.isclose(deleted["f1_macro"], zeroed["f1_macro"])


def _fake_split(n_test, monkeypatch, with_validation):
    import fyp.classification as C
    rng = np.random.default_rng(1)
    mk = lambda n, tag: ([f"{tag}{i}" for i in range(n)], list(rng.integers(0, 3, n)))
    raw = {"train": mk(300, "tr"), "test": mk(n_test, "te"), "validation": mk(40, "va") if with_validation else None}
    monkeypatch.setattr(C, "_load_split", lambda task, split: raw[split])
    monkeypatch.setattr(C, "TEST_CAP", 50)
    monkeypatch.setattr(C, "DEV_CAP", 20)
    return C


def test_full_test_keeps_dev_and_pool_identical_to_the_capped_split(monkeypatch):
    for with_val in (True, False):
        C = _fake_split(120, monkeypatch, with_val)
        capped, full = C.load_cls_task("imdb"), C.load_cls_task("imdb", full_test=True)
        assert len(capped["test"][0]) == 50 and len(full["test"][0]) == 120
        assert set(capped["test"][0]) <= set(full["test"][0])
        assert capped["dev"][0] == full["dev"][0] and capped["pool"][0] == full["pool"][0]
        assert full["info"]["n_test"] == full["info"]["n_test_hub"] == 120


def test_full_test_equals_capped_when_the_split_is_smaller_than_the_cap(monkeypatch):
    C = _fake_split(30, monkeypatch, True)
    a, b = C.load_cls_task("imdb"), C.load_cls_task("imdb", full_test=True)
    assert a["test"][0] == b["test"][0] and np.array_equal(a["test"][1], b["test"][1])


def test_bootstrap_means_equals_the_single_multinomial_draw():
    from fyp.bootstrap import bootstrap_means, resample_counts
    ps = np.random.default_rng(0).random((5, 400))
    assert np.allclose(bootstrap_means(ps, 30, 123, chunk=7), (resample_counts(400, 30, 123) @ ps.T) / 400, atol=1e-12)
