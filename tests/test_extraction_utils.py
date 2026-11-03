"""Unit tests for reused code: pooling, pair dedup, sentence normalisation, encoder inputs."""

import numpy as np
import pytest
import torch

from fyp.corpora import dedupe_pairs
from fyp.models import MODEL_SPECS
from fyp.pooling import mean_pool, mean_pool_tensor


def test_mean_pool_matches_manual_masked_mean():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(3, 5, 4, generator=g)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    got = mean_pool(h, mask)
    for i in range(3):
        n = int(mask[i].sum())
        np.testing.assert_allclose(got[i], h[i, :n].mean(0).numpy(), rtol=1e-6, atol=1e-7)


def test_mean_pool_ignores_padding_values():
    h = torch.zeros(1, 4, 2)
    h[0, :2] = 1.0
    h[0, 2:] = 1e6  # padding positions must not leak into the mean
    assert np.allclose(mean_pool(h, torch.tensor([[1, 1, 0, 0]])), [[1.0, 1.0]])


def test_mean_pool_tensor_is_the_device_generic_core_of_mean_pool():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(4, 6, 5, generator=g)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
    assert np.array_equal(mean_pool(h, mask), mean_pool_tensor(h, mask).numpy())


def test_dedupe_pairs_row_mapping():
    s1 = ["a", "b", "a", "c"]
    s2 = ["b", "c", "d", "a"]
    table, i1, i2 = dedupe_pairs(s1, s2)
    assert table == ["a", "b", "c", "d"]
    assert [table[i] for i in i1] == s1
    assert [table[i] for i in i2] == s2


def test_normalize_sentence_rule():
    from fyp.corpora import normalize_sentence
    assert normalize_sentence("  A  man\tis PLAYING ") == "a man is playing"
    assert normalize_sentence("ﬁne Ｔｅｘｔ\n\nhere") == "fine text here"   # NFKC folds ligatures and full-width forms
    assert normalize_sentence("") == ""


def test_model_specs_token_type_policy():
    assert not MODEL_SPECS["distilbert-raw"].uses_token_type_ids
    assert not MODEL_SPECS["roberta-raw"].uses_token_type_ids
    assert MODEL_SPECS["distilbert-raw"].num_hidden_states == 7
    assert all(MODEL_SPECS[k].num_hidden_states == 13 for k in MODEL_SPECS if k != "distilbert-raw")


@pytest.mark.slow
def test_distilbert_encoder_does_not_pass_token_type_ids():
    from fyp.models import load_encoder
    enc = load_encoder("distilbert-raw", "cpu")
    assert "token_type_ids" not in enc.input_names
    out = enc.encode(["a short sentence", "another one, slightly longer than the first"])
    assert out.shape == (2, 7, 768) and np.isfinite(out).all()
